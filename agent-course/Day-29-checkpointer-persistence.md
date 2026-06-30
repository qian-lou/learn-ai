# Day 29 · 状态持久化：checkpointer 与断点续跑

> **今日目标**：用 checkpointer 把 graph 状态落盘，实现"崩了/重启后从上次断点接着跑"，并理解 `thread_id`。
> **时长**：~2h ｜ **前置**：Day 28（带循环的 graph）
> **今日产出**：一个 `day29_checkpoint.py`，同一个 `thread_id` 跨多次 `invoke` 记得上文；进程重启后用 SQLite 仍能续跑。

## 1. 为什么 & 是什么

Day 28 的 graph 每次 `invoke` 都是**从零开始**——上一轮的对话、中间状态全没了。生产里这不可接受：长任务可能跑几分钟、要人工确认（Day 30）、机器可能重启。我们需要**把每一步的状态持久化**。

LangGraph 的机制叫 **checkpointer（检查点存储）**：

- 每个节点执行后，框架自动把当前完整 state **存一个 checkpoint**。
- 下次用**同一个 `thread_id`** 调用时，框架先把最新 checkpoint **加载回来**，从断点继续。
- 这同时给了你三样东西：**会话记忆**（多轮对话）、**断点续跑**（崩溃恢复）、**可中断**（Day 30 HITL 的地基）。

类比 Java 再贴切不过：

| LangGraph | Java / 后端类比 | 说明 |
|---|---|---|
| checkpointer | Spring Session / 持久化的 HttpSession store | 把"会话状态"存进外部存储 |
| `thread_id` | sessionId / 会话主键 | 同一个 id = 同一条会话/任务线 |
| checkpoint | 一次状态快照（类似事务日志的一条记录） | 节点级粒度，可回溯 |
| `InMemorySaver` | 内存 session（重启即丢） | 仅开发用 |
| `SqliteSaver` / `PostgresSaver` | 落库的 session（Redis/JDBC session store） | 生产用，重启不丢 |

**核心反直觉点**：持久化的粒度是**每个节点之后**，不是"整个任务结束后"。所以哪怕任务跑到一半进程挂了，重启后也能从最后一个成功节点接着跑，而不是从头来。

## 2. 跟着做（Hands-on）

**Step 1 — 内存 checkpointer：先看"记得上文"**

```python
"""Day 29: checkpointer 让 graph 跨调用记忆 / persistence across invokes."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver   # 开发用内存存储 / dev-only
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chat(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ★ 编译时挂上 checkpointer / attach the checkpointer at compile time
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    # thread_id 标识"同一条会话"，必须放进 config / thread_id ties calls to one thread
    cfg = {"configurable": {"thread_id": "user-42"}}

    r1 = graph.invoke({"messages": [HumanMessage(content="我叫小李，记住。")]}, cfg)
    print("第1轮:", r1["messages"][-1].content)

    # 第2轮没重发"我叫小李"，但因为同一 thread_id，模型仍记得 / it remembers
    r2 = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, cfg)
    print("第2轮:", r2["messages"][-1].content)   # 期望答出"小李" / expect "小李"

    # 换 thread_id = 全新会话，问不出名字 / a different thread = fresh session
    r3 = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]},
                      {"configurable": {"thread_id": "user-99"}})
    print("新会话:", r3["messages"][-1].content)   # 期望"不知道" / expect "I don't know"
```

运行：`python day29_checkpoint.py`。`user-42` 跨两轮记得名字；`user-99` 是干净的新会话——**这就是会话隔离**。

**Step 2 — SQLite checkpointer：进程重启也不丢**

```bash
pip install -U langgraph-checkpoint-sqlite   # SQLite 存储是独立包 / separate package
```

```python
"""把存储换成 SQLite —— 落盘，进程重启后仍能续跑 / SQLite-backed, survives restarts."""
from langgraph.checkpoint.sqlite import SqliteSaver

# from_conn_string 返回上下文管理器；:memory: 换成文件名即落盘
# yields a context manager; use a file path (not :memory:) to persist to disk
with SqliteSaver.from_conn_string("day29_checkpoints.sqlite") as cp:
    graph = builder.compile(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "user-42"}}
    graph.invoke({"messages": [HumanMessage(content="把我的项目代号设为 Phoenix")]}, cfg)

# ……即使这里进程退出再重启，只要还连同一个 .sqlite + 同一 thread_id，状态仍在 ……
with SqliteSaver.from_conn_string("day29_checkpoints.sqlite") as cp:
    graph = builder.compile(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "user-42"}}
    r = graph.invoke({"messages": [HumanMessage(content="我的项目代号是什么？")]}, cfg)
    print(r["messages"][-1].content)   # 期望"Phoenix" / expect "Phoenix"
```

> Postgres 版几乎一样：`from langgraph.checkpoint.postgres import PostgresSaver`，包名 `langgraph-checkpoint-postgres`，**首次用要显式 `checkpointer.setup()`** 建表。生产推荐 Postgres——你 Java 那边大概率已经有一个 PG 实例，直接复用。

**Step 3 — 查看/回溯历史 checkpoint**

```python
# 拿到某 thread 的当前状态快照 / inspect current state of a thread
snap = graph.get_state(cfg)
print("当前消息数:", len(snap.values["messages"]))
# 遍历历史 checkpoint（可做"时间旅行"调试，回到任意一步）
for st in graph.get_state_history(cfg):
    print("checkpoint:", st.config["configurable"]["checkpoint_id"], "| next=", st.next)
```

## 3. 今日任务

1. 跑通内存版，确认 `user-42` 记得名字、`user-99` 不记得（会话隔离）。
2. **真做一次断点续跑**：用 SQLite 版，第一次进程写入一个事实并退出；**重新运行**脚本（新进程）只问、不重发那个事实，确认仍答得出——这才是"重启后续跑"。
3. **看历史**：用 `get_state_history` 打印某 thread 的 checkpoint 列表，理解"每个节点后存一份"的粒度。
4. **接上你的 ReAct graph**：把 Day 28 那张 agent⇄tools 的图也挂上 checkpointer，验证多轮工具调用的中间状态也被持久化。

**验收标准**：会话隔离正确；**新进程**能凭 SQLite 续上旧 thread 的记忆；能列出历史 checkpoint；ReAct graph 加持久化后多轮仍连贯。

## 4. 自测清单

- [ ] 我能用 Spring Session / sessionId 的类比解释 checkpointer 和 `thread_id`。
- [ ] 我知道持久化粒度是"每个节点之后"，不是任务结束后。
- [ ] 我能说清 `InMemorySaver` 与 `SqliteSaver`/`PostgresSaver` 的适用场景。
- [ ] 我记得 Postgres 首次要 `setup()`，SQLite/Postgres 是独立安装包。
- [ ] 我会用 `get_state` / `get_state_history` 查看和回溯状态。

## 5. 延伸 & 关联

- 明天：checkpointer 是 **HITL（人在回路）** 的地基——正因为状态能存下来，graph 才能"暂停等人确认、人点了头再从断点继续"。
- 回看 Day 11（记忆与会话）：checkpointer 就是把你当时手动维护的 history 升级成了框架级、可落库的记忆。
- 本仓库相关章节：
  - 模型服务/部署（持久化存储常和服务一起部署）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
  - Day 11 记忆与会话状态铺垫：[./Day-11-memory-and-context.md](./Day-11-memory-and-context.md)
