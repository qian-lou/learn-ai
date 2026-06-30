# Day 30 · 人在回路（HITL）：暂停、等确认、再继续

> **今日目标**：用 `interrupt()` 让 graph 在关键步骤暂停，把控制权交给人，确认后用 `Command(resume=...)` 从断点继续。
> **时长**：~2h ｜ **前置**：Day 29（checkpointer，HITL 的硬前提）
> **今日产出**：一个 `day30_hitl.py`，执行"危险操作"前暂停等人批准；批准则执行，否决则跳过。

## 1. 为什么 & 是什么

Agent 越自动，越需要**刹车**。让 LLM 自动发邮件、改数据库、下订单——出错代价很高。**人在回路（Human-in-the-Loop, HITL）** 就是在关键步骤插一道闸：**Agent 跑到这里停下，把"我打算干啥"亮给人看，人点头才继续，摇头就改道。**

LangGraph 把这件事做成了一个函数：**`interrupt(payload)`**。

- 某个节点里调 `interrupt(要给人看的信息)` → 图**立刻暂停**，把当前完整状态 checkpoint 下来，并把 `payload` 抛回给调用方。
- 调用方（你的 UI / API）拿到 payload，展示给人，收集决定。
- 人决定后，用 **`Command(resume=人的输入)`** 再次 `invoke`/`stream` → 图**从那个 `interrupt` 处原地复活**，`interrupt()` 的返回值就是人填的内容，继续往下跑。

**硬前提**：必须配 checkpointer（Day 29）。没有持久化，"暂停"之后状态无处可存，自然无法复活。

类比 Java：这非常像**审批工作流（BPM）里的人工任务节点**——流程引擎跑到"经理审批"就挂起、落库、等外部事件（审批通过/驳回）再推进。也像断点调试里的 breakpoint：停在这、看一眼、决定放行还是改值。

| HITL 概念 | Java / 工作流类比 | 说明 |
|---|---|---|
| `interrupt(payload)` | BPM 的人工任务 / 断点 | 暂停并把"待办信息"抛出 |
| checkpoint | 流程实例落库 | 暂停期间状态存在哪 |
| `Command(resume=x)` | 提交审批结果，唤醒流程 | 用人的输入复活并继续 |
| `thread_id` | 流程实例 id | 定位是哪一次暂停 |

## 2. 跟着做（Hands-on）

**Step 1 — 在"危险操作"前插入人工确认**

```python
"""Day 30: HITL —— 执行前暂停等人批准 / pause for human approval before acting."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command         # ★ HITL 核心 API / core HITL API
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: str   # 待批准的草稿动作 / the action awaiting approval


def propose(state: State) -> dict:
    """节点：拟一个"要发的邮件"草稿（真实场景可由 LLM 生成）。"""
    return {"draft": "发邮件给 boss@corp.com，主题：季度报告已完成"}


def human_gate(state: State) -> dict:
    """HITL 节点：暂停，把草稿交给人审批 / pause and ask a human to approve.

    interrupt() 抛出 payload 后图会停在这里；resume 的值就是它的返回值。
    interrupt() throws the payload and pauses; the resume value becomes its return.
    """
    decision = interrupt({                       # ← 图在此暂停 / graph PAUSES here
        "action": state["draft"],
        "question": "批准执行吗？回 approve / reject",
    })
    # —— 下面这行只有在 Command(resume=...) 之后才会执行 —— / runs only after resume
    if str(decision).strip().lower() == "approve":
        return {"messages": [AIMessage(content=f"✅ 已执行：{state['draft']}")]}
    return {"messages": [AIMessage(content="🚫 已取消，未执行任何操作")]}


builder = StateGraph(State)
builder.add_node("propose", propose)
builder.add_node("human_gate", human_gate)
builder.add_edge(START, "propose")
builder.add_edge("propose", "human_gate")
builder.add_edge("human_gate", END)
# ★ 必须配 checkpointer，否则无法暂停/恢复 / checkpointer is mandatory for interrupt
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "approval-1"}}

    # 第一次跑：会在 human_gate 暂停，返回里带 __interrupt__ / first run pauses
    result = graph.invoke({"messages": [HumanMessage(content="发个汇报邮件")], "draft": ""}, cfg)
    print("暂停了，待办内容：", result["__interrupt__"][0].value)

    # 模拟人做决定（真实里来自前端按钮）/ simulate the human decision
    human_choice = input("你的决定 approve/reject > ").strip() or "approve"

    # 用 Command(resume=...) 复活，从 interrupt 处继续 / resume from the interrupt
    final = graph.invoke(Command(resume=human_choice), cfg)
    print("最终结果：", final["messages"][-1].content)
```

运行：`python day30_hitl.py`。第一次 `invoke` 不会执行操作，而是停在 `human_gate` 并打印待办；你输入 `approve`/`reject` 后，第二次 `invoke(Command(resume=...))` 才决定执行还是取消。

**Step 2 — 看清"暂停信号"长什么样**

```python
# 用 stream 跑能直接看到 __interrupt__ 事件 / streaming surfaces the interrupt event
for event in graph.stream({"messages": [HumanMessage(content="发邮件")], "draft": ""}, cfg,
                          stream_mode="updates"):
    if "__interrupt__" in event:
        print("命中中断，等待人工：", event["__interrupt__"][0].value)
```

> 关键认知：`interrupt()` **不是抛异常崩掉**，而是"优雅暂停"。状态已 checkpoint，进程哪怕退出，只要同 `thread_id` + 同 checkpointer，之后照样能 `Command(resume=...)` 复活。这就是它和"普通 `input()` 阻塞"的本质区别——后者一断电就全没了。

**Step 3 — 给 ReAct Agent 的工具加批准闸（更实用）**

```python
"""在工具执行前 interrupt：让"写操作类工具"必须人工放行 / gate write-tools."""
from langchain_core.tools import tool

@tool
def delete_records(table: str) -> str:
    """删除某张表的数据（危险操作）/ delete rows (dangerous)."""
    ok = interrupt({"confirm": f"将清空表 {table}，确认？", "options": ["yes", "no"]})
    if str(ok).lower() != "yes":
        return "已取消，未删除任何数据"
    return f"已清空表 {table}"   # 真实场景这里才执行 DB 操作 / real DB call goes here
# 把这个 tool 放进 create_react_agent(tools=[...], checkpointer=...) 即可获得"工具级 HITL"
```

## 3. 今日任务

1. 跑通 `day30_hitl.py`：分别测 `approve` 和 `reject`，确认只有 approve 才"执行"。
2. **验证真·可恢复**：用 SQLite checkpointer（Day 29）替换 `InMemorySaver`；第一次跑到暂停就**退出进程**，重新运行脚本只做 `Command(resume="approve")`，确认仍能复活并完成——证明暂停跨进程也活着。
3. **改成"修改后放行"**：让人不仅能 approve/reject，还能**返回修改后的草稿**（`Command(resume="改成发给 cto@corp.com")`），节点用人的输入覆盖 `draft` 再执行。
4. **工具级 HITL**：把 `delete_records` 接进一个 `create_react_agent`，让 Agent 想删数据时必须停下等你点头。

**验收标准**：approve/reject 行为正确；用 SQLite 时暂停能跨进程恢复；支持"人工修改后再执行";工具级批准闸生效。

## 4. 自测清单

- [ ] 我能解释 `interrupt()` 是"优雅暂停 + checkpoint"，不是抛异常。
- [ ] 我知道 HITL 必须配 checkpointer，且能说出为什么。
- [ ] 我会用 `Command(resume=x)` 复活图，并知道 `x` 会成为 `interrupt()` 的返回值。
- [ ] 我能从 `__interrupt__` 里取出待人审批的 payload。
- [ ] 我能给"写操作类工具"加一道人工放行闸。

## 5. 延伸 & 关联

- 这是 Phase 3 单 Agent 编排的收尾：你现在能搭**有状态、会循环、能持久化、可人工干预**的 graph 了。明天起进入**多 Agent**。
- HITL 在 Day 41–45 的"自动化研究 Agent"里会再用：报告发布前让人审一眼。
- 新版 LangChain 还提供 `HumanInTheLoopMiddleware`（配合 `create_agent`），声明式地给工具加审批，原理同今天的 `interrupt`。
- 本仓库相关章节：
  - 安全与输出校验（HITL 是"人肉护栏"，后面还有自动护栏）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
  - 持久化前置回看 Day 29：[./Day-29-checkpointer-persistence.md](./Day-29-checkpointer-persistence.md)
