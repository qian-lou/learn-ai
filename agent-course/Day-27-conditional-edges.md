# Day 27 · 条件分支：让边会思考

> **今日目标**：掌握 conditional edges——根据运行时 state 决定走哪条边，搭出第一个"会分流"的 graph。
> **时长**：~2h ｜ **前置**：Day 26（StateGraph / node / edge / state）
> **今日产出**：一个 `day27_router.py`，用一个路由函数把请求分到「数学/查询/闲聊」三条不同支路。

## 1. 为什么 & 是什么

Day 26 的边是**写死的**：`greet` 跑完一定去 `finalize`。但真正的 Agent 需要**分支**——"如果模型要调工具就去 tools 节点，否则直接结束"。这正是 ReAct 循环的灵魂。

LangGraph 用 **conditional edges（条件边）** 表达这种"运行时决策"：

- 你给某个节点挂一个**路由函数** `(state) -> str`，它读当前 state，返回**下一个节点的名字**（或一个 key）。
- 框架拿这个返回值去查一张 `{key: 目标节点}` 映射表，跳过去。

类比 Java：这就是责任链里"由当前 Handler 决定下一跳"的版本，或者更贴切——**Spring StateMachine 的 `choice` 伪状态 / `guard` 守卫**。守卫返回什么，状态机就往哪转。区别是这里的"守卫"可以是一次 LLM 调用的结果。

| 概念 | 写死的边 (Day 26) | 条件边 (今天) |
|---|---|---|
| API | `add_edge(a, b)` | `add_conditional_edges(a, router_fn, {...})` |
| 路由依据 | 无，固定 | `router_fn(state)` 的返回值 |
| Java 类比 | 顺序流 | `if/switch` / StateMachine guard |
| 典型用途 | 串联固定步骤 | "要不要调工具""走哪个专家""是否重试" |

**关键点**：路由函数本身**不修改 state**，只"看一眼然后说去哪"。它要快、要确定性强——别在路由函数里干重活。

## 2. 跟着做（Hands-on）

**Step 1 — 一个三向分流路由**

```python
"""Day 27: 条件边 —— 按意图分流到不同支路 / conditional edges routing."""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str  # 由分类节点写入 / written by the classifier node


def classify(state: State) -> dict:
    """分类节点：判断用户意图（这里用关键词，真实场景可换成 LLM）。

    Classifier node: decide intent (keyword rule here; swap for an LLM in prod).
    时间 O(L) L=文本长度 空间 O(1)
    """
    text: str = state["messages"][-1].content.lower()
    if any(c in text for c in "+-*/0123456789"):
        intent = "math"
    elif any(k in text for k in ("查", "数据", "select", "查询")):
        intent = "query"
    else:
        intent = "chat"
    return {"intent": intent}  # 只写 intent，不动 messages / write intent only


# 路由函数：读 state.intent，返回"下一个节点名" / read state, return next node name
# 注意：它不改 state，只做决策 / it does NOT mutate state, only decides
def route(state: State) -> Literal["do_math", "do_query", "do_chat"]:
    return {"math": "do_math", "query": "do_query", "chat": "do_chat"}[state["intent"]]


def do_math(state: State) -> dict:
    return {"messages": [AIMessage(content="[数学支路] 我来算算……")]}

def do_query(state: State) -> dict:
    return {"messages": [AIMessage(content="[查询支路] 我去查数据……")]}

def do_chat(state: State) -> dict:
    return {"messages": [AIMessage(content="[闲聊支路] 我们随便聊聊~")]}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("do_math", do_math)
builder.add_node("do_query", do_query)
builder.add_node("do_chat", do_chat)

builder.add_edge(START, "classify")
# 核心：条件边。第 3 个参数是 {路由返回值: 目标节点} 的映射表
# the conditional edge: 3rd arg maps router's return value -> target node
builder.add_conditional_edges(
    "classify",
    route,
    {"do_math": "do_math", "do_query": "do_query", "do_chat": "do_chat"},
)
# 三条支路跑完都收口到 END / all three branches converge to END
for node in ("do_math", "do_query", "do_chat"):
    builder.add_edge(node, END)

graph = builder.compile()


if __name__ == "__main__":
    for q in ["帮我算 12 * 8", "查一下昨天的订单数据", "今天心情不错"]:
        out = graph.invoke({"messages": [HumanMessage(content=q)], "intent": ""})
        print(f"Q: {q}\n→ intent={out['intent']} | {out['messages'][-1].content}\n")
```

运行：`python day27_router.py`。三个问题应分别走进 math / query / chat 三条支路。

**Step 2 — 把路由依据换成 LLM（更接近真实）**

```python
# 让模型做分类：返回严格枚举，避免自由发挥 / let the LLM classify into a strict enum
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 分类要确定性 → temperature=0

def classify_llm(state: State) -> dict:
    prompt = (
        "把用户意图分类为 math / query / chat 之一，只输出这一个词。\n"
        f"用户：{state['messages'][-1].content}"
    )
    intent = _llm.invoke(prompt).content.strip().lower()
    # 防御：模型偶尔越界，兜底成 chat / defensive fallback
    return {"intent": intent if intent in {"math", "query", "chat"} else "chat"}
```

> 这里藏着 Agent 工程的高频坑：**让 LLM 当路由器时，输出必须收敛成有限枚举**。永远要兜底，不然某次它回了句"嗯这个嘛"，你的映射表就 KeyError 了。

## 3. 今日任务

1. 跑通 `day27_router.py`，确认三向分流正确。
2. **加默认分支**：当 `intent` 是预期外的值时，路由到一个 `fallback` 节点而不是崩。改造 `route` 或 `classify`，并构造一个走进 fallback 的输入验证。
3. **换成 LLM 路由**：用 `classify_llm` 替换关键词版，测几个边界输入（"1 加 1 等于几"应进 math，"你好呀"应进 chat）。
4. **回字图**：让 `do_query` 跑完不直接去 END，而是再回到 `classify` 重新判断一次（先体会"边能指回前面的节点"，为明天的循环铺路；记得加个计数防死循环）。

**验收标准**：三向分流正确；非法 intent 能进 fallback 不崩；LLM 路由对边界输入分类合理；理解"条件边的第三个参数是映射表"。

## 4. 自测清单

- [ ] 我能解释路由函数与普通节点的区别（一个只决策、一个改状态）。
- [ ] 我知道 `add_conditional_edges` 第三个参数是 `{返回值: 目标节点}` 映射。
- [ ] 我明白用 LLM 做路由时为什么必须收敛枚举 + 兜底。
- [ ] 我能让多条支路收口到同一个节点（或 END）。
- [ ] 我看得出"边可以指回前面的节点"——这就是循环的前身。

## 5. 延伸 & 关联

- 明天：把"边指回前面"正式用起来——用 graph 重写 ReAct 循环，让图自己转圈直到任务完成。
- 条件路由是后面**多 Agent supervisor**（Day 31–32）的基础：supervisor 本质就是一个"决定把活派给哪个专家"的超级路由节点。
- 本仓库相关章节：
  - Prompt 进阶（分类/路由 prompt 怎么写得稳）：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
  - 结构化输出（让分类结果强类型化）回看 Day 4：[./Day-04-structured-output.md](./Day-04-structured-output.md)
