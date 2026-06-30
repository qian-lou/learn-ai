# Day 28 · 循环：用 graph 重写 ReAct

> **今日目标**：让 graph 循环执行直到满足条件，把 Day 9 手写的 ReAct 循环改写成"agent ⇄ tools"自转的图。
> **时长**：~2h ｜ **前置**：Day 9（ReAct）、Day 26–27（graph / 条件边）
> **今日产出**：一个 `day28_react.py`，graph 自己在 agent 和 tools 之间转圈，调完工具自动收尾。

## 1. 为什么 & 是什么

ReAct 的本质是一个**循环**："思考 → 要不要调工具？→ 调 → 看结果 → 再思考 …… → 够了就回答"。Day 9 你用 `while True` 手写过它，痛点是：循环条件、消息拼接、工具分发全堆在一个函数里，难调试、难加持久化。

今天把这个循环画成图：

```
        ┌──────────────────────────┐
START → │ agent (调模型，可能要工具) │
        └──────────┬───────────────┘
                   │ 条件边：要工具吗？
         有 tool_calls │           │ 没有
              ▼      │           ▼
        ┌─────────┐  │          END
        │  tools  │──┘  (工具结果回填后，边指回 agent)
        └─────────┘
```

关键就一条：**tools 节点跑完，边指回 agent**——这就形成了循环。模型下一轮看到工具结果，决定"再调一个工具"还是"可以回答了"。循环的"出口"由条件边判断：模型这次的回复里**还有没有 `tool_calls`**。

类比 Java：等价于一个 `do { 调模型; if(要调工具) 执行; } while(还要调工具)` 的循环，但每一步的状态都被框架接管、可持久化、可中断。

**两种实现，今天都要会：**
1. **手搓 graph**（理解原理）：自己写 agent 节点、tools 节点、条件边——看清循环是怎么转的。
2. **`create_react_agent` 预制件**（生产常用）：一行把上面整张图建好。

## 2. 跟着做（Hands-on）

**Step 1 — 手搓 ReAct graph（看清循环机制）**

```python
"""Day 28: 用 graph 重写 ReAct —— agent ⇄ tools 自转 / ReAct as a cyclic graph."""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode          # 预制工具执行节点 / prebuilt tool runner
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_weather(city: str) -> str:
    """查询某城市天气 / get weather for a city."""
    return f"{city}：晴，26°C"  # 假数据演示 / stub

@tool
def add(a: float, b: float) -> float:
    """两数相加 / add two numbers."""
    return a + b


tools = [get_weather, add]
# 把工具 schema 绑给模型 → 模型才知道有哪些工具可调 / bind tools so the model can call them
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def agent(state: State) -> dict:
    """agent 节点：调模型，回复里可能带 tool_calls / call the model."""
    return {"messages": [llm.invoke(state["messages"])]}


# 条件边：看最后一条 AI 消息有没有 tool_calls，决定继续还是结束
# the loop condition: does the last AI msg request tools?
def should_continue(state: State) -> Literal["tools", END]:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))   # ToolNode 自动按 tool_calls 执行并回填 ToolMessage
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")           # ★ 回边：tools 跑完回到 agent，形成循环
graph = builder.compile()


if __name__ == "__main__":
    out = graph.invoke({"messages": [HumanMessage(content="北京天气怎么样？再帮我算 3.5 + 4.5")]})
    for m in out["messages"]:
        tc = getattr(m, "tool_calls", None)
        print(f"[{m.type}] {m.content!r}" + (f"  tool_calls={[t['name'] for t in tc]}" if tc else ""))
```

运行：`python day28_react.py`。观察消息序列：human → ai(要 get_weather + add) → tool(两条结果) → ai(用结果作答)。**图自己转了一圈**，你没写任何 `while`。

**Step 2 — 用预制件，一行搞定同一张图**

```python
"""生产里通常不手搓，直接用 create_react_agent / use the prebuilt in practice."""
from langgraph.prebuilt import create_react_agent

# 2026 现代写法：model 可直接传字符串 "openai:gpt-4o-mini"
# modern: pass the model as a string id; prompt 设定系统人格
react = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt="你是简洁的中文助手，需要时调用工具，拿到结果后直接作答。",
)
out = react.invoke({"messages": [HumanMessage(content="上海天气？顺便算 10 + 15")]})
print(out["messages"][-1].content)
```

> 2026 备注：`create_react_agent` 仍是 LangGraph 官方预制件且广泛在用；新版 LangChain 还提供了 `langchain.agents.create_agent`（带 middleware 中间件体系，能更细粒度地插入 HITL、护栏等）。本系列先用 `create_react_agent` 打基础，理解它内部就是今天手搓的这张图即可。

**Step 3 — 防死循环（必做）**

```python
# 模型可能反复调工具停不下来 → 用 recursion_limit 兜底，超了直接抛错而非烧钱
# cap the loop with recursion_limit so a runaway agent fails fast instead of burning budget
out = graph.invoke(
    {"messages": [HumanMessage(content="不停查天气")]},
    config={"recursion_limit": 8},   # 超过 8 步抛 GraphRecursionError
)
```

## 3. 今日任务

1. 跑通手搓版 `day28_react.py`，**画出 `draw_ascii()`**，确认 agent↔tools 之间有回边。
2. **数循环轮数**：在 State 加 `iterations: int`，每次进 agent 自增并打印，观察一个需要连调 2 个工具的任务转了几圈。
3. **触发并捕获死循环**：给一个会让模型反复调工具的刁钻 prompt，把 `recursion_limit` 设小，确认抛 `GraphRecursionError` 而不是无限跑。
4. **对比体感**：把 Day 9 手写的 ReAct 和今天的 graph 版并排看，写 3 句话总结"显式状态机"相比裸 while 循环的好处（可观测 / 可持久化 / 可中断 任选）。

**验收标准**：手搓版能自动完成多工具任务且 `draw_ascii()` 含回边；能打印循环轮数；能触发并捕获 `GraphRecursionError`；说得清 graph 版相对 while 版的优势。

## 4. 自测清单

- [ ] 我能指出"循环"是哪条边形成的（tools → agent 的回边）。
- [ ] 我理解循环出口由"最后一条 AI 消息有无 tool_calls"决定。
- [ ] 我知道 `ToolNode` 负责按 `tool_calls` 执行并回填 `ToolMessage`。
- [ ] 我会用 `recursion_limit` 防死循环。
- [ ] 我清楚 `create_react_agent` 内部就是这张 agent⇄tools 的图。

## 5. 延伸 & 关联

- 明天：给这张会转圈的图加上**记忆/持久化**——checkpointer，让它能断点续跑、中途崩了也能从上次状态接着跑。
- 回看 Day 9–10：今天的循环 + Day 10 的错误处理，合起来就是一个健壮的单 Agent 内核。
- 本仓库相关章节：
  - LangChain Agent 与工具（ReAct 的另一种封装）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - 条件边回看 Day 27（循环本质是"指回前面"的条件边）：[./Day-27-conditional-edges.md](./Day-27-conditional-edges.md)
