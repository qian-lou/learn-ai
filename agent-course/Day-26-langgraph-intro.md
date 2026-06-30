# Day 26 · LangGraph 入门：把 Agent 想成状态机

> **今日目标**：建立 LangGraph 的核心心智模型——node / edge / state schema，亲手搭出第一个能跑的 graph。
> **时长**：~2h ｜ **前置**：Day 6–10（tool calling、ReAct）、会用 OpenAI/Claude 裸调用
> **今日产出**：一个 `day26_graph.py`，定义 state schema + 2 个节点 + 边，`invoke` 后能看到状态在节点间流转。

## 1. 为什么 & 是什么

前 25 天你写的 Agent，本质是**一个 while 循环里反复 if-else**：调模型 → 看要不要调工具 → 调工具 → 再调模型……任务一复杂，这个循环就变成意大利面——分支嵌套、状态散落在一堆局部变量里、想加"断点续跑/人工确认"几乎无从下手。

LangGraph 的答案是：**把 Agent 显式建模成一张有向图（状态机）**。

- **State（状态）**：一个贯穿全程的数据结构，所有节点读它、写它。等价于 Spring 里在 filter chain 中传递的那个 `RequestContext`——只不过这里是**唯一事实来源**。
- **Node（节点）**：一个函数 `(state) -> 局部状态更新`。等价于责任链里的一个 `Handler.handle()`，只不过它**不直接调用下一个**，只负责"干自己这块活、返回要更新的字段"。
- **Edge（边）**：决定"这个节点跑完，下一步去哪"。普通边是写死的路由；条件边（Day 27）是运行时决策。等价于状态机的转移函数。

给 Java 工程师最贴切的类比是 **Spring StateMachine / 工作流引擎**：你声明状态和转移，引擎负责驱动。区别在于 LangGraph 的"转移"可以由 LLM 在运行时决定，且整张图自带持久化（Day 29）、可中断（Day 30）。

| LangGraph | Java 类比 | 说明 |
|---|---|---|
| `StateGraph` | 工作流/状态机的 builder | 声明节点和边，最后 `compile()` 成可执行图 |
| State schema（`TypedDict`） | 贯穿流程的 `Context` DTO | 唯一事实来源，节点间靠它传数据 |
| Node 函数 | 责任链 `Handler` | 只管处理 + 返回增量，不管路由 |
| Reducer（`add_messages`） | `List.addAll()` 的合并策略 | 决定"新返回的字段"如何合并进总状态 |

**一个关键反直觉点**：节点返回的是**增量（partial update）**，不是整个新 state。框架按 reducer 把增量合并进全局 state。默认 reducer 是"覆盖"；对 `messages` 这种要"追加"的字段，用 `add_messages`。

## 2. 跟着做（Hands-on）

**Step 1 — 装包（2026 现代栈）**

```bash
pip install -U langgraph langchain "langchain-openai>=0.2"
# create_react_agent 仍在 langgraph.prebuilt；底层模型用 langchain-openai
export OPENAI_API_KEY="sk-..."   # 沿用 Day 1 的 key
```

**Step 2 — 最小可运行 graph：state + 两个节点 + 边**

```python
"""Day 26: 第一个 LangGraph —— 状态在节点间流转 / first StateGraph."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


# State schema：唯一事实来源 / the single source of truth
# messages 用 add_messages reducer → 节点返回的消息会被"追加"而非"覆盖"
# messages uses the add_messages reducer → returned msgs are appended, not replaced
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史 / chat history
    step: int                                            # 普通字段，默认覆盖 / plain field, overwritten


# 节点 = (state) -> 局部状态更新 / a node is (state) -> partial update
def greet(state: State) -> dict:
    """节点 1：根据最新用户消息生成一条回应 / node 1: produce a reply."""
    last: str = state["messages"][-1].content
    reply = AIMessage(content=f"收到你说的：{last}")
    # 只返回"要更新的字段"，不是整个 state / return ONLY the delta
    return {"messages": [reply], "step": state.get("step", 0) + 1}


def finalize(state: State) -> dict:
    """节点 2：收尾，打个标记 / node 2: wrap up."""
    return {"messages": [AIMessage(content="（流程结束）")], "step": state["step"] + 1}


# 组图：声明节点 → 连边 → 编译 / declare nodes -> wire edges -> compile
builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("finalize", finalize)
builder.add_edge(START, "greet")       # 入口 / entry
builder.add_edge("greet", "finalize")  # 普通边：写死的路由 / static routing
builder.add_edge("finalize", END)      # 出口 / exit
graph = builder.compile()


if __name__ == "__main__":
    # invoke 传入"初始状态"，拿回"最终状态" / invoke: initial state in, final state out
    result = graph.invoke({"messages": [HumanMessage(content="你好，LangGraph")], "step": 0})
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
    print("step =", result["step"])  # 期望 2 / expect 2
```

运行：`python day26_graph.py`。你会看到 user→greet→finalize 三条消息按序出现，`step` 累加到 2——**状态确实在节点间流过并被合并**。

**Step 3 — 看清"图长什么样"（强烈建议）**

```python
# 打印 ASCII 结构图，或导出 PNG（需 graphviz/mermaid）
print(graph.get_graph().draw_ascii())
# graph.get_graph().draw_mermaid_png()  # 存成图片更直观
```

> 调试心法：以后每搭一个 graph，先 `draw_ascii()` 确认拓扑对不对，再跑逻辑。图画错了，逻辑再对也白搭。

## 3. 今日任务

1. 跑通 `day26_graph.py`，确认三条消息按序输出、`step == 2`。
2. **验证 reducer 语义**：把 `greet` 的返回从 `{"messages": [reply]}` 改成 `{"messages": reply}`（去掉 list），观察报错；再把 `step` 字段试着返回字符串，体会"默认覆盖"和类型约束。
3. **加第三个节点** `route_log`：插在 `greet` 和 `finalize` 之间，往 messages 追加一条"已记录"，重新 `draw_ascii()` 确认拓扑变了。
4. **接真模型**：把 `greet` 里写死的 reply 换成真实 LLM 调用（用 `langchain_openai.ChatOpenAI(model="gpt-4o-mini").invoke(state["messages"])`），让节点真正"思考"。

**验收标准**：能解释"节点返回增量、reducer 负责合并"；改坏 reducer 能看懂报错；新增节点后 `draw_ascii()` 反映出新拓扑；至少一个节点接上了真实模型。

## 4. 自测清单

- [ ] 我能用一句话说清 node / edge / state 各自的职责。
- [ ] 我知道节点返回的是**增量**，不是整个 state。
- [ ] 我理解 `add_messages` 这个 reducer 解决的是"追加 vs 覆盖"的问题。
- [ ] 我能把"以前 while 循环里的 Agent"和"现在的图"对应起来。
- [ ] 我会用 `draw_ascii()` 先验证拓扑再跑逻辑。

## 5. 延伸 & 关联

- 明天：让边"会思考"——条件分支 conditional edges，根据 state 决定走哪条路。
- 回顾你 Day 9 手写的 ReAct 循环——Day 28 会用 graph 把它重写一遍，对比体会"显式状态机"的好处。
- 本仓库相关章节：
  - LangChain Agent 与工具（图的节点里常放这种 Agent）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - LangChain 基础（LCEL / Runnable，和 graph 是两种编排思路）：[../07-llm-applications/05-langchain/01-langchain-basics.md](../07-llm-applications/05-langchain/01-langchain-basics.md)
  - 本系列总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
