# Day 33 · Agent 间通信与共享状态：怎么传、怎么不污染

> **今日目标**：掌握多 Agent 之间传递中间结果的正确姿势，用 reducer / 私有字段 / 子图隔离避免状态污染。
> **时长**：~2h ｜ **前置**：Day 32（多 Agent 流水线、共享 state）
> **今日产出**：一个 `day33_state.py`，用并行分支 + reducer 合并结果，演示"污染版"与"干净版"的差别。

## 1. 为什么 & 是什么

多 Agent 真正的难点不是"怎么连起来"，而是**怎么共享状态而不互相搞坏**。典型翻车现场：

- **消息污染**：把子 Agent 内部的思考、工具调用消息整团塞进主 `messages`，下游 Agent 被一堆噪音带偏。
- **字段覆盖**：两个 Agent 都往同一个字段写，后写的把先写的冲掉（默认 reducer 是"覆盖"）。
- **并发竞争**：并行的 Agent 同时写同一个 list，结果丢数据或顺序乱。
- **tool_calls 配对断裂**：handoff 时把"模型发起的 tool_call"和"工具返回的 ToolMessage"拆散，下个 Agent 看到残缺对话直接报错或幻觉。

LangGraph 提供三层武器，按"侵入性从低到高"排：

| 手段 | 解决什么 | Java 类比 |
|---|---|---|
| **Reducer**（如 `add_messages` / 自定义合并函数） | 多方写同一字段时如何合并（追加？去重？取最大？） | `ConcurrentHashMap.merge()` / `Collectors` 的合并器 |
| **私有/分区字段** | 给每个 Agent 划自己的工作区，不踩别人 | 每个线程一个 `ThreadLocal` / 各自的 DTO 字段 |
| **子图（subgraph）隔离** | 子 Agent 在自己的 state schema 里折腾，只把结论"映射"回父图 | 微服务边界 / 防腐层（ACL），只暴露 DTO |

核心原则一句话：**共享要显式、隔离是默认**——明确哪些字段是"公共总线"，其余都关在各自房间里，只在边界上交换"成品"。

## 2. 跟着做（Hands-on）

**Step 1 — 自定义 reducer：让两个 Agent 安全地写同一个列表**

```python
"""Day 33: 自定义 reducer 合并多 Agent 的产出 / safe merge via reducers."""

from typing import Annotated
from typing_extensions import TypedDict
from operator import add        # 内置：list 相加=拼接，可直接当 reducer / built-in append reducer

from langgraph.graph import StateGraph, START, END


# findings 用 operator.add 作 reducer → 多个节点各 append 自己的发现，不互相覆盖
# findings uses `add` as reducer → each node appends; nothing gets overwritten
class State(TypedDict):
    topic: str
    findings: Annotated[list[str], add]   # 公共总线：可被多方追加 / shared bus, append-only
    summary: str


def source_a(state: State) -> dict:
    return {"findings": [f"[来源A] 关于 {state['topic']}：数据点 1"]}  # 只 append 一条

def source_b(state: State) -> dict:
    return {"findings": [f"[来源B] 关于 {state['topic']}：数据点 2"]}  # 也只 append 一条

def merge(state: State) -> dict:
    # 两个来源的 findings 已被 reducer 安全合并，这里只读不冲突 / safely merged by reducer
    return {"summary": "汇总：\n" + "\n".join(state["findings"])}


builder = StateGraph(State)
builder.add_node("source_a", source_a)
builder.add_node("source_b", source_b)
builder.add_node("merge", merge)

# 并行：A 和 B 同时从 START 出发（fan-out），都连到 merge（fan-in）
# fan-out to A & B in parallel, fan-in to merge
builder.add_edge(START, "source_a")
builder.add_edge(START, "source_b")
builder.add_edge("source_a", "merge")
builder.add_edge("source_b", "merge")
builder.add_edge("merge", END)
graph = builder.compile()

if __name__ == "__main__":
    out = graph.invoke({"topic": "向量数据库选型", "findings": [], "summary": ""})
    print(out["summary"])          # A、B 两条都在，没有互相覆盖 / both kept, no clobber
    assert len(out["findings"]) == 2, "reducer 失效则这里会丢数据"
```

运行：`python day33_state.py`。A、B 并行写 `findings`，靠 `add` reducer 安全拼接——**换成默认 reducer，后跑的会覆盖先跑的，只剩一条**。把 `Annotated[list[str], add]` 改成普通 `list[str]` 跑一次，亲眼看它丢数据。

**Step 2 — 反面教材 vs 正确做法：子 Agent 结果回填**

```python
"""污染版 vs 干净版：怎么把子 Agent 的结果并入主 state / dirty vs clean handback."""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def lookup(q: str) -> str:
    """查询（桩）/ stub."""
    return f"{q} 的答案是 42"

sub_agent = create_react_agent(model="openai:gpt-4o-mini", tools=[lookup],
                               prompt="你是助手，用 lookup 查询后回答。")

def node_dirty(state) -> dict:
    """❌ 反面：把子 Agent 的全部内部消息灌进主 messages（污染）。"""
    sub = sub_agent.invoke({"messages": [HumanMessage(content="查一下 X")]})
    return {"messages": sub["messages"]}        # 思考+tool_call+ToolMessage 全进来了，噪音！

def node_clean(state) -> dict:
    """✅ 正面：只提取子 Agent 的最终结论，包成一条干净消息。"""
    sub = sub_agent.invoke({"messages": [HumanMessage(content="查一下 X")]})
    final = sub["messages"][-1].content          # 只要结论 / extract just the conclusion
    return {"messages": [AIMessage(content=f"[子任务结论] {final}")]}
```

> 心法：把每个子 Agent 当**微服务**。它内部怎么调工具、想了几轮，是它的私事；对外只暴露一个干净的"返回 DTO"。父图的共享 `messages` 是"对外 API"，别把实现细节泄漏进去。

**Step 3 — 私有字段：给 Agent 划独立工作区**

```python
# 给每个 Agent 一块只有它读写的字段，公共总线只放"已定稿"的产出
# private scratch fields per agent; the shared bus holds only finalized output
class CleanState(TypedDict):
    shared_summary: str                 # 公共总线：只放定稿 / shared, finalized only
    _researcher_scratch: str            # 研究员私有草稿区 / private to researcher
    _analyst_scratch: str               # 分析师私有草稿区 / private to analyst
# 约定：下划线前缀=私有，其他 Agent 不读它 / convention: leading underscore = private
```

## 3. 今日任务

1. 跑通 `day33_state.py`，确认并行写入靠 reducer 安全合并（`findings` 有 2 条）。
2. **复现污染**：把 reducer 去掉（用普通 `list`），观察 `findings` 丢成 1 条；再换个**去重 reducer**（合并时去重）验证你能定制合并策略。
3. **对比两种回填**：分别用 `node_dirty` 和 `node_clean` 接子 Agent，打印主 `messages`，直观对比噪音量。
4. **加并发字段防护**：构造两个 Agent 同时写一个非 list 字段的场景，体会默认覆盖的坑，再用合适的 reducer 或拆成私有字段修好。

**验收标准**：reducer 合并正确且能换去重策略；能复现并解释"丢数据";`node_clean` 的主 messages 明显比 `node_dirty` 干净；理解并能修复并发写冲突。

## 4. 自测清单

- [ ] 我能列出多 Agent 状态污染的至少 3 种形态。
- [ ] 我理解 reducer 的作用，会写/换一个自定义合并函数。
- [ ] 我知道"把子 Agent 当微服务、只暴露返回 DTO"的隔离原则。
- [ ] 我会用私有字段 / 子图给 Agent 划独立工作区。
- [ ] 我清楚 handoff 时为什么不能拆散 tool_call 与 ToolMessage。

## 5. 延伸 & 关联

- 明天：换个角度看多 Agent——用 **CrewAI** 这种"角色化框架"快速搭，对比 LangGraph 的取舍。
- 状态污染的坑会一直伴随你到 Day 41–45 的主力项目；今天的隔离原则就是那时候的护身符。
- 本仓库相关章节：
  - 结构化输出（Agent 间交换"成品"最好是强类型 DTO）回看 Day 4：[./Day-04-structured-output.md](./Day-04-structured-output.md)
  - 多 Agent 流水线回看 Day 32：[./Day-32-multi-agent-pipeline.md](./Day-32-multi-agent-pipeline.md)
