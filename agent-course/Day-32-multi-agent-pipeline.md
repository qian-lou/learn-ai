# Day 32 · 搭多 Agent 流水线：研究员 → 分析师 → 报告

> **今日目标**：用 LangGraph 搭一条三段式多 Agent 流水线（研究员→分析师→报告生成），跑通端到端。
> **时长**：~2h ｜ **前置**：Day 26–31（StateGraph、节点即 Agent、多 Agent 范式）
> **今日产出**：一个 `day32_pipeline.py`，输入一个主题，自动产出一份带数据支撑的简报。

## 1. 为什么 & 是什么

今天是把前面零件**组装成一个真东西**的第一天。我们用最直观的 **Role/流水线范式**：三个角色固定顺序协作。

```
START → [研究员 researcher] → [分析师 analyst] → [报告生成 writer] → END
         产出：原始资料        产出：结构化洞察      产出：成文简报
```

为什么用流水线而不是 supervisor？因为这个任务**步骤是确定的**——先有料、再分析、最后成文，顺序不会变。这种情况下，固定流水线比"每步都问 supervisor 派给谁"更简单、更省 token、更好调试。**范式跟着任务走，别为了用 supervisor 而用 supervisor。**

每个节点其实是一个**子 Agent**（可以是 `create_react_agent`，也可以是一次专门的 LLM 调用）。它们靠**共享 state** 接力：研究员把资料写进 state，分析师从 state 读资料、写洞察，报告员读洞察、写成稿。

类比 Java：这就是一条**装配流水线 / 责任链**——每个工位（节点）从传送带（state）上取上一工位的半成品，加工后放回传送带。区别是每个工位是个会思考的 LLM。

**今天的设计要点（明天 Day 33 深挖）**：state 里**给每个角色的产出单独开字段**（`research` / `analysis` / `report`），而不是全塞进一个 `messages` 里混着。这样下游清楚该读哪个字段，不会被上游的"思考过程噪音"污染。

## 2. 跟着做（Hands-on）

**Step 1 — 定义共享 state：每个角色一块产出区**

```python
"""Day 32: 多 Agent 流水线 研究员→分析师→报告 / a 3-stage agent pipeline."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI


# 共享 state：给每个阶段单独开字段，职责清晰、互不污染
# shared state: a dedicated field per stage → clear ownership, no cross-pollution
class PipelineState(TypedDict):
    topic: str                                          # 输入主题 / the input topic
    research: str                                       # 研究员产出 / raw findings
    analysis: str                                       # 分析师产出 / structured insights
    report: str                                         # 报告员产出 / final report
    messages: Annotated[list[AnyMessage], add_messages] # 可选：全程留痕 / optional audit trail


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```

**Step 2 — 三个角色节点**

```python
def researcher(state: PipelineState) -> dict:
    """研究员：围绕 topic 收集要点（生产里这里接检索工具/RAG）。"""
    prompt = f"你是研究员。围绕主题「{state['topic']}」列出 5 条关键事实，每条一行，简洁。"
    findings = llm.invoke(prompt).content
    # 只写 research 字段 / write ONLY the research field
    return {"research": findings, "messages": [HumanMessage(content="[研究员] 资料已就绪")]}


def analyst(state: PipelineState) -> dict:
    """分析师：把原始资料提炼成结构化洞察（读 research，写 analysis）。"""
    prompt = (
        "你是分析师。基于以下资料，提炼 3 条核心洞察 + 1 个风险提示：\n"
        f"--- 资料 ---\n{state['research']}"
    )
    insights = llm.invoke(prompt).content
    return {"analysis": insights, "messages": [HumanMessage(content="[分析师] 洞察已生成")]}


def writer(state: PipelineState) -> dict:
    """报告员：把洞察写成一页简报（读 analysis，写 report）。"""
    prompt = (
        f"你是报告撰写者。就主题「{state['topic']}」，把下面的洞察写成一份 200 字以内、"
        f"含标题/要点/结论的中文简报：\n--- 洞察 ---\n{state['analysis']}"
    )
    report = llm.invoke(prompt).content
    return {"report": report, "messages": [HumanMessage(content="[报告员] 简报已完成")]}
```

**Step 3 — 串成流水线并运行**

```python
builder = StateGraph(PipelineState)
builder.add_node("researcher", researcher)
builder.add_node("analyst", analyst)
builder.add_node("writer", writer)

# 固定顺序流水线：研究员 → 分析师 → 报告员 / a fixed linear pipeline
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)
graph = builder.compile()


if __name__ == "__main__":
    init: PipelineState = {"topic": "2026 年 AI Agent 在企业客服的落地",
                           "research": "", "analysis": "", "report": "", "messages": []}
    out = graph.invoke(init)

    print("===== 研究员产出 =====\n", out["research"])
    print("\n===== 分析师产出 =====\n", out["analysis"])
    print("\n===== 最终简报 =====\n", out["report"])
    print("\n流水线留痕:", [m.content for m in out["messages"]])
```

运行：`python day32_pipeline.py`。你会拿到三段递进的产出，最后一份成文简报。**每个角色只读自己该读的字段、只写自己的字段**——这就是干净的多 Agent 协作。

**Step 4 — 升级：把节点换成真·子 Agent（带工具）**

```python
# 研究员需要真检索时，把"一次 LLM 调用"升级成带工具的子 Agent
# upgrade a node from a single LLM call to a tool-using sub-agent
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(q: str) -> str:
    """联网检索（演示桩；可换成 Day 20 的 RAG 检索）/ search stub."""
    return f"[检索] {q} 的最新数据：……"

_researcher_agent = create_react_agent(model="openai:gpt-4o-mini", tools=[search],
                                       prompt="你是研究员，用 search 工具收集资料后汇总。")

def researcher_v2(state: PipelineState) -> dict:
    sub = _researcher_agent.invoke({"messages": [HumanMessage(content=f"研究主题：{state['topic']}")]})
    return {"research": sub["messages"][-1].content}   # 只把"最终结论"提取进共享 state
```

> 进阶要点：子 Agent 内部有它自己的一堆 `messages`（思考、工具调用…）。**别把这些内部噪音整团灌进主流水线的共享 state**——只提取它的"最终结论"放进 `research` 字段。这正是明天 Day 33 要系统讲的"避免状态污染"。

## 3. 今日任务

1. 跑通 `day32_pipeline.py`，确认三段产出递进合理、最终简报成形。
2. **换主题压测**：换 2~3 个不同主题跑，看流水线是否稳定（每段都该有内容、不串字段）。
3. **升级研究员**：用 `researcher_v2`（带工具/或接你 Day 20 的 RAG 检索）替换纯 LLM 版，确认只把"结论"写进 `research`、没把子 Agent 的内部消息污染进来。
4. **加一道质检节点**：在 `writer` 后加 `reviewer` 节点，检查简报是否含"结论"二字，缺了就（先简单）打个警告——为后面 Day 36 健壮性、Day 30 HITL 复审埋点。

**验收标准**：流水线端到端跑通；多主题稳定不串字段；研究员升级为子 Agent 后无状态污染；质检节点能对产出做基本校验。

## 4. 自测清单

- [ ] 我能说清为什么这个任务用流水线比用 supervisor 更合适。
- [ ] 我理解"每个角色单独开 state 字段"带来的好处。
- [ ] 我会把一个节点从"单次 LLM 调用"升级成"带工具的子 Agent"。
- [ ] 我知道子 Agent 的内部 messages 不该整团污染主 state。
- [ ] 我能在流水线里插入质检/复审节点。

## 5. 延伸 & 关联

- 明天：系统化今天点到的"状态污染"问题——Agent 间到底怎么传中间结果、怎么隔离各自的工作区。
- 这条流水线是 Day 41–45「自动化研究 Agent」主力项目的骨架，今天先把骨架立住。
- 本仓库相关章节：
  - RAG 检索（研究员节点的真实数据来源）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - 多 Agent 范式回看 Day 31：[./Day-31-multi-agent-paradigms.md](./Day-31-multi-agent-paradigms.md)
