# Day 41 · 阶段项目①：自动化研究 Agent 架构设计

> **今日目标**：为本阶段主力项目「自动化研究 Agent」定架构——输入→检索→分析→产出报告，画清 state、节点、数据流，写好接口契约。
> **时长**：~2h ｜ **前置**：Day 26–30（LangGraph 状态机/分支/循环/持久化/HITL）、Day 32（多 agent 流水线）
> **今日产出**：一个 `design.md`（架构图 + state schema + 节点职责表 + 验收标准），以及空壳 `state.py`（仅类型定义，明天填实现）。

## 1. 为什么 & 是什么

接下来 5 天要做一个能写进简历的主力项目：**给定一个研究问题，agent 自动把它拆成子问题、并行检索、分析综合、产出一份带引用的结构化报告**——这正是 2025–2026 各家"Deep Research"产品的开源版骨架。

**先设计后编码**，这是和"只会写 demo 的人"拉开差距的关键。对 Java 工程师来说，今天做的就是**画时序图 + 定 DTO + 写接口契约**那一套，只是把"Service 调用链"换成"LangGraph 节点流"。

整体架构（plan → research → analyze → report，带循环与人审）：

```
            ┌──────────────────────────────────────────────┐
  question  │                                              │
  ─────────▶│  plan ──▶ research ──▶ analyze ──▶ [enough?]  │
            │   ▲          (并行检索)              │  no    │
            │   │                                  ▼  yes   │
            │   └──────────── refine ◀── gap?     report ──▶│──▶ 报告(带引用)
            │                                  (HITL 审一道) │
            └──────────────────────────────────────────────┘
                         ↑ checkpointer 持久化每步状态 ↑
```

五个核心节点的职责（这就是你的"Service 接口清单"）：

| 节点 | 输入 | 输出 | 单一职责 |
|---|---|---|---|
| `plan` | 原始问题 | 子问题列表 | 把大问题拆成可检索的子问题 |
| `research` | 子问题列表 | 各子问题的来源片段 | **并行**检索（呼应 Day 40） |
| `analyze` | 来源片段 | 结构化发现 + 是否充分 | 综合证据、判断是否还有 gap |
| `refine` | 已知 gap | 新增子问题 | gap 驱动的二次检索（循环回 research）|
| `report` | 结构化发现 | Markdown 报告（带引用） | 产出，经 HITL 确认 |

设计要点（今天就要想清，否则后面返工）：

1. **State 是唯一真相源**。所有节点读写同一个 state（呼应 Day 33"避免状态污染"）。子问题、来源、发现、报告全挂在 state 上，节点之间不直接传参。
2. **循环要有上限**。`research↔refine` 是循环（Day 28），必须设 `max_rounds`，否则可能永远"还差一点"——这是 Day 36 健壮性的直接应用。
3. **来源必须可溯源**。每个发现都要挂上它来自哪个 source，这样 `report` 才能生成引用（呼应 Day 22 引用与溯源）。这决定了 state 里 `sources` 和 `findings` 的结构。
4. **断点 + 人审是产品级要求**。`report` 前插一个 HITL 暂停点（Day 30），让人确认或打回——这是 Day 44 的事，但 state 今天就要为它留好字段。

## 2. 跟着做（Hands-on）

今天不写业务逻辑，只**固化数据契约**。把 state schema 用 `TypedDict` 写死——它就是贯穿 5 天的"DTO"。

```bash
pip install "langgraph>=0.2.50" "langchain-openai>=0.2" "pydantic>=2.7"
```

```python
"""Day 41: 研究 Agent 的状态契约 / the state contract for the research agent.

只定义类型，不含逻辑。明天起逐节点填实现。
Types only — no logic yet. Nodes get implemented from Day 42 on.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Source(BaseModel):
    """一条检索来源，report 阶段据此生成引用 / a retrieved source for citation."""

    url: str
    title: str
    snippet: str


class Finding(BaseModel):
    """一条带溯源的发现 / a single finding with provenance."""

    claim: str = Field(description="结论/论点 / the claim")
    source_urls: list[str] = Field(description="支撑该结论的来源 url / supporting sources")


class ResearchState(TypedDict, total=False):
    """贯穿全流程的唯一状态 / the single source of truth across all nodes."""

    question: str                                   # 原始问题 / original question
    sub_questions: list[str]                        # plan 产出 / from plan
    # 用 operator.add 让多轮/并行 research 的来源自动累加，而非互相覆盖
    # operator.add merges sources across parallel/looped research instead of overwriting
    sources: Annotated[list[Source], operator.add]
    # findings 同样用 operator.add：多轮 gap 循环里每轮发现需累加而非覆盖，否则报告只剩最后一轮
    findings: Annotated[list[Finding], operator.add]  # analyze 产出，多轮累加 / from analyze, merged
    is_sufficient: bool                             # analyze 判定是否充分 / enough?
    round: int                                      # 当前研究轮次 / loop counter
    max_rounds: int                                 # 循环上限（健壮性）/ loop cap
    report: str                                     # report 产出 / final markdown
    human_approved: bool                            # HITL 结果（Day 44 用）/ HITL gate
```

> 设计注解：`sources` 和 `findings` 都用 `Annotated[..., operator.add]` 是 LangGraph 的 **reducer** 机制——并行/多轮节点对同一字段的写入会被"归并"而非"覆盖"。研究 Agent 有 gap 循环，每轮都会新增来源与发现，若不加 reducer，第二轮的写入会把第一轮覆盖掉，最终报告只剩最后一轮。这正是 Day 33"多 agent 共享状态不互相踩"的标准解法，今天先埋好。

## 3. 今日任务

1. 写出 `design.md`：包含上面的架构图（可手绘拍照或用文字版）、节点职责表、以及 3 条明确的**项目验收标准**（例：能对一个真实问题产出含 ≥3 条引用的报告 / 循环有上限不会失控 / report 前能暂停等人确认）。
2. 落地 `state.py`（即上面的代码），确认 `import` 无误、`ResearchState` 可被实例化为一个 dict。
3. **画数据流走一遍**：拿一个具体问题（如"2026 年 multi-agent 框架怎么选"），在纸上手动推演 state 从 `plan` 到 `report` 每一步会被填进什么字段——验证你的 schema 不缺字段。

**验收标准**：`design.md` 含架构图 + 节点表 + ≥3 条验收标准；`state.py` 可导入且 `ResearchState` 字段覆盖了五节点的全部输入输出；能口头讲清"为什么 sources 要用 reducer 累加"。

## 4. 自测清单

- [ ] 我能画出 plan→research→analyze→(refine 循环)→report 的整体流，并说清每条边的触发条件。
- [ ] 我的 state schema 覆盖了五个节点的全部 I/O，且为 HITL 和循环上限预留了字段。
- [ ] 我理解 state 是唯一真相源，节点之间不直接传参。
- [ ] 我知道 `research↔refine` 必须有 `max_rounds` 上限，并知道这是 Day 36 的延续。
- [ ] 我能解释每条 finding 为何必须挂 source（决定了能否生成引用）。
- [ ] 我懂 `Annotated[list, operator.add]` 这个 reducer 解决的是并行/多轮写入归并问题。

## 5. 延伸 & 关联

- LangGraph state/node/edge 心智模型：本课程 Day 26（LangGraph 入门）。
- 条件分支与循环（analyze 的 enough? 判断、refine 回环）：本课程 Day 27–28。
- 引用与溯源（report 节点的依据）：本课程 Day 22 ；底层 RAG 概念：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
- 多 agent 共享状态不污染：本课程 Day 33。
- 明天 Day 42：把今天的空壳 state 接上真正的 `plan/research/analyze` 节点，跑通主干 graph。
