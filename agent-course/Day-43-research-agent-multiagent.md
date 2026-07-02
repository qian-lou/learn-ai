# Day 43 · 阶段项目③：多 Agent 协作 + 工具接入

> **今日目标**：把昨天的桩检索换成真实工具 + 子 agent，让 `research` 并行检索多个来源，并把 `analyze` 升级成"判断 gap→循环补检"的多轮流程。
> **时长**：~2h ｜ **前置**：Day 42（主干 graph）、Day 39（工具+子 agent 编排）、Day 40（并行）、Day 28（循环）
> **今日产出**：升级版 `research_agent.py`，`research` 节点并行调用真实检索工具与一个"网页阅读"子 agent，`analyze→refine→research` 形成带上限的 gap 循环。

## 1. 为什么 & 是什么

昨天主干通了但"水质"很差：检索是假的、只走一轮。今天补两件让它真正"像个研究员"的事：

**① 真实工具 + 子 agent 协作（呼应 Day 39）**
- `research` 不再返回假来源，而是：对每个子问题**并行**（Day 40）调用一个真实搜索工具拿到候选链接，再交给一个 **`web_reader` 子 agent**（agent-as-tool）去"读"并提炼。主 graph 只把它当一个能力，不关心其内部步数。

**② gap 驱动的多轮循环（呼应 Day 27–28）**
- `analyze` 现在要真判断"证据够不够"：若发现某些子问题没被覆盖，就置 `is_sufficient=False` 并产出新的 gap 子问题；graph 通过**条件边**走 `refine→research` 再来一轮；够了就走 `report`。**循环必须有 `max_rounds` 上限**（Day 36），否则会无限"还差一点"。

对 Java 工程师，这两步分别是：把"假 DAO"换成"真调外部服务 + 门面封装的子流程"，以及给 Service 编排加一个"条件回环 + 最大重试次数"。

graph 拓扑从直线变成带回环：

```
plan → research → analyze ──[sufficient or round≥max]──▶ report → END
          ▲                       │ gap & round<max
          └──────── refine ◀──────┘
```

## 2. 跟着做（Hands-on）

只改昨天文件里的 `research_node`、`analyze_node`，新增 `refine_node` 与一条**条件边**。检索工具用 DuckDuckGo（免 key，适合学习）。

```bash
pip install ddgs   # 免 key 搜索；ddgs 是 duckduckgo-search 更名后的现行包 / no-key search
```

```python
"""Day 43: 真实工具 + 子 agent + gap 循环 / real tools, a sub-agent, and a gap loop.

仅展示相对 Day 42 的改动；其余节点与 build/main 复用昨天。
Shows only the deltas vs Day 42; other nodes and build/main are reused.
"""

from __future__ import annotations

import asyncio
import json

from ddgs import DDGS  # duckduckgo-search 已更名为 ddgs，接口 drop-in
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from state import Finding, ResearchState, Source

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---- 子 agent：web_reader，对外是"一个工具" / a sub-agent exposed as one tool ----
def web_reader(question: str, raw_snippets: list[str]) -> list[Finding]:
    """子 agent：读检索片段，提炼出带溯源的发现（内部可多步，对外一函数）。

    Args:
        question: 当前子问题 / the sub-question being researched.
        raw_snippets: 检索到的原始片段（含 url）/ raw retrieved snippets with urls.

    Returns:
        带 source_urls 的发现列表 / findings with provenance.
    """
    prompt = (
        f"针对子问题「{question}」，从片段提炼 1~2 条结论，每条 JSON "
        '{"claim":"...","source_urls":["..."]}，只返回 JSON 数组：\n' + "\n".join(raw_snippets)
    )
    raw = llm.invoke(prompt).content
    items = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
    return [Finding(**it) for it in items]


# ---- 真实检索 + 子 agent，按子问题并行 / real search + sub-agent, parallel per sub-question ----
async def _research_one(q: str) -> tuple[list[Source], list[Finding]]:
    """单个子问题：搜索→读→出发现 / search, read, produce findings for one sub-question."""
    with DDGS() as ddgs:                       # 同步库放线程里跑 / run sync lib off-loop
        hits = await asyncio.to_thread(lambda: list(ddgs.text(q, max_results=3)))
    sources = [Source(url=h["href"], title=h["title"], snippet=h["body"]) for h in hits]
    snippets = [f"{s.snippet} ({s.url})" for s in sources]
    findings = web_reader(q, snippets) if snippets else []
    return sources, findings


def research_node(state: ResearchState) -> dict:
    """并行研究所有（子）问题，汇总来源与发现 / research all sub-questions in parallel."""
    async def _gather():
        return await asyncio.gather(*[_research_one(q) for q in state["sub_questions"]])
    results = asyncio.run(_gather())          # 并行收口（呼应 Day 40）/ concurrent fan-out
    all_sources = [s for src, _ in results for s in src]
    all_findings = [f for _, fnd in results for f in fnd]
    print(f"[research] 并行得到 {len(all_sources)} 来源 / {len(all_findings)} 发现")
    return {"sources": all_sources, "findings": all_findings}


# ---- analyze：真判 gap / decide whether evidence is sufficient ----
def analyze_node(state: ResearchState) -> dict:
    """判断证据是否覆盖原问题；不足则给出 gap 子问题 / sufficiency + gaps."""
    claims = "\n".join(f"- {f.claim}" for f in state["findings"])
    prompt = (
        f"原问题：{state['question']}\n已有结论：\n{claims}\n"
        '判断证据是否充分。只返回 JSON：{"sufficient": true/false, "gaps": ["补充子问题", ...]}'
    )
    raw = llm.invoke(prompt).content
    verdict = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    print(f"[analyze] 充分={verdict['sufficient']} gaps={verdict.get('gaps', [])}")
    return {"is_sufficient": verdict["sufficient"], "sub_questions": verdict.get("gaps", [])}


# ---- refine：把 gap 变成下一轮的子问题，并推进轮次 / turn gaps into next round ----
def refine_node(state: ResearchState) -> dict:
    """进入下一轮研究 / advance the loop counter for another research pass."""
    return {"round": state.get("round", 1) + 1}


# ---- 条件边：够了 or 到上限 → report；否则 → refine / route on sufficiency & cap ----
def route_after_analyze(state: ResearchState) -> str:
    """充分或达轮次上限就出报告，否则继续补检（带上限，Day 36）/ loop with a hard cap."""
    if state["is_sufficient"] or state.get("round", 1) >= state.get("max_rounds", 3):
        return "report"
    return "refine"


# 组装时把昨天的 add_edge("research","analyze") 之后改为条件边：
#   g.add_conditional_edges("analyze", route_after_analyze, {"report": "report", "refine": "refine"})
#   g.add_edge("refine", "research")   # 回环 / loop back
```

把这些改动并入昨天的文件，并按注释把 `analyze` 后改成条件边、`refine→research` 连成回环。再跑一次：对一个较宽的问题，你会看到它可能研究 2~3 轮、补检 gap，最后产出覆盖更全的报告。

> 注意 `research_node` 每轮 `return {"sources": all_sources, "findings": all_findings}`，二者靠 Day 41 state 里的 `operator.add` reducer **跨轮累加**——`findings` 也必须带 reducer（Day 41 已同步补上），否则第二轮会覆盖第一轮，最终报告只剩最后一轮的发现，与"覆盖更全"矛盾。

## 3. 今日任务

1. 把改动并入 `research_agent.py`，跑通"真实搜索 + 子 agent + 条件循环"，确认报告里的引用是**真实可点的 URL**。
2. **触发循环**：故意问一个宽泛问题（如"2026 AI agent 全景"），观察 `[analyze] 充分=False` 触发 `refine→research` 二轮；再把 `max_rounds=1` 验证上限能强制收口。
3. **验证并行**：在 `_research_one` 里加耗时打印，确认多个子问题是并发而非串行检索（呼应 Day 40）。

**验收标准**：报告含真实来源 URL；能观察到 gap 循环至少跑两轮且受 `max_rounds` 约束；多子问题检索为并行；`web_reader` 作为子 agent 被主 graph 当作单一能力调用。

## 4. 自测清单

- [ ] 我把 `web_reader` 封装成 agent-as-tool，主 graph 不感知其内部步数（呼应 Day 39）。
- [ ] 我用并行（`asyncio.gather`）检索多个子问题（呼应 Day 40）。
- [ ] 我用条件边实现了"充分→report / 不足→refine→research"的回环（Day 27–28）。
- [ ] 我的循环有 `max_rounds` 硬上限，绝不会无限补检（Day 36）。
- [ ] 我的每条发现都挂了真实 source_urls，report 能据此生成可点引用。

## 5. 延伸 & 关联

- 工具 + 子 agent 编排（agent-as-tool）：本课程 Day 39。
- 并行工具调用：本课程 Day 40。
- 条件分支与循环：本课程 Day 27–28。
- 进阶检索（混合检索/重排，可替换今天的 DDG）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
- 明天 Day 44：给这条已经会"多轮研究"的 graph 加上 **checkpointer 断点续跑 + report 前 HITL + 错误恢复**。
