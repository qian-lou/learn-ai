# Day 42 · 阶段项目②：搭主干 graph + 状态 schema

> **今日目标**：把昨天的 state 契约接上真实节点，用 LangGraph 跑通 `plan→research→analyze→report` 的串行主干，端到端产出第一版报告。
> **时长**：~2h ｜ **前置**：Day 41（架构设计 + state.py）、Day 26–28（LangGraph 节点/分支/循环）
> **今日产出**：一个 `research_agent.py`，能对一个真实问题跑完整条主干并打印 Markdown 报告（先单轮、不含循环/HITL，明后两天加）。

## 1. 为什么 & 是什么

今天的目标是**让骨架立起来、能跑通**——哪怕检索是桩、分析很糙也无所谓。先有一条**端到端的绿色通路**，再在它上面迭代。这是工程铁律：**先让管道通水，再调水质。** 对 Java 工程师，这等于"先把 Controller→Service→DAO 串通返回个假数据，再逐层填真实现"。

LangGraph 的核心三步（Day 26 学过，今天落到项目）：

| 步骤 | API | 含义 | Java 类比 |
|---|---|---|---|
| 定 state | `StateGraph(ResearchState)` | 声明流转的数据类型 | 定义流转的 DTO |
| 加节点 | `g.add_node("plan", plan_node)` | 注册一个处理函数 | 注册一个 `@Service` 方法 |
| 连边 | `g.add_edge("plan", "research")` | 规定执行顺序 | 编排调用链 |

**节点函数的统一签名是 `(state) -> dict`**：读 state、干活、返回"要更新到 state 的字段"。LangGraph 负责把返回的 dict 合并进全局 state（`sources` 字段会按昨天定义的 reducer 累加）。节点之间零直接传参——全靠 state。

## 2. 跟着做（Hands-on）

承接昨天的 `state.py`。今天写四个节点 + 组装 graph。检索先用桩函数（明天 Day 43 换成真实工具/子 agent）。

```python
"""Day 42: 研究 Agent 主干 graph（单轮串行）/ the backbone graph, single-pass."""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from state import Finding, ResearchState, Source  # 复用昨天的契约 / reuse yesterday's contract

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---- 节点 1：plan，拆子问题 / decompose into sub-questions ----
def plan_node(state: ResearchState) -> dict:
    """把原始问题拆成 3 个可检索子问题 / split into 3 sub-questions."""
    prompt = (
        f"把研究问题拆成 3 个互补、可独立检索的子问题，只返回 JSON 数组：\n问题：{state['question']}"
    )
    raw = llm.invoke(prompt).content
    subs = json.loads(raw[raw.find("["): raw.rfind("]") + 1])  # 容错截取 JSON / robust JSON slice
    print(f"[plan] 子问题 / sub-questions: {subs}")
    return {"sub_questions": subs, "round": state.get("round", 0) + 1}


# ---- 节点 2：research，检索（今日为桩，明日换真工具）/ retrieval (stub today) ----
def _search_stub(q: str) -> Source:
    """演示桩：返回一条假来源 / a fake source for now."""
    return Source(url=f"https://example.com/{abs(hash(q)) % 999}",
                  title=f"关于「{q}」的资料", snippet=f"{q} 的关键信息（示例）")


def research_node(state: ResearchState) -> dict:
    """对每个子问题检索一条来源 / one source per sub-question."""
    hits = [_search_stub(q) for q in state["sub_questions"]]
    print(f"[research] 收集到 {len(hits)} 条来源 / sources")
    return {"sources": hits}  # 经 reducer 累加进 state.sources / merged via reducer


# ---- 节点 3：analyze，综合发现 / synthesize findings ----
def analyze_node(state: ResearchState) -> dict:
    """把来源综合成带溯源的发现 / turn sources into cited findings."""
    bundle = "\n".join(f"- {s.title}: {s.snippet} ({s.url})" for s in state["sources"])
    prompt = (
        "基于以下来源，给出 2~3 条结论，每条 JSON 形如 "
        '{"claim": "...", "source_urls": ["..."]}，只返回 JSON 数组：\n' + bundle
    )
    raw = llm.invoke(prompt).content
    items = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
    findings = [Finding(**it) for it in items]
    print(f"[analyze] 得出 {len(findings)} 条发现 / findings")
    # 单轮版先恒定为充分；Day 43 再做 gap 判断与循环
    # single-pass: mark sufficient for now; gap-loop comes on Day 43
    return {"findings": findings, "is_sufficient": True}


# ---- 节点 4：report，产出带引用的报告 / produce a cited report ----
def report_node(state: ResearchState) -> dict:
    """把发现渲染成 Markdown 报告（含引用）/ render findings to cited Markdown."""
    lines = [f"# 研究报告：{state['question']}\n"]
    for i, f in enumerate(state["findings"], 1):
        cites = " ".join(f"[{u}]({u})" for u in f.source_urls)
        lines.append(f"{i}. {f.claim}  \n   来源 / sources: {cites}")
    return {"report": "\n".join(lines)}


def build_graph():
    """组装并编译主干 graph / assemble and compile the backbone."""
    g = StateGraph(ResearchState)
    for name, fn in [("plan", plan_node), ("research", research_node),
                     ("analyze", analyze_node), ("report", report_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")        # 入口 / entry
    g.add_edge("plan", "research")   # 串行主干 / serial backbone
    g.add_edge("research", "analyze")
    g.add_edge("analyze", "report")
    g.add_edge("report", END)        # 出口 / exit
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    # 初始 state：带上循环上限字段，为明后天铺路 / seed state, incl. loop cap
    final = app.invoke({"question": "2026 年选型 multi-agent 框架要看哪些维度？",
                        "max_rounds": 3})
    print("\n" + "=" * 40 + "\n" + final["report"])
```

跑起来你会看到 `[plan] → [research] → [analyze]` 的日志，最后吐出一份带引用链接的 Markdown 报告。**主干通了**——这就是今天的全部目标。

## 3. 今日任务

1. 跑通 `research_agent.py`，确认四个节点依次执行并产出含引用的报告。
2. **验证 state 流转**：在 `report_node` 开头 `print(state.keys())`，确认 `question/sub_questions/sources/findings` 都已就位——亲眼看到 state 是如何被逐节点填满的。
3. **导出 graph 结构**：用 `app.get_graph().draw_mermaid()` 打印 Mermaid 文本，贴进 `design.md`，得到一张和代码同步的架构图。

**验收标准**：能对任意问题端到端产出带引用的 Markdown 报告；能展示 state 在 report 前已含全部上游字段；能导出与代码一致的 graph 结构图。

## 4. 自测清单

- [ ] 我理解节点函数签名是 `(state) -> dict`，返回的是"要更新的字段"而非整个 state。
- [ ] 我知道 `sources` 的累加靠的是昨天定义的 reducer，节点只管"追加"。
- [ ] 我能用 `START`/`END` 和 `add_edge` 串起一条主干，并解释执行顺序。
- [ ] 我清楚为什么"先桩后真"——先打通管道再换真实检索。
- [ ] 我能导出 graph 结构图，做到图与代码一致。

## 5. 延伸 & 关联

- LangGraph 三件套（state/node/edge）：本课程 Day 26。
- 结构化输出（节点间用 Pydantic 对象传递发现）：本课程 Day 4 ；项目内已用 `Finding`/`Source`。
- 引用与溯源（report 的引用渲染）：本课程 Day 22。
- 昨天 Day 41：本文件直接复用其 `state.py`。
- 明天 Day 43：把 `research` 的桩换成真实工具 + 子 agent，并把 `analyze` 升级成带 gap 判断的循环。
