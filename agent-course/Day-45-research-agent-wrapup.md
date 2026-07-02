# Day 45 · 🎯 阶段项目完成 + 复盘：自动化研究 Agent

> **今日目标**：把 Day 41–44 的代码收口成一个能完整跑通的项目，补一个最小 CLI 入口和 README，并准备好"在面试里讲清楚它"。
> **时长**：~2h ｜ **前置**：Day 41–44（设计→主干→多 agent→韧性）
> **今日产出**：一个可一键运行的研究 Agent（`python -m research_agent "你的问题"` 出报告），一份讲清架构/取舍的 README，和一段 3 分钟的项目自述。

## 1. 为什么 & 是什么

这是本系列**第三个里程碑、简历主力项目**。今天不加新功能，专做三件让它"值钱"的事：**收口（能一键跑）、表达（能讲清）、复盘（知道边界）**。

为什么"能讲清"和"能跑"同样重要：面试官不会读你 500 行代码，他听你**用 3 分钟讲明白"你做了什么决策、为什么"**。一个能说清"我为什么用条件边做循环、为什么循环要设上限、为什么 report 前要 HITL"的人，和一个只会说"我用 LangGraph 搭了个 agent"的人，差距是数量级的。这个项目的全部价值，浓缩在你**讲取舍**的能力里。

回看这 5 天，你其实把整个 Phase 3 串成了一条线——这张"知识点→项目落点"映射就是你的面试弹药：

| Phase 3 知识点 | 在本项目的落点 | 一句话价值主张 |
|---|---|---|
| LangGraph state/node/edge（Day 26）| 五节点主干 graph | 用状态机而非 if-else 堆叠管理复杂流程 |
| 条件分支 + 循环（Day 27–28）| analyze→refine→research 的 gap 回环 | 让 agent 自己判断"够不够"，不够就再查 |
| 状态持久化（Day 29）| SqliteSaver 断点续跑 | 长任务挂了能续，不浪费已花的 token |
| HITL（Day 30）| report 前人审 gate | 关键产出有人把关，可控可信 |
| 多 agent + 共享状态（Day 32–33）| web_reader 子 agent + reducer 累加 | 分层编排，状态不互相污染 |
| 健壮性（Day 36）| max_rounds + 节点降级 | 不死循环、不被单点异常拖垮 |
| MCP/A2A 取舍（Day 37–39）| 同进程用函数、判断何时该上协议 | 知道"不该用什么"也是工程能力 |
| 并行 + 流式（Day 40）| 并行检索多子问题 | 用 max(tᵢ) 而非 Σtᵢ 的时间出结果 |

## 2. 跟着做（Hands-on）

把四天的散件收成一个包，补一个 CLI 入口。建议目录：

```
research_agent/
├── __init__.py
├── __main__.py     # CLI 入口 / CLI entry
├── state.py        # Day 41 的契约 / the contract
├── nodes.py        # Day 42–44 的所有节点 / all nodes
└── graph.py        # build_graph_v2()（含 checkpointer + HITL）/ assembly
```

最小 CLI 入口（把人审做成命令行交互，端到端可跑）：

```python
"""research_agent/__main__.py — 一键运行研究 Agent / one-command runner."""

from __future__ import annotations

import sys
import uuid

from langgraph.types import Command

from .graph import build_graph_v2


def main() -> None:
    """命令行入口：问题→（人审）→报告 / CLI: question -> HITL -> report."""
    if len(sys.argv) < 2:
        raise SystemExit('用法 / usage: python -m research_agent "你的研究问题"')

    app = build_graph_v2()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}  # 每次运行独立存档 / fresh slot

    # 跑到人审处会暂停 / runs until the HITL interrupt
    state = app.invoke({"question": sys.argv[1], "max_rounds": 3}, cfg)

    # 命令行里完成人审（生产中是前端/审批系统）/ do HITL on the terminal
    if state.get("__interrupt__"):
        payload = state["__interrupt__"][0].value
        print("\n=== 待审发现 / findings to review ===")
        for c in payload["待审发现 / findings"]:
            print(" •", c)
        decision = input("approve / reject? > ").strip() or "approve"
        state = app.invoke(Command(resume=decision), cfg)  # 从断点续跑 / resume

    print("\n" + "=" * 50)
    print(state.get("report", "（被打回，未生成报告 / rejected）"))


if __name__ == "__main__":
    main()
```

跑通它：

```bash
export OPENAI_API_KEY="sk-..."
python -m research_agent "2026 年中小团队如何选型 multi-agent 框架？"
# → 拆子问题 → 并行检索 → 分析(可能多轮) → 暂停让你 approve → 输出带引用报告
```

**README 必须包含的五块**（这是面试官的阅读路径）：① 一句话项目定位；② 架构图（用 Day 42 导出的 mermaid）；③ 怎么跑（上面三行）；④ **关键设计决策与取舍**（为什么循环要上限、为什么 HITL、为什么不上 A2A）；⑤ 已知局限与下一步（见下面复盘）。

## 3. 今日任务

1. 收口成包，`python -m research_agent "..."` 能端到端跑通：拆问题→并行检索→（可能多轮）→人审→出带引用报告。
2. 写完整 README（含上面五块），架构图与代码一致。
3. **录一段 3 分钟自述**（讲给自己听并计时）：项目做什么 → 整体架构一句话 → **挑 3 个设计决策讲清取舍**（建议选：条件边循环+上限 / checkpointer 断点续跑 / HITL）→ 已知局限。
4. **写复盘清单**：至少列 3 条"现在的局限 + 怎么改"。参考方向：
   - 检索质量：DDG 桩→换 Tavily/Bing 或接 Day 16–25 的 RAG 向量检索做混合检索。
   - 可观测性：现在靠 print→Day 47 接 LangSmith/Langfuse trace 每步 token/延迟/成本。
   - 评估：没有质量度量→Day 49–50 建测试集，量化引用准确率/幻觉率。
   - 成本：每轮全量重检索→加缓存、对 analyze 充分性判断用更小模型（Day 51–52）。

**验收标准**：项目一键可跑并产出带引用报告；README 五块齐全且图文一致；能脱稿 3 分钟讲清项目与 3 个关键取舍；复盘清单 ≥3 条且每条都给出"现状→改法"。

## 4. 自测清单

- [ ] 我的项目能用一条命令端到端跑通，产出带真实引用的报告。
- [ ] 我能在 3 分钟内讲清项目架构和至少 3 个设计决策的取舍。
- [ ] 我能解释每一个 Phase 3 知识点在本项目里的具体落点。
- [ ] 我的 README 含架构图，且图与代码保持一致。
- [ ] 我清楚项目当前的 3 个以上局限，以及各自的改进路径。
- [ ] 我知道这个项目接下来会在 Phase 4 被加上可观测性/评估/安全（Day 46–60），逐步变成"能上线"。

> **里程碑三档自评**：🟩 跑通（demo 端到端不崩：拆题→并行检索→人审→出报告一条命令走完）
> → 🟨 能讲清取舍（为什么条件边循环要设上限、为什么 report 前加 HITL、为什么同进程不上 A2A，替代方案与代价）
> → 🟥 能扛住追问（失败路径：检索全挂/循环打满怎么降级；成本：一次研究烧多少 token；安全：网页内容注入怎么防；规模：并发多任务时 checkpointer 怎么扛）。
>
> **量化亮点卡**：写一张「指标 前→后 + 一句话讲法」
> （例：任务完成率 70%→95%、并行检索耗时从 Σtᵢ 降到 max(tᵢ)、断点续跑省掉整轮重复 token、报告引用可溯源率 100%），面试直接用。

## 5. 延伸 & 关联

- 本课程 Day 41–44：本项目的设计与实现来源。
- 下一步可观测性（让 print 变成真正的 trace）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)，对应本课程 Day 46–48。
- 把检索升级为真正的 RAG（混合检索/重排）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)，对应本课程 Day 16–25。
- 部署为服务（让它能被调用而不止跑在终端）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)，对应本课程 Day 56。
- 里程碑回看：这是计划里的第 3 个里程碑（自动化研究 Agent ⭐ 主力项目），见 [../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)。
