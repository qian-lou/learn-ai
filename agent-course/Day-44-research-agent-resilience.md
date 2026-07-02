# Day 44 · 阶段项目④：断点续跑 + HITL + 错误恢复

> **今日目标**：给研究 Agent 加三层生产保障——checkpointer 断点续跑、report 前人审（HITL）、节点错误恢复，让它"挂了能续、关键步等人、出错不崩盘"。
> **时长**：~2h ｜ **前置**：Day 43（多 agent + 循环）、Day 29（状态持久化）、Day 30（HITL）、Day 36（健壮性）
> **今日产出**：带 `SqliteSaver` 持久化、`interrupt` 人审、节点级 try/except 降级的 `research_agent.py`，可中断后从断点恢复、可在出报告前暂停等人确认。

## 1. 为什么 & 是什么

到 Day 43，agent 功能完整了，但还是个"玻璃大炮"：跑到一半进程挂了要从头来、生成报告前没人把关、某个工具抛异常就整条 graph 崩。今天补的三件事，正是"能 demo"和"能交付"的分水岭：

| 能力 | 解决 | 用到的 API | Java 类比 |
|---|---|---|---|
| **断点续跑** | 长任务中断后从头重跑（费钱费时） | `checkpointer=SqliteSaver` + `thread_id` | Saga/工作流引擎的**状态持久化 + 续跑** |
| **HITL 人审** | 关键产出无人把关就发出 | `interrupt()` + `Command(resume=...)` | 审批流的**人工审核节点** |
| **错误恢复** | 单节点异常炸掉整个流程 | 节点内 try/except → 降级写回 state | `try/catch` + 降级返回兜底值 |

三个关键认知：

1. **checkpointer = 自动存档点**。开启后，LangGraph 在**每个节点执行后**把整个 state 落盘（SQLite），并用 `thread_id` 标识一次会话。进程崩了、或你主动中断了，下次用同一个 `thread_id` 调 `invoke`，它会**从最后一个存档点接着跑**，而不是从头。这正是 Day 29 学的能力，今天落到项目。
2. **HITL = 在节点里"暂停并交还控制权"**。`interrupt(payload)` 会让 graph 在此**冻结并返回**，把 `payload` 抛给外部（人）；人看完后用 `Command(resume=人的决定)` 再次 `invoke`，graph 从**断点之后**继续。它和 checkpointer 是绝配——能暂停正因为状态被持久化了。
3. **错误恢复要"就地降级"**。节点抛异常默认会终止整个 graph。生产做法是：在节点内 `try/except`，失败时**写一个降级结果回 state**（如"该来源检索失败，跳过"）并让流程继续——而不是让一个坏链接搞垮整份报告。这是 Day 36"优雅降级"在项目里的落地。

## 2. 跟着做（Hands-on）

在 Day 43 的 graph 上加三处改动：编译时挂 checkpointer、`report` 前插一个 HITL 节点、给 `research` 包错误恢复。

```bash
pip install "langgraph-checkpoint-sqlite>=2.0"   # SQLite 持久化后端 / sqlite checkpointer
```

```python
"""Day 44: 断点续跑 + HITL + 错误恢复 / persistence, human-in-the-loop, and recovery.

仅展示相对 Day 43 的改动。
Shows only the deltas vs Day 43.
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from state import ResearchState


# ---- ① 错误恢复：给检索包一层就地降级 / wrap research with graceful degradation ----
def research_node_safe(state: ResearchState) -> dict:
    """检索失败不崩，写降级结果让流程继续 / degrade gracefully instead of crashing."""
    try:
        # ... 这里是 Day 43 的真实并行检索逻辑 ... / the real parallel research from Day 43
        return _do_real_research(state)  # 假设已抽成函数 / assume extracted
    except Exception as exc:  # 兜底：任何检索异常都降级 / catch-all, degrade
        print(f"[research] 检索失败，降级跳过 / failed, degrading: {exc}")
        # 不新增来源，但保留已有 state，让 analyze 用现有证据继续
        return {"sources": [], "is_sufficient": True}  # 强制收口，避免卡死


# ---- ② HITL：report 前暂停等人确认 / pause for human approval before reporting ----
def human_gate_node(state: ResearchState) -> dict:
    """把发现摘要抛给人，等待批准/打回 / surface findings, await approve/reject."""
    summary = [f.claim for f in state.get("findings", [])]
    # interrupt 会冻结 graph 并把 payload 返回给外部调用者（人）
    # interrupt freezes the graph and returns this payload to the caller (a human)
    decision = interrupt({"待审发现 / findings": summary, "请回复 / reply": "approve | reject"})
    if decision == "reject":
        # 人打回：标记未充分，graph 可路由回 refine 再研究一轮
        return {"is_sufficient": False, "human_approved": False}
    return {"human_approved": True}


def build_graph_v2():
    """挂 checkpointer + 插 HITL 节点 / attach checkpointer and the HITL gate."""
    g = StateGraph(ResearchState)
    # ... add_node: plan / research_node_safe / analyze / refine（同 Day 43）...
    g.add_node("human_gate", human_gate_node)
    # analyze 充分后先过人审，再出报告 / route: analyze -> human_gate -> report
    g.add_edge("human_gate", "report")
    g.add_edge("report", END)

    # 关键：编译时挂持久化后端 / the key line — attach the checkpointer
    # 用 from_conn_string 的 with 写法，连接会随函数返回而关闭，后续 invoke 必然失败。
    # 这里手动持有连接、直接构造 SqliteSaver，让它跨 invoke/进程重启存活。
    # check_same_thread=False：LangGraph 会在别的线程读写该连接。
    conn = sqlite3.connect("research.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    return g.compile(checkpointer=saver)


if __name__ == "__main__":
    app = build_graph_v2()
    cfg = {"configurable": {"thread_id": "run-001"}}  # 一次会话的存档标识 / save-slot id

    # 第一次 invoke：跑到 human_gate 时会 interrupt 并停下
    # first invoke: runs until the HITL gate interrupts and halts
    state = app.invoke({"question": "2026 年 agent 可观测性怎么做？", "max_rounds": 3}, cfg)
    print("已暂停，等待人审 / paused for human review:", state.get("__interrupt__"))

    # 人看完后恢复（用同一 thread_id，从断点继续）/ resume from the checkpoint
    final = app.invoke(Command(resume="approve"), cfg)
    print("\n" + "=" * 40 + "\n" + final["report"])
```

体会两件事：第一次 `invoke` 会**停在人审处返回**；用同一个 `thread_id` 带 `Command(resume="approve")` 再 `invoke`，它**不重跑前面的检索分析**，直接从断点续到 `report`——这就是 checkpointer + interrupt 的合力。

## 3. 今日任务

1. 跑通三件改动：确认 graph 会在 `human_gate` 暂停、`resume="approve"` 后从断点续到报告。
2. **测续跑**：第一次 `invoke` 后**直接结束进程**；重启脚本，仅用同一 `thread_id` 调 `app.invoke(Command(resume="approve"), cfg)`，验证它能从 SQLite 恢复状态、不重跑前序节点。
3. **测打回**：`resume="reject"`，确认 graph 把 `is_sufficient` 置 False 并路由回 `refine→research` 再研究一轮（需在 `human_gate` 后也加一条按 `human_approved` 的条件边）。
4. **测错误恢复**：在 `_do_real_research` 里手动 `raise`，确认 `research_node_safe` 降级、报告仍能产出而非全流程崩溃。

**验收标准**：能在 report 前暂停人审；进程重启后凭 `thread_id` 续跑且不重算前序节点；`reject` 能触发再研究；检索抛异常时优雅降级、报告照常产出。

## 4. 自测清单

- [ ] 我理解 checkpointer 在每个节点后落盘 state，`thread_id` 标识一次会话。
- [ ] 我能解释为什么"能暂停人审"依赖"状态被持久化"（两者是绝配）。
- [ ] 我会用 `interrupt()` 冻结 graph、用 `Command(resume=...)` 从断点恢复。
- [ ] 我的节点级 try/except 是"降级写回 state"，而非吞掉异常假装没事。
- [ ] 我知道这三件事分别对应 Day 29 / Day 30 / Day 36，是它们在项目里的落地。

## 5. 延伸 & 关联

- 状态持久化与断点续跑：本课程 Day 29。
- 人在回路（HITL）：本课程 Day 30。
- 健壮性（重试/超时/降级）：本课程 Day 36——`research_node_safe` 还可叠加 Day 36 的重试。
- 生产部署里持久化后端的选择（SQLite→Postgres）：[../08-llm-engineering/02-model-serving/03-docker-deployment.md](../08-llm-engineering/02-model-serving/03-docker-deployment.md)
- 明天 Day 45：收尾、端到端跑通、写 README + 复盘，把它打磨成简历主力项目。
