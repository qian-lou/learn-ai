# Day 39 · 复杂工具编排：工具 + 子 agent 完成一个复杂任务

> **今日目标**：掌握"工具不够用时，把一整个子 agent 当成一个工具"的 orchestrator–worker 模式，在一个进程内编排多工具 + 子 agent。
> **时长**：~2h ｜ **前置**：Day 9（ReAct）、Day 32（多 Agent 流水线）、Day 38（A2A，理解为何这里不上 A2A）
> **今日产出**：一个 `day39_orchestrator.py`，一个主 agent 把"网页检索"子 agent 与"计算/格式化"工具组合起来，完成一个需要多步、跨能力的任务。

## 1. 为什么 & 是什么

单层 ReAct agent 有个天花板：当任务需要**多种异质能力**（检索 + 计算 + 撰写）时，把十几个工具全塞给一个模型，它会选错、漏选、上下文爆炸。生产级做法是**分层编排（orchestrator–worker）**：

- **Orchestrator（主 agent）**：只负责"拆解任务、决定调谁、汇总结果"，工具列表很短。
- **Worker（子 agent / 工具）**：每个只精通一件事。关键技巧——**把一个子 agent 包装成主 agent 眼里的一个普通 tool**（"agent-as-tool"）。主 agent 不关心子 agent 内部有几步、调了什么，只看输入输出。

| 概念 | 含义 | Java 类比 |
|---|---|---|
| Orchestrator | 任务分派 + 汇总 | **Service 编排层 / Saga 协调者** |
| Worker tool | 单一职责的能力 | 一个 `@Service` Bean |
| agent-as-tool | 子 agent 对外只暴露 1 个函数签名 | 门面模式（Facade）——内部多步，对外一个方法 |
| 结果汇总 | 把多个 worker 输出合成答案 | `CompletableFuture.allOf(...).thenApply(汇总)` |

**为什么这里不上 A2A（呼应昨天）**：今天所有 worker 都在**同一个进程**里，是你自己的函数/子 agent。直接函数调用零序列化、零网络、好调试。A2A 的标准化发现与任务生命周期在这里纯属负担。**"同进程用函数，跨进程才上 A2A"**——这条边界要刻进脑子。

## 2. 跟着做（Hands-on）

我们用最朴素的方式手写编排（不依赖重框架，看清本质）：一个 `web_research` 子 agent（内部自己跑一个小 ReAct 循环）被包装成工具，主 agent 把它和 `compute`、`format_report` 两个普通工具一起编排。

```bash
pip install "openai>=1.40"
```

```python
"""Day 39: orchestrator–worker 编排，含 agent-as-tool / orchestration with agent-as-tool."""

from __future__ import annotations

import json

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---- Worker 1：一个"子 agent"，对外只暴露成一个函数 / a sub-agent exposed as one function ----
def web_research(topic: str) -> str:
    """子 agent：就给定主题做多步检索并返回要点（演示用桩检索）。

    Args:
        topic: 研究主题 / the research topic.

    Returns:
        归纳后的要点文本 / distilled findings.
    """
    # 真实场景这里是一个完整的内部 ReAct 循环（检索→读→再检索）
    # In reality this is a full internal ReAct loop; stubbed for the lesson.
    fake_hits = [
        f"{topic}：2025 起 orchestrator-worker 成为主流多 agent 模式",
        f"{topic}：子 agent 包装成 tool 可降低主模型的选择负担",
    ]
    summary = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "把检索片段归纳成两条要点，中文。"},
            {"role": "user", "content": "\n".join(fake_hits)},
        ],
    )
    return summary.choices[0].message.content


# ---- Worker 2/3：普通工具 / plain tools ----
def compute(expression: str) -> str:
    """安全计算简单算术（仅演示，生产应用 AST 白名单）/ tiny arithmetic, demo only."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:  # 防注入：拒绝非算术字符 / reject non-arith chars
        return "拒绝：含非法字符"
    return str(eval(expression))  # noqa: S307 — 已白名单约束 / whitelisted above


def format_report(title: str, body: str) -> str:
    """把内容格式化成 Markdown 报告 / wrap content into a Markdown report."""
    return f"# {title}\n\n{body}\n"


# ---- 主 agent 可见的工具表（子 agent 在这里就是一个 tool）/ tools the orchestrator sees ----
def _tool(name: str, desc: str, *params: str) -> dict:
    """用必填字符串参数快速生成一个 tool schema / build a tool schema with required string params."""
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string"} for p in params},
                       "required": list(params)}}}


TOOLS = [
    _tool("web_research", "对一个主题做多步联网研究，返回归纳要点", "topic"),
    _tool("compute", "计算一个算术表达式", "expression"),
    _tool("format_report", "把标题和正文组装成 Markdown 报告", "title", "body"),
]
DISPATCH = {"web_research": web_research, "compute": compute, "format_report": format_report}


def orchestrate(task: str, max_steps: int = 6) -> str:
    """主 agent 循环：决定调谁→执行→回填→汇总 / orchestrator ReAct loop."""
    messages = [
        {"role": "system", "content": "你是编排者。拆解任务，调用工具，最后产出一份报告。"},
        {"role": "user", "content": task},
    ]
    for _ in range(max_steps):  # 步数上限兜底（呼应 Day 36）/ hard cap from Day 36
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:           # 模型给出最终答案 / final answer
            return msg.content
        messages.append(msg)             # 回填模型的工具调用意图 / append tool-call intent
        for call in msg.tool_calls:      # 逐个执行并回填结果 / run each tool, feed back
            args = json.loads(call.function.arguments)
            result = DISPATCH[call.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
    return "达到步数上限 / hit step cap"


if __name__ == "__main__":
    print(orchestrate(
        "研究『2026 年多 agent 编排趋势』，估算如果每个子 agent 节省 3 步、共 5 个子 agent 一共节省多少步，"
        "最后产出一份带标题的简短报告。"
    ))
```

模型会自己决定：先 `web_research` → 再 `compute("3*5")` → 最后 `format_report`。注意主 agent 全程**不知道** `web_research` 内部还调了一次 LLM——这就是 agent-as-tool 的封装价值。

## 3. 今日任务

1. 跑通 `day39_orchestrator.py`，从打印中确认主 agent 至少调用了 2 个不同工具并产出 Markdown 报告。
2. **加一个 worker**：新增一个 `translate(text, lang)` 工具（可继续用 LLM 桩），让任务要求"把报告标题翻成英文"，观察主 agent 是否自动多调一步。
3. **对照昨天**：写下两三句话——如果把 `web_research` 改成"由另一团队独立部署的 A2A agent"，你的代码要怎么改、为什么今天这种同进程场景不值得那样做。

**验收标准**：主 agent 能在一次任务里编排 ≥3 个能力（含子 agent）；新增 worker 后能被自动选用；能讲清 agent-as-tool 的封装收益与"同进程 vs 跨进程"的取舍。

## 4. 自测清单

- [ ] 我能解释 orchestrator–worker 为什么比"把所有工具塞给一个 agent"更稳。
- [ ] 我会把一个子 agent 包装成主 agent 眼里的单个 tool（agent-as-tool）。
- [ ] 我理解主 agent 不需要知道 worker 内部步数，只看输入输出（门面）。
- [ ] 我能说清为什么今天不上 A2A，而 Day 38 那种场景才上。
- [ ] 我的编排循环带步数上限兜底（呼应 Day 36 健壮性）。

## 5. 延伸 & 关联

- ReAct 循环基础（编排循环的内核）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- 本课程 Day 32（研究员→分析师→报告生成）：今天的能力将直接用于 Day 41–45 的研究 Agent。
- 本课程 Day 36（健壮性）：编排循环必须配步数上限/超时。
- 明天 Day 40（性能）：把今天串行的多个 worker 调用改成**并行**，缩短整体延迟。
- 一个完整 LangChain 应用的组织方式：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
