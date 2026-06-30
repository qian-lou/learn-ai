# Day 46 · 可观测性概念：为什么 Agent 必须 trace

> **今日目标**：搞懂 Agent 为什么**离了 trace 就没法维护**，建立 trace / span / 属性 的心智模型，亲手用结构化日志记录一次「多步 Agent 运行」。
> **时长**：~2h ｜ **前置**：Day 1~45（尤其 Day 41~45 的研究 Agent）
> **今日产出**：一个 `mini_trace.py`，用一个轻量上下文管理器把一次 Agent 运行的每个 LLM 调用 / 工具调用记成带耗时、token、父子关系的 span，并打印成树。

## 1. 为什么 & 是什么

前 45 天你能让 Agent **跑起来**。但跑起来 ≠ 能上线。Agent 上线后会遇到一类你以前没见过的 bug：

- **复现不出来**：温度不为 0 时同一 prompt 每次输出都不同，用户说"它答错了"你却重现不了。
- **慢在哪、贵在哪不知道**：一次请求慢了 30 秒、月底账单翻倍，但具体卡在哪一步、钱花在哪个环节全是黑盒。
- **循环不可见**：Agent"思考→调工具→再思考"循环了十几轮，中间过程看不见。

传统后端这些问题你早有答案：**链路追踪（distributed tracing）**。给 Java 工程师的对照——这正是你熟悉的那套东西：

| Agent 可观测性 | Java / 微服务世界 | 说明 |
|---|---|---|
| **Trace（一次完整运行）** | 一条 SkyWalking / Zipkin 调用链 | 用户一次提问的端到端记录，有唯一 trace_id |
| **Span（一个步骤）** | 链路里的一个 Span（一次方法/RPC 调用） | 一次 LLM 调用、一次工具调用，各是一个 span |
| **父子 span** | 调用栈 / Span 的 parent_id | "Agent 决策"是父，它派生出的"工具调用"是子 |
| **属性（attributes）** | Span 的 Tag / MDC 上下文 | 模型名、token 数、延迟、成本、工具入参出参 |
| **采集后端** | Prometheus + Grafana + APM | LangSmith / Langfuse / Jaeger 等 |

**核心心智：Agent 是一个非确定性的、会自己分叉的调用树。** 普通 REST 接口是"请求→处理→响应"一条直线，可观测性是加分项；Agent 是"请求→模型决策→可能调 N 个工具→把结果喂回模型→再决策……"的**动态树**，每次运行形状都可能不同。**没有 trace，你就是在闭着眼睛维护一个黑盒。** 这就是为什么生产级 Agent 把可观测性当**地基**而非装饰。

一次典型 Agent trace（缩进表示父子）：顶层"Agent 运行"下挂"LLM 决策#1 → 工具A → LLM 决策#2 → 工具B → 最终回答"，每个 span 标注耗时/token/成本/入出参。一眼就能回答：慢在哪、花了多少、模型每一步**为什么**这么决策——下面的代码会真实产出这样一棵树。

## 2. 跟着做（Hands-on）

今天**先不接专业工具**（Day 47 接 Langfuse / LangSmith）。我们手写一个 mini-tracer，把"trace / span / 父子 / 属性"**变成代码**——理解了原理，明天接现成 SDK 就秒懂。

```python
"""Day 46: 手写最小 trace 体系，把概念变成可触摸的代码 / a minimal hand-rolled tracer."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class Span:
    """一个步骤的记录 / one step in the run (an LLM call, a tool call...)."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    start: float = 0.0
    duration_ms: float = 0.0
    # 业务属性：token / 成本 / 入出参，等价于 APM 的 Tag
    # business attributes (tokens / cost / io) — like APM tags
    attrs: dict = field(default_factory=dict)


class Tracer:
    """收集一次运行内所有 span，并维护当前父节点 / collects spans for one run."""

    def __init__(self) -> None:
        self.spans: List[Span] = []
        self._stack: List[str] = []  # 当前父 span 栈 / current parent stack

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        """开一个 span，自动记录耗时与父子关系；块内可往 .attrs 塞结果 / open a span."""
        s = Span(name=name, parent_id=self._stack[-1] if self._stack else None, attrs=attrs)
        s.start = time.perf_counter()
        self._stack.append(s.span_id)
        try:
            yield s  # 块内代码执行 / user code runs here
        finally:
            # 无论是否异常都要收尾 —— 这是可观测性的铁律
            # always close the span, exception or not
            s.duration_ms = (time.perf_counter() - s.start) * 1000
            self._stack.pop()
            self.spans.append(s)

    def print_tree(self) -> None:
        """把扁平的 span 列表按父子缩进打印 / render spans as a tree."""
        by_parent: dict = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)
        total_cost = sum(s.attrs.get("cost", 0.0) for s in self.spans)
        print(f"Trace 总成本 ${total_cost:.4f}")

        def walk(parent_id: Optional[str], depth: int) -> None:
            for s in by_parent.get(parent_id, []):
                extra = " ".join(f"{k}={v}" for k, v in s.attrs.items())
                print(f"{'  ' * depth}└─ {s.name:<16} {s.duration_ms:6.0f}ms  {extra}")
                walk(s.span_id, depth + 1)  # 递归子 span / recurse children

        walk(None, 0)


# ---- 用 mini-tracer 记录一次「假的」两步 Agent 运行 / trace a fake 2-step agent ----
def fake_llm(tracer: Tracer, name: str, in_tok: int, out_tok: int) -> None:
    """模拟一次 LLM 调用；真实场景这里换成 client.chat.completions.create。"""
    with tracer.span(name, model="gpt-4o-mini", in_tok=in_tok, out_tok=out_tok) as s:
        time.sleep(0.05)  # 模拟网络/推理耗时 / simulate latency
        # 价格示例：$0.15/1M 输入, $0.60/1M 输出 / illustrative pricing
        s.attrs["cost"] = round(in_tok / 1e6 * 0.15 + out_tok / 1e6 * 0.60, 6)


def fake_tool(tracer: Tracer, name: str, args: dict, result) -> None:
    """模拟一次工具调用 / simulate a tool call."""
    with tracer.span(name, args=args) as s:
        time.sleep(0.01)
        s.attrs["result"] = result


if __name__ == "__main__":
    tracer = Tracer()
    # 顶层 span 包住整次运行，子 span 自动挂在它下面
    # the top span wraps the whole run; children nest under it automatically
    with tracer.span("Agent 运行"):
        fake_llm(tracer, "LLM 决策#1", in_tok=320, out_tok=45)
        fake_tool(tracer, "Tool 天气", {"city": "北京"}, {"temp_c": 22})
        fake_llm(tracer, "LLM 决策#2", in_tok=410, out_tok=38)
        fake_tool(tracer, "Tool 换算", {"c": 22}, 71.6)
        fake_llm(tracer, "LLM 终答", in_tok=480, out_tok=60)

    tracer.print_tree()
```

运行 `python mini_trace.py`，你会得到一棵带耗时、token、成本的 span 树。**这棵树的每个概念，明天接 Langfuse / LangSmith 时会一一对应**——它们只是把这套东西做成了云端可视化 + 持久化 + 团队协作。

> 关键体会：`span()` 用 `contextmanager` + `try/finally` 保证**异常也能收尾**——这正是 OpenTelemetry 等真实库的做法。可观测性代码必须"无论成功失败都记下来"，否则恰恰丢掉最该看的失败样本。

## 3. 今日任务

1. 跑通 `mini_trace.py`，确认输出是一棵缩进树，且每个 span 有耗时/属性。
2. **制造一次失败**：在某个 `fake_tool` 里 `raise RuntimeError("超时")`，给 `span()` 的 `finally` 之外**捕获异常并在 span 上记 `attrs["error"]`**，确认失败的 span 仍被记录（而不是整棵树丢失）。
3. **接真实调用**：把 `fake_llm` 换成一次真正的 `client.chat.completions.create`，从 `response.usage` 取真实 `prompt_tokens` / `completion_tokens` 填进 span，跑一次真实的两步小流程。
4. **写一段话**：用你自己的话回答——"为什么普通 CRUD 接口可以不 trace，但 Agent 不行？"（提示：非确定性 + 动态调用树 + 成本不可见）。

**验收标准**：能打印 span 树；失败的 span 带 `error` 属性且不影响其它 span；至少一个 span 的 token 来自真实 `usage`；能讲清 Agent 比传统接口"更需要" trace 的三个理由。

## 4. 自测清单

- [ ] 我能用 Java 链路追踪（trace / span / parent / tag）类比 Agent 可观测性。
- [ ] 我理解 Agent 是"非确定性 + 动态分叉的调用树"，这是它必须 trace 的根因。
- [ ] 我知道一个 span 至少要记哪些属性：模型、token、延迟、成本、入出参。
- [ ] 我的 tracer 用 `try/finally` 保证失败也收尾，不丢失败样本。
- [ ] 我能说出"不 trace 会死在哪些坑上"：复现难、定位慢、成本黑盒、循环不可见。

## 5. 延伸 & 关联

- 明天把这套手写概念换成生产工具：[Day-47-tracing-setup.md](./Day-47-tracing-setup.md)；这套 trace 数据正是后面评估（Day 49~50）与成本优化（Day 51）的原料。
- 本仓库相关章节：
  - 评估与监控（LLM 上线后的质量/延迟/成本监控全景）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - 实验管理（W&B / MLflow，可观测性在训练侧的近亲）：[../08-llm-engineering/03-mlops/01-experiment-tracking.md](../08-llm-engineering/03-mlops/01-experiment-tracking.md)
