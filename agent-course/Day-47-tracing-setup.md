# Day 47 · 接入 tracing：把 Agent 接上 Langfuse / LangSmith / OTel

> **今日目标**：给昨天还在手写 span 的 Agent，接上一个**真正的 tracing 后端**，让每次运行自动上报、在 Web UI 里可视化。
> **时长**：~2h ｜ **前置**：Day 46（trace / span 心智模型）
> **今日产出**：一个 `traced_agent.py`，跑一次多步 Agent，能在 Langfuse（或 LangSmith）的 Web 控制台里看到完整 trace 树。

## 1. 为什么 & 是什么

昨天你手写了 60 行 tracer，理解了 trace / span / 父子 / 属性。今天**别再造轮子**——直接接生产工具，它们帮你搞定持久化、Web 可视化、跨进程、团队共享、检索聚合这些脏活。给 Java 工程师的对照：昨天的 mini-tracer 像手写 `System.out` 打日志；今天接 Langfuse 像引入 **SkyWalking Agent + 控制台**——**一个注解 / 一次 import 就自动埋点上报**。

三个主流选型（**任选其一接通即可**，本课主推 Langfuse）：

| 工具 | 定位 | 适合 | 类比 |
|---|---|---|---|
| **Langfuse**（主推） | LLM 专用、开源可自托管 | 想要开箱即用的 LLM trace + eval + 成本面板 | 自带控制台的 APM |
| **LangSmith** | LangChain 官方、托管 | 重度用 LangChain / LangGraph | 与框架深度绑定的 APM |
| **OpenTelemetry** | 通用标准协议 | 已有 Jaeger / Grafana 栈，想统一 | 行业标准的 trace 协议（不是某个产品） |

**关键认知（2026）：它们底层都在向 OpenTelemetry 的 GenAI 语义约定靠拢。** Langfuse v3 就是基于 OTel 重写的。所以"接 Langfuse"和"用 OTel"不是对立——前者是后者的"开箱即用版"。学会任一，概念都通。

> 一个必须知道的 2026 细节：OTel 的 GenAI 标准属性名近期改过——用 `gen_ai.provider.name`（**不是**老的 `gen_ai.system`），token 用 `gen_ai.usage.input_tokens` / `output_tokens`（**不是** `prompt_tokens` / `completion_tokens`）。下面 OTel 示例按新名字写。

## 2. 跟着做（Hands-on）

### 方案 A：Langfuse v3（主推，最少代码）

```bash
pip install "langfuse>=3" openai          # SDK 为 3.x 线 / the OTel-based v3 line
```

到 [cloud.langfuse.com](https://cloud.langfuse.com)（或自托管）建项目，拿到三件套写进环境变量：

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # 自托管则填你的地址 / self-host URL
```

```python
"""Day 47 方案A: Langfuse v3 自动埋点 / auto-instrument with Langfuse v3."""

from langfuse import get_client, observe
# 关键：用 Langfuse 包装过的 openai，调用方式不变但自动上报 / drop-in, same API, auto-traced
from langfuse.openai import openai

langfuse = get_client()  # 读环境变量拿单例 / singleton from env vars
MODEL = "gpt-4o-mini"


@observe()  # 这一个装饰器 = 给函数自动开一个 span（含入参/返回/耗时）
def pick_tool(question: str) -> str:
    """模拟一步 LLM 决策：问题该用哪个工具 / an LLM 'decide tool' step."""
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "只回答一个工具名：weather 或 calc。"},
            {"role": "user", "content": question},
        ],
        name="decide-tool",  # 给这个 LLM span 起名 / name this generation span
    )
    return resp.choices[0].message.content.strip()


@observe()  # 顶层函数 = 整条 trace；内部 @observe 函数自动成为子 span
def run_agent(question: str) -> str:
    """一次两步 Agent 运行 / a tiny 2-step agent run."""
    tool = pick_tool(question)  # 子 span 自动挂在本 trace 下 / nested automatically
    answer = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"用「{tool}」工具的口吻回答：{question}"}],
        name="final-answer",
    )
    return answer.choices[0].message.content


if __name__ == "__main__":
    print(run_agent("北京明天多少度？"))
    langfuse.flush()  # 短脚本必须 flush，否则进程退出前数据没发出去 / flush before exit
    print("已上报，去 Langfuse 控制台看 trace 树。")
```

跑 `python traced_agent.py`，到 Langfuse 控制台 → Traces，就能看到一棵 trace：顶层 `run_agent`，下挂 `decide-tool`、`final-answer` 两个 generation，每个带 token、延迟、**成本**（Langfuse 按模型价目表自动算）。**这就是昨天手写树的生产版。**

### 方案 B：LangSmith（重度用 LangChain 选它）

```bash
pip install -U langsmith openai
export LANGSMITH_TRACING=true            # 开关 / on switch
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="agent-course"
```

```python
"""Day 47 方案B: LangSmith 自动埋点 / auto-instrument with LangSmith."""

import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai

client = wrap_openai(openai.Client())  # 包装后自动记录 model+token / auto child runs


@traceable  # 等价于 Langfuse 的 @observe
def run_agent(question: str) -> str:
    """一次 LLM 调用，自动成为一条 trace / one call, auto-traced."""
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}],
    )
    return r.choices[0].message.content


# run_agent(...) → 异步上报，进程结束自动 flush；去 smith.langchain.com 看
```

### 方案 C：原生 OpenTelemetry（已有 Jaeger/Grafana 栈选它）

```python
# pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
"""Day 47 方案C: 手动打 OTel GenAI 标准 span（核心是属性名）/ manual OTel span."""
# 省略 provider/exporter 初始化（标准 OTel 套路：TracerProvider + OTLPSpanExporter，
# 上报到 OTEL_EXPORTER_OTLP_ENDPOINT）。重点是这些 2026 标准属性名：
with tracer.start_as_current_span("chat gpt-4o-mini") as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", "openai")        # 不是 gen_ai.system
    span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "OTel 是什么？"}],
    )
    u = resp.usage
    span.set_attribute("gen_ai.usage.input_tokens", u.prompt_tokens)    # 标准名非 prompt_tokens
    span.set_attribute("gen_ai.usage.output_tokens", u.completion_tokens)
```

> **避坑**：别在同一个 OpenAI client 上同时启用两套自动埋点（如 Langfuse + OpenInference），会产生**重复 span**。选一个主埋点即可。

## 3. 今日任务

1. **三选一接通**（推荐 Langfuse），跑 `traced_agent.py`，在 Web UI 里看到至少一棵含 2 个子 span 的 trace 树。
2. **核对属性**：在 UI 里确认每个 LLM span 都有 model、input/output token、延迟；Langfuse 还应显示**成本**。
3. **加业务属性**：给某个 span 附加自定义元数据（如 `user_id`、`question_type`），在 UI 里能看到——为后面"按用户/场景筛 trace"做准备。
4. **制造一次异常**：在工具里 `raise`，确认失败的 trace 在 UI 里被标红/记录（验证"失败也能被观测"）。

**验收标准**：Web 控制台能看到完整 trace 树；每个 LLM span 含 token + 延迟（Langfuse 含成本）；自定义元数据可见；异常运行被记录而非凭空消失。

## 4. 自测清单

- [ ] 我能说清 Langfuse / LangSmith / OTel 三者定位与适用场景。
- [ ] 我理解"它们底层都在向 OTel GenAI 约定靠拢"，学一个即通。
- [ ] 我会用 `@observe` / `@traceable` 把函数变成自动埋点的 span。
- [ ] 我知道 OTel 2026 属性名：`gen_ai.provider.name`、`gen_ai.usage.input_tokens`。
- [ ] 我知道短脚本要 `flush`，且不能在一个 client 上叠两套埋点。

## 5. 延伸 & 关联

- 明天用这些 trace **调一个真实 bug**——看 token / 延迟 / 成本定位问题：[Day-48-trace-debugging.md](./Day-48-trace-debugging.md)；这些数据也是 Day 49~50 评估、Day 51 成本优化的原料。
- 本仓库相关章节：
  - 评估与监控（生产监控全景，trace 是其数据源）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - 实验管理（W&B / MLflow，可观测性在训练侧的近亲）：[../08-llm-engineering/03-mlops/01-experiment-tracking.md](../08-llm-engineering/03-mlops/01-experiment-tracking.md)
