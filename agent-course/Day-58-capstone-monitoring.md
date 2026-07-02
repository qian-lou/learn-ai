# Day 58 · 阶段项目①：给 Agent 服务加完整监控

> **今日目标**：给 Day 56~57 的 agent 服务接上**完整可观测性**——结构化日志（带 trace_id）+ Prometheus 指标 + LLM 调用 tracing，做到"线上出问题能定位到是哪一步、花了多久、花了多少钱"。
> **时长**：~2h ｜ **前置**：Day 46~48（可观测概念）、Day 56~57（加固后的服务）
> **今日产出**：服务暴露 `/metrics`（QPS/延迟/错误率/token 成本），每条请求日志带可串联的 `trace_id`，并接一个 tracing 后端看到逐步 span。

## 1. 为什么 & 是什么（概念 + Java 类比）

Agent 是个"黑盒套黑盒"：你的代码调 LLM，LLM 内部看不见；多步 agent 还会调工具、再调 LLM。**线上一旦慢了/错了/贵了，没有监控就是抓瞎。** 这是"能 demo"和"能运维"的分水岭。

可观测性三大支柱，给 Java 类比：

| 支柱 | 回答什么问题 | Python 工具 | Java 类比 |
|---|---|---|---|
| **Logs** | 这一条请求发生了什么 | `structlog` | Logback + MDC（带 traceId） |
| **Metrics** | 整体健康吗（QPS/延迟/错误率） | `prometheus-client` | Micrometer + Actuator `/prometheus` |
| **Traces** | 一次请求每一步耗时/调用树 | OpenTelemetry | Sleuth / Micrometer Tracing |

Agent 特有、普通 Web 服务没有的两类指标：

- **Token / 成本**：每次调用的 `prompt_tokens` / `completion_tokens`，换算成钱。**这是会真实烧钱的指标**，必须上 dashboard。
- **质量类信号**：降级率、工具调用失败率、空回答率——光看延迟和 500 不够，agent 可能"成功返回了一坨废话"。

核心心智：**trace_id 是把三大支柱缝起来的线**。一条请求进来生成一个 id，日志、span、报错都带上它——出事时拿一个 id 就能把整条链路捞出来。这正是你在 Spring 里用 MDC `traceId` 做的事。

## 2. 跟着做（Hands-on）

**Step 1 — 装监控栈**（2026 现代组合）

```bash
pip install "structlog>=24.4" "prometheus-client>=0.21" \
  "opentelemetry-sdk>=1.29" "opentelemetry-instrumentation-fastapi>=0.50" \
  "opentelemetry-exporter-otlp>=1.29"
```

**Step 2 — 三支柱接入（中间件统一注入 trace_id）**

```python
"""Day 58: 给 agent 服务加完整监控 / full observability for the agent service."""
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from openai import AsyncOpenAI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

# 结构化日志：JSON 输出、机器可解析 / structured JSON logs
structlog.configure(processors=[
    structlog.contextvars.merge_contextvars,  # 自动带上 contextvars 里的 trace_id / inject trace_id
    structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()
# Prometheus 指标：QPS / 延迟 / 错误 / token 成本 / core metrics
REQ = Counter("agent_requests_total", "请求总数", ["endpoint", "status"])
LAT = Histogram("agent_latency_seconds", "端到端延迟", ["endpoint"])
TOK = Counter("agent_tokens_total", "token 消耗", ["kind"])  # kind=prompt|completion
clients: dict[str, AsyncOpenAI] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    clients["llm"] = AsyncOpenAI()
    yield
    await clients["llm"].close()

app = FastAPI(title="Agent Service (observed)", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())  # Prometheus 抓取端点 / scrape endpoint
MODEL = "gpt-4o-mini"

@app.middleware("http")
async def observe(request: Request, call_next):
    """中间件：每请求生成 trace_id、记录延迟与日志 / per-request tracing."""
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:16]
    structlog.contextvars.bind_contextvars(trace_id=trace_id, path=request.url.path)  # 绑到 contextvars，后续日志自动带 trace_id / bind so every log carries it
    start, status = time.perf_counter(), 500  # 默认 500，异常也能记到指标 / default for errors
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        LAT.labels(request.url.path).observe(elapsed)
        REQ.labels(request.url.path, status).inc()
        log.info("request_done", latency_ms=round(elapsed * 1000, 1))  # 一行串起整条请求 / one line per request
        structlog.contextvars.clear_contextvars()

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, object]:
    """带 token/成本埋点的端点 / endpoint with token & cost metrics."""
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": req.message}]
    resp = await clients["llm"].chat.completions.create(model=MODEL, messages=msgs)
    u = resp.usage
    TOK.labels("prompt").inc(u.prompt_tokens)  # 记录 token 供成本看板 / tokens for cost dashboard
    TOK.labels("completion").inc(u.completion_tokens)
    cost_usd = u.prompt_tokens / 1e6 * 0.15 + u.completion_tokens / 1e6 * 0.60  # 单价示意 / illustrative pricing
    log.info("llm_call", prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
             cost_usd=round(cost_usd, 6), llm_latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return {"answer": resp.choices[0].message.content, "cost_usd": round(cost_usd, 6)}
```

**Step 3 — 接 OpenTelemetry 链路（一行自动埋点）**

```python
# tracing.py —— main 启动前 import 一次，自动给 FastAPI 埋 span / auto-instrument FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

trace.set_tracer_provider(TracerProvider())
# 导出到本地 collector（Jaeger/Tempo），4317 是 OTLP gRPC 默认端口 / export to local collector
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)))
FastAPIInstrumentor.instrument_app(app)  # 一行给 FastAPI 自动埋点 / auto-instrument the app
```

> 自动埋点 = 不改业务代码，框架层面把每请求变成一个 trace——和 Spring 加 `micrometer-tracing` 依赖就自动有 traceId 同一套思路。本地可 `docker run jaegertracing/all-in-one` 起后端看 span 树。

**Step 4 — 验证**

```bash
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"hi"}'
curl -s localhost:8000/metrics | grep -E 'agent_(requests|latency|tokens)'   # 看到指标在涨
# 日志应是一行行 JSON，每条含 trace_id —— 用 jq 过滤某条链路
```

**Step 5 — 从指标到告警：SLO 与 error budget**

到这里指标只是"挂在墙上的仪表盘"——没人 24 小时盯着看。**监控的终点不是画图，是"何时该叫醒人"。** 这一步把 Step 2 的指标变成会主动发火的告警。给 Java 工程师的直接类比：这就是你在 **Prometheus + Alertmanager / SkyWalking / ARMS** 上配的**告警规则 + 值班（on-call）**，只是把阈值从"JVM 老年代占用 > 90%"换成了"agent p95 延迟 / 错误率 / 烧钱速度"。

先分清三个词，别混：

| 术语 | 是什么 | 本服务的例子 |
|---|---|---|
| **SLI**（指标） | 一个能测的健康数值 | p95 延迟、5xx 错误率、每千次调用成本 |
| **SLO**（目标） | 给 SLI 定的承诺线 | "p95 < 3s"、"错误率 < 0.5%"、"每千次调用 < ¥3" |
| **error budget**（预算） | `1 - SLO` 允许坏的余量 | SLO 99.5% ⇒ 每月允许 0.5% 请求失败 |

**为什么要 error budget 而不是"错一次就报警"**：偶发抖动天天有，逐次告警只会把人练成"狼来了"。正确姿势是盯**烧预算的速度（burn rate）**——预算 30 天烧完记为 `1x`；若当前错误率是预算的 14 倍（`14.4x`），照这速度约 **2 天**（720h ÷ 14.4）就烧光整月预算，这才值得半夜叫醒人。快慢双窗（fast/slow burn）是 Google SRE 的标准配方：快窗抓"正在失火"，慢窗抓"温水煮青蛙"。

给本服务定 3 条 SLO（对应 Step 2 已埋的指标，不用新加埋点）：

| SLO | SLI 来源指标 | 承诺线 | 告警触发 |
|---|---|---|---|
| 延迟 | `agent_latency_seconds` 直方图 | p95 < 3s | 持续 5min 超阈值 |
| 可靠性 | `agent_requests_total{status}` | 5xx 错误率 < 0.5% | error budget 快速烧（14.4x） |
| 成本 | `agent_tokens_total` | 每分钟烧钱 < ¥Y | token 增速突刺 |

**最小可用告警规则（`alerts.yml`，交给 Prometheus 加载）：**

```yaml
# alerts.yml —— Prometheus 规则文件，prometheus.yml 里用 rule_files 引入 / alerting rules
groups:
  - name: agent-slo
    rules:
      # 1) p95 延迟持续 5min 超 3s / p95 latency SLO breach
      - alert: AgentP95LatencyHigh
        expr: histogram_quantile(0.95, sum by (le) (rate(agent_latency_seconds_bucket[5m]))) > 3
        for: 5m                      # 抖动不算，持续 5min 才发 / sustained, not a blip
        labels: { severity: warning }
        annotations:
          summary: "agent p95 延迟 {{ $value | humanize }}s > 3s"

      # 2) 错误率 burn-rate 快烧：5xx 占比 > 预算的 14.4 倍(=0.5%*14.4≈7.2%) / fast error-budget burn
      - alert: AgentErrorBudgetFastBurn
        expr: |
          sum(rate(agent_requests_total{status=~"5.."}[5m]))
            / sum(rate(agent_requests_total[5m])) > 0.072
        for: 2m
        labels: { severity: page }   # page = 直接呼叫值班 / wake on-call
        annotations:
          summary: "错误率 error budget 正在 14.4x 速率燃烧，约 2 天烧光整月预算"

      # 3) 成本 Counter 增速异常：每分钟 completion token 增量突刺 / cost spike
      - alert: AgentTokenSpendSpike
        expr: sum(rate(agent_tokens_total{kind="completion"}[5m])) * 60 > 50000
        for: 10m
        labels: { severity: warning }
        annotations:
          summary: "completion token 烧速 {{ $value | humanize }}/min，疑似 prompt 膨胀或被刷"

      # 4) 就绪探针连续失败：/readyz（Day 57）被 blackbox 探测标红 / readiness down
      - alert: AgentNotReady
        expr: probe_success{job="agent-readyz"} == 0
        for: 3m
        labels: { severity: page }
        annotations:
          summary: "/readyz 连续 3min 失败，下游(LLM/DB)不通，已摘流量待人工介入"
```

> 规则 4 的 `probe_success` 来自 blackbox_exporter 对 `/readyz` 的定时探测——把 Day 57 做的"依赖不通就 503"接上了"连续失败就呼人"。三种 severity 分级：`page` 立刻叫醒值班，`warning` 进工作时间处理，避免告警疲劳。

**经 Alertmanager 路由到通知渠道（最小 `alertmanager.yml`）：**

```yaml
# alertmanager.yml —— 按 severity 路由到不同渠道 / route by severity
route:
  receiver: slack-warning
  group_by: [alertname]          # 同类告警聚合成一条，别刷屏 / dedupe & group
  routes:
    - matchers: [severity="page"]
      receiver: pagerduty-oncall  # page → 电话/短信呼叫值班 / wake on-call
receivers:
  - name: slack-warning
    slack_configs:
      - api_url: "https://hooks.slack.com/services/XXX"  # ← Webhook，勿硬编码进仓库，用环境变量/密钥管理 / secret, not in repo
        channel: "#agent-alerts"
  - name: pagerduty-oncall
    pagerduty_configs:
      - routing_key: "YYY"        # ← PagerDuty 密钥 / secret
```

链路一句话串起来：**Prometheus 按规则算出 firing → 推给 Alertmanager → 按 severity 分组/去重/路由 → warning 进 Slack、page 呼 PagerDuty 叫值班**。这正是 Day 59 反向引用的"Day 57/58 已有的告警护栏"落地的地方。

**接回 Day 59 的数据飞轮**：告警不止是"叫人"，更是**坏 case 的自动捕手**。`AgentErrorBudgetFastBurn` 一响，值班顺着告警里的时间窗 → 用 Step 2 的 `trace_id` 从日志/Jaeger 捞出这批失败请求 → 把它们清洗标注后**回灌进 Day 59 的 `REGRESSION` / `DATASET`**。于是链路闭合：**告警触发 → 抓到坏 case → 回灌 eval 集 → 修复 → 回归门禁挡住复发**。告警是飞轮的"进料口"——它把线上真实故障源源不断喂给 eval，比拍脑袋编测试值钱得多。

## 3. 今日任务

1. 把三支柱接到**你的 agent 服务**：结构化日志（带 `trace_id`）+ `/metrics`（QPS/延迟/错误/token）+ OTel 链路。
2. **成本看板雏形**：连发 20 次请求，`curl /metrics | grep tokens` 手算花了多少钱，建立"每千次请求 ≈ ?元"的直觉。
3. **复现一次定位**：用 `asyncio.sleep` 让某分支变慢，凭 `trace_id` 从日志捞出这条慢请求的每步耗时，指出瓶颈。
4. **多步 agent 加 span**：给关键 node 手动开 span（`with tracer.start_as_current_span("retrieve"):`），在 Jaeger 看调用树。
5. **指标变告警**：为你的服务写出**至少 1 条 SLO**（如"p95 < 3s"）和对应的告警规则（Step 5 的 `alerts.yml` 挑一条），并在 `annotations` 或注释里写清**触发后的响应动作**——例：`AgentErrorBudgetFastBurn` firing → 呼值班 → 顺 `trace_id` 捞失败 trace → 疑似 prompt/模型劣化则回滚上一版本，并把这批坏 case 回灌 Day 59 的 `REGRESSION`。

**验收标准**：`/metrics` 四类指标随请求变化；日志是 JSON 且能用一个 `trace_id` 串起整条请求；能从 trace 指出"哪一步最慢/最贵"；写得出 ≥1 条 SLO + 告警规则，并能讲清触发后"叫谁、看哪个 trace、做什么补救"。

## 4. 自测清单

- [ ] 能说清三支柱各回答什么问题、分别对应 Java 的什么组件。
- [ ] 知道为什么 agent 必须额外监控 token/成本和质量类信号。
- [ ] 理解 `trace_id` 如何把日志/指标/链路缝在一起（= Spring 的 MDC traceId），并据此定位"哪一步慢/贵"。
- [ ] 懂自动埋点 vs 手动 span 的区别与适用场景。
- [ ] 能说清 SLI/SLO/error budget/burn-rate 四者关系，理解"盯烧预算速度而非逐次告警"为何能治告警疲劳。
- [ ] 会写一条 SLO + Prometheus 告警规则，并说明经 Alertmanager 按 severity 路由到 Slack/PagerDuty 的链路，以及告警如何把坏 case 回灌 Day 59 的 eval 飞轮。

## 5. 延伸 & 关联

- 本仓库 评估与监控（指标体系更系统）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 本仓库 实验管理（离线指标/记录）：[../08-llm-engineering/03-mlops/01-experiment-tracking.md](../08-llm-engineering/03-mlops/01-experiment-tracking.md)
- Step 5 的 SLO/告警是 Day 59 数据飞轮的"进料口"：告警抓到的坏 case 直接回灌 Day 59 的 `REGRESSION`/`DATASET`，也是它引用的"Day 57/58 告警护栏"的落地处。
- 明天 Day 59 补 **eval + 安全防护** 凑齐"生产级"；Day 65 把今天的 trace 升级成**贯通 Python→Java**的端到端链路，今天的 OTel 是地基。
- 本系列总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
