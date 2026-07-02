# Day 65 · 可观测性贯通两端：trace 从 Python 一路追到 Java

> **今日目标**：用 **OpenTelemetry** 把双栈的链路缝成**一条** trace——一次请求从 Python agent 的 LLM 调用,到 REST 跨栈,再到 Java 服务的 DB 操作,在 Jaeger 里显示为**同一棵 span 树**。出事时一眼看出"卡在 Python 推理还是 Java 查库"。
> **时长**：~2h ｜ **前置**：Day 58(单栈 OTel)、Day 62(Java Actuator/Micrometer)、Day 64(双栈已打通)
> **今日产出**：Jaeger 里看到一条横跨 Python+Java 的 trace,共享同一个 `trace_id`,span 树清晰展示每一步耗时与归属栈。

## 1. 为什么 & 是什么(概念 + Java 类比)

Day 58 你给单栈接了 trace,Day 64 把双栈打通了。但现在的问题是:**Python 有自己的 trace,Java 有自己的 trace,它们是两段断开的**。线上一旦慢了,你只知道"整体 3 秒",却分不清是 LLM 推理慢、还是 Java 查库慢。**这就是分布式追踪要解决的核心痛点。**

魔法在于一个标准:**W3C Trace Context(`traceparent` HTTP 头)**。

```
[Python] 一次请求生成 trace_id=abc...,在发起 REST 调用时,
         自动把 `traceparent: 00-abc...-span1-01` 塞进 HTTP 头
              │
              ▼  GET /api/v1/orders/u1   (Header: traceparent: 00-abc...)
[Java]   OTel 自动读取这个头,把自己的 span 挂到同一个 trace_id=abc... 下,
         成为 Python span 的"子节点"
              │
              ▼
[Jaeger] 看到一棵完整的树:
         └─ python: chat (3.1s)
            ├─ python: llm.plan (0.4s)
            ├─ java: GET /orders (0.2s)        ← 同一 trace,跨栈
            │  └─ java: db.query orders (0.15s)
            ├─ python: llm.decide (0.3s)
            └─ java: POST /refunds (1.9s)       ← 瓶颈在这!
               └─ java: db.tx refund (1.8s)
```

给 Java 工程师的精确类比:这**就是 Spring Cloud Sleuth / Micrometer Tracing 跨服务传 `traceId` 的同一套机制**——你早就在纯 Java 微服务里见过"一个 traceId 串起 A→B→C"。今天只是把"A、B 之一换成了 Python"。**W3C `traceparent` 是跨语言的通用语**,所以 Python 的 OTel 和 Java 的 Micrometer 能对上。

核心心智:**trace_id 是跨栈的"线",`traceparent` 头是"针"。** 只要发请求时把这根针带上、收请求时认得它,两段 trace 就缝成一条。最妙的是——**这一切几乎都是自动埋点(auto-instrumentation)完成的,你基本不用手写传播逻辑。**

## 2. 跟着做(Hands-on)

**Step 1 — Python 侧:让 httpx 自动注入 `traceparent`**

```python
"""Day 65: instrument httpx,跨栈调用自动带上 traceparent / propagate trace context."""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 服务名让 Jaeger 区分两栈 / service.name distinguishes the two stacks in Jaeger
provider = TracerProvider(resource=Resource.create({"service.name": "python-agent"}))
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True))
)
trace.set_tracer_provider(provider)

# 关键一行:httpx 自动埋点,所有出站请求自动注入 W3C traceparent 头 / THE key line
HTTPXClientInstrumentor().instrument()

tracer = trace.get_tracer("python-agent")
```

```python
def run_chat(message: str) -> str:
    """跑一次 agent,给 LLM 推理步骤手动开 span / run agent, manual span for the LLM step.

    Args:
        message: 用户消息 / the user message.

    Returns:
        agent 最终回答 / the final answer.
    """
    # 手动 span:标出 LLM 规划阶段,便于在 Jaeger 区分推理 vs 调 Java / mark the LLM step
    with tracer.start_as_current_span("agent.chat") as span:
        span.set_attribute("user.message", message[:80])  # 截断,勿记敏感全文 / truncate
        # 内部 tools 的 httpx 调用会自动续上同一 trace / inner httpx calls auto-propagate
        result = _agent.invoke({"messages": [("user", message)]})
        return result["messages"][-1].content
```

**Step 2 — Java 侧:几乎零代码,自动接住 `traceparent`**

Day 62 已经加了 `micrometer-tracing-bridge-otel` 和 OTLP endpoint。Spring Boot 3 的可观测栈**默认就会读取入站 `traceparent` 头并续接**,你只需确认配置:

```yaml
# application.yml —— 确认这几项(Day 62 已配大部分)/ confirm these
management:
  tracing:
    sampling.probability: 1.0          # 学习期全采样 / sample all in dev
    propagation.type: w3c              # 用 W3C 标准,与 Python OTel 对齐 / match Python
  otlp.tracing.endpoint: http://jaeger:4317
  observations.annotations.enabled: true   # 让 @Observed 注解生效 / enable annotations
spring.application.name: java-service       # 这就是 Jaeger 里的 service.name
```

```java
/** 给关键业务方法加 span,让 Java 侧在 trace 树里更细粒度可见。*/
@Service
public class OrderService {

    /** 退款:@Observed 自动产生一个挂在当前 trace 下的 span(低基数标签便于在 Jaeger 聚合)。 */
    @Observed(name = "order.refund", contextualName = "refund")
    @Transactional
    public RefundResultVO refund(RefundRequestDTO request) {
        // ...同 Day 62 的方法体:幂等查重 + 金额上限/可退额度校验 + doRefund + saveWithKey...
        // span 由 @Observed 自动开合,无需手写 / span is automatic
        // return result;
    }
}
```

> Java 侧的妙处:**入站 `traceparent` 由框架自动解析并续接**(等价 Sleuth 的行为),你只要加 `@Observed` 给业务方法补细粒度 span 即可。DB 调用若用 Spring Data,通常也已被自动埋点。

**Step 3 — 跑起来,在 Jaeger 看那棵跨栈的树**

```bash
docker compose up --build -d        # 含 jaeger（Day 60/64 的 compose 已有）
# 触发一次端到端请求 / fire one end-to-end request
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"我是 u1,退掉超过 500 的订单"}'

# 打开 Jaeger UI: http://localhost:16686
#   Service 选 "python-agent" → Find Traces → 点开最近一条
#   你应看到一棵树: python-agent 的 span 下,挂着 java-service 的 span,
#   且根 span 与子 span 共享同一个 Trace ID
```

**Step 4 — 用这条 trace 定位一个真实瓶颈**

```text
练习:在 Java 的 refund 里塞一个 Thread.sleep(1500)(模拟慢 DB),重跑。
观察:Jaeger 树里 java: order.refund 这一段明显变长 1.5s,
     而 python 的 llm 段没变 —— 一眼定位"瓶颈在 Java 退款,不在 LLM"。
这正是贯通 trace 的全部价值:把"整体慢"精确归因到"哪栈哪步"。
```

## 3. 今日任务

1. **给你的双栈接上贯通 trace**:Python 侧 instrument httpx(自动注入 `traceparent`),Java 侧确认 W3C 传播 + 加 `@Observed`。
2. **在 Jaeger 看到一棵跨栈的树**:同一 `trace_id` 下,python-agent 的 span 与 java-service 的 span 连成一棵,层级清晰。
3. **做一次归因演练**:分别在 Python 的 LLM 步骤、Java 的 DB 步骤注入延迟,各跑一次,确认你能仅凭 trace 树就指出"这次慢在哪一栈、哪一步"。
4. **检查 span 属性卫生**:确认 span 标签是低基数、不含敏感全文(用户消息要截断/脱敏),避免把 PII 写进 trace。

**验收标准**:Jaeger 里能稳定看到横跨 Python+Java、共享同一 `trace_id` 的完整 span 树;两次注入延迟的演练都能仅凭 trace 正确归因;span 里无敏感原文。

## 4. 自测清单

- [ ] 我能解释 W3C `traceparent` 头如何把两段 trace 缝成一条。
- [ ] 我知道这和 Spring Cloud Sleuth 跨服务传 traceId 是同一套机制。
- [ ] 我理解"自动埋点"做了大部分传播工作,我只需补少量手动 span。
- [ ] 我能仅凭一条跨栈 trace,判断瓶颈在 Python 推理还是 Java 查库。
- [ ] 我注意了 span 属性卫生(低基数、不写 PII)。

## 5. 延伸 & 关联

- 本仓库 评估与监控(trace/指标/告警体系):[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 本仓库 实验管理(把可观测数据沉淀分析):[../08-llm-engineering/03-mlops/01-experiment-tracking.md](../08-llm-engineering/03-mlops/01-experiment-tracking.md)
- 回顾单栈 trace 起点:[Day 58 · 给 Agent 服务加完整监控](./Day-58-capstone-monitoring.md)
- **Phase 5(上)收官**:你已具备"设计→实现两栈→打通→贯通可观测"的完整双栈能力。Day 66~70 将打磨边界、压测优化、并把它收进作品集——这是你区别于纯 AI 工程师的差异化王牌。
- 本系列总计划:[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
