# Day 66 · 完善①：混合系统的边界情况与失败路径

> **今日目标**：给「Python 编排层 + Java 服务层」混合系统补齐失败路径——超时、降级、部分失败、幂等，让它在生产里"坏得体面"。
> **时长**：~2h ｜ **前置**：Day 61~65（混合架构已端到端跑通 + trace 贯通两端）
> **今日产出**：一份《故障模式清单（FMEA-lite）》+ 在 Python 调用 Java 的边界上落地 3 个防护（超时/重试、断路器、降级回退），并补一条"Java 挂了但 Agent 不崩"的 e2e 验证。

## 1. 为什么 & 是什么（概念 + Java 类比）

Demo 和生产的差距，几乎全在**失败路径**上。Day 64 你让混合系统"跑通了一次 happy path"——必要但远不够。真实世界里：Java 服务会超时、会 503、网络会抖、模型会返回乱七八糟的 tool 参数、同一请求会因重试被执行两次。**会写 demo 的人停在 happy path，能上线的人把失败路径也写完。** 这正是 Day 70 你要和面试官讲的差异化。

给 Java 工程师的类比——这些你其实早就会，只是要把它"搬到 Agent 边界上"：

| Agent 混合系统 | Java/Spring 世界类比 | 要点 |
|---|---|---|
| Python 调 Java 的超时 + 重试 | `RestTemplate`/`WebClient` 超时 + Spring Retry | LLM 步骤本就慢，下游再无超时 = 整链卡死 |
| 断路器（circuit breaker） | Resilience4j `CircuitBreaker` | Java 连续失败时**快速失败**，别让 Agent 空转烧 token |
| 降级回退（fallback） | Resilience4j `@Fallback` / Hystrix | Java 业务挂了，Agent 给"暂查不到，基于已知作答"而非崩溃 |
| 幂等键（idempotency key） | 接口幂等设计（唯一键 + 去重表） | Agent 重试会重复调 Java；写操作必须幂等 |
| 失败模式清单 | FMEA / 故障演练 | 上线前系统性列出"哪里会坏、坏了怎么办" |

**核心心智：把"边界"当一等公民。** 混合系统最脆弱的地方不是 Python 内部、也不是 Java 内部，而是**两者之间那根网络调用**。它同时具备"LLM 不确定性"和"分布式调用不可靠性"两种风险，必须显式加固。

## 2. 跟着做（Hands-on）

**Step 1 — 先列清单，再写代码（FMEA-lite）**。别急着改代码，先用「失败模式 → 触发 → 影响 → 对策」把风险想清楚，例如：Java 超时（DB 慢查询 → 整链卡死 → readTimeout + 重试上限）、Java 返回 5xx（过载 → 工具结果缺失 → 断路器 + 降级文案）、模型给出非法 tool 参数（幻觉/schema 漂移 → Java 400 → 调用前校验 + 回喂自纠）、重复写（重试同一步 → 数据写两次 → 幂等键去重）、部分失败（多工具中 1 个挂 → 结果不完整但可用 → 标注"部分结果"不整体失败）。

**Step 2 — 在 Python 调用边界落地防护（2026 现代写法，`httpx` + `tenacity`）**：

```python
"""Day 66: Python→Java 调用边界的失败防护 / hardening the call boundary."""

from __future__ import annotations
import time

import httpx
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

JAVA_BASE = "http://localhost:8080"


class OrderQuery(BaseModel):
    """模型要传给 Java 的参数 / the tool args the model produced."""
    user_id: int = Field(gt=0)
    status: str = Field(pattern="^(paid|shipped|refunded)$")


class _Breaker:
    """极简断路器：连续失败 N 次则在冷却期内快速失败 / minimal circuit breaker."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 10.0) -> None:
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self._fails = 0
        self._open_until = 0.0

    def allow(self) -> bool:
        # 时间 O(1) 空间 O(1)：处于熔断期则拒绝 / reject while open
        return time.monotonic() >= self._open_until

    def record(self, ok: bool) -> None:
        self._fails = 0 if ok else self._fails + 1
        if self._fails >= self.threshold:  # 连续失败触发熔断 / trip open
            self._open_until = time.monotonic() + self.cooldown_s


_breaker = _Breaker()


@retry(  # 仅对可重试的网络错误退避重试 / retry only transient errors
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
def _post(path: str, json: dict, idem_key: str) -> dict:
    # 显式超时 + 幂等键，缺一不可 / explicit timeout + idempotency key
    resp = httpx.post(
        f"{JAVA_BASE}{path}",
        json=json,
        headers={"Idempotency-Key": idem_key},  # Java 端据此去重 / server dedups
        timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0),
    )
    resp.raise_for_status()  # 4xx/5xx 抛错，进入降级 / raise to trigger fallback
    return resp.json()


def query_orders(raw_args: dict) -> dict:
    """Agent 工具入口：校验→断路→重试→降级 / validate, break, retry, fallback.

    Args:
        raw_args: 模型产出的原始参数 / raw tool args from the model.

    Returns:
        结果字典，含 ``degraded`` 标志位 / result dict with a degraded flag.
    """
    # 1) 调用前强类型校验，把模型幻觉挡在 Java 之外 / validate before the call
    try:
        args = OrderQuery.model_validate(raw_args)
    except ValidationError as e:
        # 不硬失败：把错误回喂模型让它自纠 / feed error back for self-correction
        return {"degraded": True, "reason": "invalid_args", "detail": e.errors()}

    # 2) 断路器：Java 连续挂时快速失败，别烧 token / fail fast when open
    if not _breaker.allow():
        return {"degraded": True, "reason": "circuit_open", "data": None}

    # 3) 调用 + 幂等键（同一逻辑请求复用同一 key）/ stable idem key per logical req
    idem_key = f"orders-{args.user_id}-{args.status}"
    try:
        data = _post("/api/orders/query", args.model_dump(), idem_key)
        _breaker.record(ok=True)
        return {"degraded": False, "data": data}
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
        # 4) 降级回退：Java 挂了，Agent 仍能优雅作答 / graceful degradation
        _breaker.record(ok=False)
        return {"degraded": True, "reason": type(e).__name__, "data": None}
```

**Step 3 — Java 侧把幂等键用起来**：上面的 `Idempotency-Key` 头要在服务端落地——`@PostMapping` 方法用 `@RequestHeader("Idempotency-Key")` 取键，交给一个去重服务 `executeOnce(key, () -> orderService.query(dto))`，命中去重表则直接返回上次结果。接口幂等是写操作的底线，Agent 重试才不会造成重复副作用。

**Step 4 — 验证"坏得体面"**：把 Java 服务停掉（或返回 503），跑一遍 Agent，确认它**返回降级文案而不是抛栈**；连续打 4 次，确认第 4 次直接走断路器快速失败（延迟应骤降）。

## 3. 今日任务

1. **写故障清单**：对你的混合系统列出 ≥5 个失败模式（用上面的 FMEA 表格式），每条写明"触发/影响/对策"。
2. **落地 3 个防护**：在 Python→Java 调用边界实现 (a) 显式超时 + 退避重试、(b) 断路器、(c) 降级回退；写操作补幂等键。
3. **校验前置化**：在调用 Java 前用 Pydantic 校验模型产出的 tool 参数，非法时**回喂模型自纠**而不是直接 500。
4. **混沌小演练**：手动制造 3 类故障（停服 / 注入 503 / 喂非法参数），各跑一次，记录 Agent 的实际行为。

**验收标准**：①清单 ≥5 条且对策具体；②停掉 Java 后 Agent 返回降级结果而非崩溃；③连续失败后断路器生效（延迟明显下降、不再烧 token）；④非法参数被挡在 Java 之外并触发模型自纠。

## 4. 自测清单

- [ ] 我能说清混合系统里"最脆弱的是两者之间那根调用"，以及它为何同时有两类风险。
- [ ] 我的每个跨服务调用都有**显式超时**，且重试有上限。
- [ ] 断路器在 Java 连续失败时能快速失败，避免 Agent 空转烧 token。
- [ ] 写操作带幂等键，Agent 重试不会造成重复副作用。
- [ ] 失败时 Agent 走"降级文案"，且会在响应里**显式标注**这是降级/部分结果。

## 5. 延伸 & 关联

- 生产关注点（限流/超时/降级/健康检查）回顾：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 容器化部署与隔离（为明天压测做准备）：[../08-llm-engineering/02-model-serving/03-docker-deployment.md](../08-llm-engineering/02-model-serving/03-docker-deployment.md)
- 评估 + 监控（失败路径也要纳入回归）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)

> 衔接 Day 67：今天把系统变得"打不死"；明天给它**测压力、加缓存、压延迟**，把"不崩"升级成"又快又稳"。
