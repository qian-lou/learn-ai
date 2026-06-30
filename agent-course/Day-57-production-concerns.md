# Day 57 · 生产关注点：限流 · 超时 · 并发 · 降级 · 健康检查

> **今日目标**：给 Day 56 的 agent 服务套上"五件护具"——**超时、并发上限、限流、降级、健康检查**，让它在异常和高压下不崩、可预测。
> **时长**：~2h ｜ **前置**：Day 56（已有 FastAPI agent 服务）
> **今日产出**：升级版 `app/main.py`，超长请求自动超时返回 504、超并发返回 429、LLM 挂了走降级、`/readyz` 真实探测依赖。

## 1. 为什么 & 是什么（概念 + Java 类比）

"能跑"和"能上线"之间隔着这五件事。Agent 服务有个独特风险：**每次请求都依赖外部 LLM**——它可能慢（30s+）、可能限流你（429）、可能宕。没有护具，一个慢请求能拖垮整个进程。五件护具，逐个给 Java 类比：

| 护具 | 解决什么 | Java 世界类比 |
|---|---|---|
| **超时（timeout）** | LLM 卡住时不无限等 | `RestTemplate` 的 `connect/readTimeout`、`@Transactional(timeout)` |
| **并发上限（bulkhead）** | 在途请求太多打爆内存/连接 | 线程池 + `Semaphore` 隔离舱 / Resilience4j `Bulkhead` |
| **限流（rate limit）** | 单客户端刷爆你 | 网关限流 / Resilience4j `RateLimiter`、Sentinel |
| **降级（fallback）** | 依赖挂了给个兜底而非 500 | Resilience4j `CircuitBreaker` + fallback、Hystrix |
| **健康检查（health）** | 让 k8s 知道何时重启/摘流量 | Spring Boot Actuator `/health`（liveness/readiness 分离） |

一个关键心智：**liveness ≠ readiness**。

- **liveness（存活，`/healthz`）**：进程还活着吗？挂了就**重启**。必须轻量、绝不打外部依赖——否则 LLM 抖一下你的 pod 就被反复重启。
- **readiness（就绪，`/readyz`）**：现在能接流量吗？依赖（LLM/DB）不通就**摘流量但不重启**，这里**才**去探测下游。把这俩搞反，是新手部署 agent 最常见的事故来源。

## 2. 跟着做（Hands-on）

**Step 1 — 装一个轻量限流库**（也可纯手写，这里用成熟件）

```bash
pip install "slowapi>=0.1.9"   # 基于 limits，FastAPI 友好 / rate limiting for FastAPI
```

**Step 2 — 五件护具的核心实现**

```python
"""Day 57: 生产护具 / production guards——超时·并发·限流·降级·健康检查。"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# 并发隔离舱：全局信号量限住同时在途的 LLM 调用数 / bulkhead: cap concurrent in-flight LLM calls
MAX_CONCURRENCY = 32
llm_gate = asyncio.Semaphore(MAX_CONCURRENCY)
LLM_TIMEOUT_S = 25.0  # 单次 LLM 调用硬超时 / hard per-call timeout

clients: dict[str, AsyncOpenAI] = {}
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])  # 按来源 IP 限流 / rate limit by IP

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """初始化带超时配置的客户端 / build client with timeouts."""
    clients["llm"] = AsyncOpenAI(timeout=LLM_TIMEOUT_S, max_retries=1)
    yield
    await clients["llm"].close()

app = FastAPI(title="Agent Service (hardened)", lifespan=lifespan)
app.state.limiter = limiter
MODEL = "gpt-4o-mini"

class ChatRequest(BaseModel):
    """聊天请求 / chat request."""

    message: str = Field(min_length=1, max_length=4000)

@app.exception_handler(Exception)
async def ratelimit_handler(request: Request, exc: Exception) -> JSONResponse:
    """把 slowapi 限流异常转成 429，其它异常交回默认处理 / map rate-limit errors to 429, re-raise others."""
    if exc.__class__.__name__ == "RateLimitExceeded":
        return JSONResponse(status_code=429, content={"detail": "请求过于频繁 / too many requests"})
    raise exc

async def call_llm_guarded(message: str) -> tuple[str, bool]:
    """带超时 + 并发闸 + 降级地调用 LLM，返回 (答案文本, 是否降级) / call LLM guarded, returns (answer, degraded)."""
    # 隔离舱：拿不到名额说明满载，立刻降级而非排队等死 / bulkhead: shed load immediately when saturated
    if llm_gate.locked() and llm_gate._value == 0:
        return "系统繁忙，请稍后再试 / busy, try later", True
    async with llm_gate:
        try:
            # 双保险超时：客户端 timeout + asyncio.wait_for 外层兜底 / belt-and-suspenders timeout
            resp = await asyncio.wait_for(
                clients["llm"].chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": message}],
                ),
                timeout=LLM_TIMEOUT_S + 2,
            )
            return resp.choices[0].message.content, False
        except (APITimeoutError, asyncio.TimeoutError):
            return "（响应超时，已降级）请稍后重试 / timed out", True  # 降级 / fallback
        except APIError:
            return "（上游异常，已降级）服务暂不可用 / upstream error", True

@app.post("/chat")
@limiter.limit("60/minute")  # 端点级限流 / per-endpoint rate limit
async def chat(request: Request, req: ChatRequest) -> dict[str, object]:
    """带全套护具的同步端点 / guarded blocking endpoint."""
    answer, degraded = await call_llm_guarded(req.message)
    return {"answer": answer, "status": "degraded" if degraded else "ok"}

# ============ 健康检查：liveness 与 readiness 分离 ============
@app.get("/healthz")
async def liveness() -> dict[str, str]:
    """存活探针：进程活着即 200，绝不打外部依赖 / liveness, never touch deps."""
    return {"status": "alive"}

@app.get("/readyz")
async def readiness() -> JSONResponse:
    """就绪探针：真实探测下游，200=可接流量 / 503=依赖不通（摘流量但不重启）/ readiness probe."""
    try:
        # 轻量探测：models.list 比一次 completion 便宜 / cheap probe
        await asyncio.wait_for(clients["llm"].models.list(), timeout=3.0)
        return JSONResponse(status_code=200, content={"ready": True})
    except Exception:  # noqa: BLE001 —— 探针处吞所有异常是合理的 / intentional broad catch
        return JSONResponse(status_code=503, content={"ready": False})
```

**Step 3 — 验证护具真的生效**

```bash
# 1) 限流：连发 70 次，后面应出现 429 / hammer to trigger 429
for i in $(seq 1 70); do
  curl -s -o /dev/null -w "%{http_code} " -X POST localhost:8000/chat \
    -H 'Content-Type: application/json' -d '{"message":"hi"}'
done; echo
# 2) 就绪探针：临时改错 OPENAI_BASE_URL 重启，/readyz 应返回 503 / break base_url, expect 503
curl -s -o /dev/null -w "readyz=%{http_code}\n" localhost:8000/readyz
# 3) 超时降级：把 LLM_TIMEOUT_S 调到 0.001 重启，/chat 应返回 status=degraded
```

> 工程要点：`asyncio.Semaphore` 是**进程内**的并发闸。多 worker / 多副本时每进程各有一份——要全局并发上限，得上**网关层限流**或共享 Redis 计数（明天阶段项目会提到外部化）。

## 3. 今日任务

1. 把五件护具全套到你 Day 56 的服务上：**超时、并发闸、限流、降级、`/readyz`**。
2. **造三次故障验证降级**：①超时调到极小 → `status=degraded`；②连发请求触发 429；③base_url 改错 → `/readyz` 返 503 而 `/healthz` 仍 200。
3. **画一张决策表**（写进文件注释）：每种故障（慢/限流/宕/参数非法）返回什么 HTTP 码、降级还是报错——面试高频追问。
4. **思考并写下**：`MAX_CONCURRENCY` 该设多少？依据是什么（LLM 供应商 RPM/TPM 配额 ÷ 副本数）？

**验收标准**：三种故障各能复现一次且行为符合预期；依赖挂时 `/healthz` 与 `/readyz` 一个 200 一个 503；决策表覆盖至少 4 种异常场景。

## 4. 自测清单

- [ ] 能讲清 liveness 与 readiness 的区别、搞反会出什么事，以及为什么 `/healthz` 绝不能打外部依赖。
- [ ] 能说出"超时、隔离舱、限流、熔断/降级"分别对应 Resilience4j 的哪个组件。
- [ ] 理解 `asyncio.Semaphore` 是进程内的，多副本下全局限流要靠网关/Redis。
- [ ] 能为不同故障给出正确的 HTTP 状态码（429/504/503/422）。

## 5. 延伸 & 关联

- 本仓库 API 服务开发（限流/超时等更系统）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 本仓库 评估与监控（健康检查接入监控）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 明天起 Day 58~60 是**阶段项目**：给这个已加固的服务补齐监控 → eval + 安全 → 打包成"能上线"的证明。
- 本系列总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
