# Day 67 · 完善②：压测 + 加缓存 + 优化延迟

> **今日目标**：给混合系统做一次像样的压测，定位瓶颈，加两层缓存（语义缓存 + Java 结果缓存），把 p95 延迟和单请求成本一起压下来。
> **时长**：~2h ｜ **前置**：Day 66（失败路径已加固）、Day 65（trace 贯通两端）
> **今日产出**：一份《压测前后对比表》（QPS / p50 / p95 / 错误率 / 单请求 token 成本）+ 落地语义缓存与 Java 缓存，并用 trace 指认"省下来的延迟出在哪一段"。

## 1. 为什么 & 是什么（概念 + Java 类比）

性能不是"感觉快不快"，是**数字**。没有压测，你不知道系统在 20 并发下会不会雪崩；没有 trace，你不知道 8 秒延迟里 6 秒花在 LLM、还是花在那个慢 SQL 上。今天做两件事：**先量（压测 + 看 trace 找瓶颈），再治（缓存 + 并行 + 选段降级）**。这也是面试高频追问——"你的 Agent p95 多少？瓶颈在哪？怎么优化的？"

给 Java 工程师的类比，工具和直觉几乎一一对应：

| Agent 系统 | Java 世界类比 | 要点 |
|---|---|---|
| 压测（locust / k6） | JMeter / Gatling | 关注 QPS、p50/p95/p99、错误率，**别只看平均值** |
| 语义缓存（embedding 近似命中） | Spring Cache / Caffeine，但 key 是"语义" | 相似问题直接复用历史答案，省一整次 LLM 调用 |
| Java 结果缓存 | `@Cacheable` + Redis/Caffeine | 工具层 DB 查询结果缓存，避开重复慢查询 |
| 并行工具调用 | `CompletableFuture.allOf` | 互不依赖的工具/子调用并发发起 |
| 瓶颈定位（看 trace 分段耗时） | APM 火焰图（SkyWalking / Arthas trace） | 延迟要拆到"每一段"，否则优化全凭猜 |

**核心心智：延迟是分段累加的，成本是 token 累加的。** Agent 的端到端延迟 ≈ Σ(每次 LLM 调用 + 每次工具/Java 调用 + 网络)。优化的本质是**砍掉某一段**：要么不调（缓存命中）、要么并发调（并行）、要么换更快的段（小模型/直答）。盲目"换个大模型"通常只会更慢更贵。

## 2. 跟着做（Hands-on）

**Step 1 — 先压一遍拿基线（locust，2026 仍是 Python 压测主力）**：

```python
"""Day 67: 用 locust 给混合系统压测 / load-test the hybrid system.

运行 / run:  locust -f locustfile.py --host http://localhost:8000
"""

from locust import HttpUser, task, between


class AgentUser(HttpUser):
    """模拟用户：以接近真实的间隔发问 / a simulated user asking questions."""

    wait_time = between(1, 3)  # 思考间隔，别打成纯压力洪流 / human-like pacing

    @task
    def ask(self) -> None:
        # 打 Agent 的 HTTP 入口；name 用于聚合统计 / group stats by name
        self.client.post(
            "/agent/ask",
            json={"q": "查一下用户 42 最近已支付的订单"},
            name="POST /agent/ask",
        )
```

跑 `locust`，在 Web UI 把并发拉到 10→20→50，**记录基线**：QPS、p50/p95、错误率。然后打开 Day 65 的 trace，找出"哪一段最长"——通常不是网络，是某次 LLM 调用或某个慢工具。

**Step 2 — 加语义缓存（相似问题不重复烧 LLM）**：

```python
"""语义缓存：用 embedding 近似命中，复用历史答案 / semantic cache via embeddings."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

client = OpenAI()
_EMBED_MODEL = "text-embedding-3-small"
_THRESHOLD = 0.92  # 余弦相似度阈值，过低会"张冠李戴" / too low → wrong reuse


class SemanticCache:
    """极简语义缓存：命中则跳过整次 Agent 调用。
    Minimal semantic cache; a hit skips a whole Agent invocation.
    """

    def __init__(self) -> None:
        self._vecs: list[np.ndarray] = []   # shape: 每条 (d,)
        self._answers: list[str] = []

    def _embed(self, text: str) -> np.ndarray:
        v = client.embeddings.create(model=_EMBED_MODEL, input=text).data[0].embedding
        v = np.asarray(v, dtype=np.float32)        # (d,)
        return v / (np.linalg.norm(v) + 1e-8)      # 归一化便于点积即余弦 / unit-norm

    def get(self, q: str) -> str | None:
        # 时间 O(N·d) 空间 O(1)：与所有缓存向量算余弦取最大
        # cosine vs all cached vectors, take the best
        if not self._vecs:
            return None
        qv = self._embed(q)                         # (d,)
        sims = np.array([float(qv @ v) for v in self._vecs])  # (N,)
        best = int(sims.argmax())
        return self._answers[best] if sims[best] >= _THRESHOLD else None

    def put(self, q: str, answer: str) -> None:
        self._vecs.append(self._embed(q))
        self._answers.append(answer)
```

> 工程提醒：上千条以上别用 Python 列表线性扫，换向量库做 ANN 检索（pgvector / Faiss），把 O(N·d) 降到近似 O(log N)。语义缓存命中率高，但**阈值要保守**——宁可少命中，也别把"用户 42 的订单"答成"用户 24 的"。

**Step 3 — Java 侧给慢查询加缓存**（工具层最常见的瓶颈）：

```java
// 工具层 DB 查询加缓存，避开 Agent 反复触发的同一慢 SQL
// cache the DB query so repeated Agent calls skip the slow SQL
@Cacheable(value = "orderQuery", key = "#dto.userId + ':' + #dto.status")
public OrderResultVO query(OrderQueryDTO dto) {
    return orderMapper.selectByUserAndStatus(dto.getUserId(), dto.getStatus());
}
```

**Step 4 — 并行 + 选段降级**：互不依赖的工具用 `asyncio.gather` 并发发起；对"简单问句"在路由层直接走小模型甚至直答（呼应 Day 52 智能路由），把最贵的那一段从热路径上摘掉。

**Step 5 — 复压一遍，做对比**：同样的并发梯度再跑一次 locust，填这张表，用数字说话：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| QPS（@20 并发） | … | … |
| p50 / p95 延迟 | … | … |
| 错误率 | … | … |
| 单请求 token 成本 | … | … |
| 语义缓存命中率 | — | … |

## 3. 今日任务

1. **拿基线**：用 locust 在 10/20/50 并发下压测，记录 QPS、p50/p95、错误率。
2. **指认瓶颈**：打开两端贯通的 trace，明确写出"延迟主要花在哪一段"（LLM？某工具？某 SQL？）。
3. **加两层缓存**：落地语义缓存（Python）+ `@Cacheable`（Java），并各自验证命中。
4. **再优化一处热路径**：并行工具调用 **或** 简单问句走小模型/直答，二选一做掉。
5. **出对比表**：复压一遍，填满上面的「优化前/后」表格。

**验收标准**：①有优化前/后两组真实压测数字；②能基于 trace 指出瓶颈段位（不靠猜）；③语义缓存与 Java 缓存均可见命中；④p95 或单请求成本至少一项有可量化下降，且错误率未恶化。

## 4. 自测清单

- [ ] 我能报出系统的 p95 延迟和单请求 token 成本（不是"感觉挺快"）。
- [ ] 我能基于 trace 把端到端延迟拆到"每一段"，并指出最长的那段。
- [ ] 我理解语义缓存阈值过低的风险（答非所问），并把它设得偏保守。
- [ ] 我知道"换更大模型"常常更慢更贵，优化应优先砍段/并行/缓存。
- [ ] 我的优化是**用数字验证**的，而非自我感觉。

## 5. 延伸 & 关联

- 推理加速 / 吞吐优化的系统手段：[../08-llm-engineering/01-model-optimization/02-inference-acceleration.md](../08-llm-engineering/01-model-optimization/02-inference-acceleration.md)
- API 服务的限流/超时/并发关注点：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 评估 + 监控（把延迟/成本纳入持续观测）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 向量库（语义缓存上量后的落点）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
- 总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)

> 衔接 Day 68：系统已"又稳又快"。接下来三天从"写代码"切到"讲清楚"——先把三个主力项目收拢成一套能拿去面试的**作品集**。
