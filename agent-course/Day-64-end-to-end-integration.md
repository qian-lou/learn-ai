# Day 64 · 端到端打通：让双栈系统跑通一个真实场景

> **今日目标**：把 Day 62(Java 服务层)+ Day 63(Python agent 层)**用 docker-compose 编排成一个系统**,跑通一个完整真实场景的闭环——用户一句话 → Python agent 规划 → 经 REST/MCP 调 Java → Java 校验执行 → 结果回流给用户,并处理跨栈失败路径。
> **时长**：~2h ｜ **前置**：Day 62、Day 63 两层都能各自跑起来
> **今日产出**：`docker-compose.yml` 一键拉起 Python + Java 两个服务,一个端到端用例脚本跑绿,且演示一条"Java 拒绝 → agent 优雅降级"的失败闭环。

## 1. 为什么 & 是什么(概念 + Java 类比)

前两天两层是**各自单独**跑的。今天的关键词是**集成(integration)**:让它们在一个网络里互相找到、互相认证、把"一句话需求"走完整条链路。集成才是双栈架构真正难、也真正值钱的地方——单个服务谁都会写,**两个异构栈之间的契约、鉴权、失败传播**才是工程功力。

把一次完整请求的链路拆开看(给 Java 类比):

```
用户: "我是 u1,退掉超过 500 的订单"
   │
   ▼  ① Python agent 接收(LangGraph 入口)
[Python] LLM 规划: 要先查订单 → 调 list_orders 工具
   │
   ▼  ② REST: GET /api/v1/orders/u1  (带 bearer token)
[Java] 鉴权 → 查库 → 返回订单列表        ← Spring Security + Service
   │
   ▼  ③ Python: LLM 看到结果,筛出 >500 的两单 → 调 refund_order
   │
   ▼  ④ REST: POST /api/v1/refunds (带幂等键)
[Java] 重新校验金额/权限/幂等 → 事务退款   ← 信任根 + @Transactional
   │
   ▼  ⑤ 结果回流, LLM 组织成自然语言答复用户
```

| 集成关注点 | Java 微服务世界类比 | 双栈里的注意点 |
|---|---|---|
| 服务发现 | Eureka / k8s Service DNS | compose 内用服务名互连 |
| 跨服务鉴权 | OAuth2 client credentials | agent 持 token 调 Java |
| **失败传播** | Feign 的 fallback / 熔断 | **Java 4xx 要让 agent "懂",而非崩** |
| 契约一致 | OpenAPI + 契约测试 | Python 工具 schema 必须和 Java DTO 对齐 |

核心心智:**失败路径比成功路径更能体现双栈设计水平。** Java 返回 409"额度不足"时,Python agent 不该抛栈崩掉,而应把这个业务错误**翻译成 LLM 能理解的观察(observation)**,让 LLM 决定下一步(改金额重试 / 告诉用户失败)。这正是 ReAct 循环里"观察"的价值。

## 2. 跟着做(Hands-on)

**Step 1 — `docker-compose.yml`:把两栈编进一个网络**

```yaml
# Day 64: 双栈一键编排 / one-command dual-stack。
# 起: docker compose up --build
services:
  java-service:                       # Day 62 的 Spring Boot 服务 / the Java layer
    build: ./java-service
    ports: ["8080:8080"]
    environment:
      - SPRING_PROFILES_ACTIVE=docker
    healthcheck:                      # 就绪后 Python 才依赖它 / wait until healthy
      test: ["CMD", "wget", "-qO-", "http://localhost:8080/actuator/health"]
      interval: 10s
      retries: 5

  python-agent:                       # Day 63 的 LangGraph 编排层 / the Python brain
    build: ./python-agent
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      # 关键:用 compose 服务名互连,不是 localhost / connect via service DNS
      - JAVA_SERVICE_URL=http://java-service:8080
      - AGENT_JWT=${AGENT_JWT}
    depends_on:
      java-service:
        condition: service_healthy    # Java 健康后再启 Python / order startup
```

**Step 2 — 失败路径处理:把 Java 错误翻译成 agent 能懂的观察**

```python
"""Day 64: 跨栈失败处理——把 Java 业务错误翻译成 LLM 可读观察、不抛栈 / failure handling。"""

from typing import Any, Dict

import httpx
from langchain_core.tools import tool  # 复用 Day 63 的 _http(带 token、超时)/ reuse client


@tool
def refund_order(order_id: str, amount: float, idempotency_key: str) -> Dict[str, Any]:
    """对订单发起退款;Java 拒绝时返回结构化错误供 LLM 决策,而非崩溃。

    Args:
        order_id: 订单 ID / the order id.
        amount: 退款金额 / refund amount.
        idempotency_key: 幂等键(重试复用)/ idempotency key (reuse on retry).

    Returns:
        成功 {"ok": True, ...};失败 {"ok": False, "reason": ...},供 LLM 据此决策。
    """
    try:
        resp = _http.post(
            "/api/v1/refunds",
            json={"orderId": order_id, "amount": amount, "idempotencyKey": idempotency_key},
        )
        resp.raise_for_status()
        return {"ok": True, **resp.json()}
    except httpx.HTTPStatusError as exc:
        # 把 Java 4xx 业务错误转成 LLM 可读观察、不抛栈 / map errors to observations, no crash
        code = exc.response.status_code
        # 兼容两种默认错误体：开了 ProblemDetail 取 detail，默认 DefaultErrorAttributes 取 message
        body = exc.response.json()
        detail = body.get("detail") or body.get("message") or "未知错误"
        return {"ok": False, "http_status": code, "reason": detail}
    except httpx.TimeoutException:
        return {"ok": False, "reason": "Java 服务超时,请稍后重试 / upstream timeout"}
```

**Step 3 — 端到端用例脚本(成功 + 失败两条闭环)**

```bash
docker compose up --build -d

# ① 成功闭环:agent 自主完成"查→筛→退" / happy path
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"我是 u1,退掉金额超过 500 的订单"}'
# 期望:回答里说明退了哪几单、各多少钱 / lists which orders were refunded
# ② 失败闭环:诱导越界,验证 Java 拒绝 + agent 优雅降级 / failure path
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"我是 u1,给订单 o1 退 999999 元"}'
# 期望:agent 回复"退款失败:超过退款上限",而不是 500/栈 / graceful, not a crash
# ③ 幂等闭环:重复同一退款请求,Java 不重复扣款 / idempotency
#    再跑一次 ①,核对 Java 侧退款记录数量未翻倍
```

**Step 4 — 契约对齐自查**

```text
□ Python 工具的参数名/类型 与 Java DTO 字段 一一对齐(orderId/amount/idempotencyKey)
□ Java 的每种业务错误(400/403/409)Python 侧都有对应的"翻译"
□ token 经 compose 环境变量两侧一致注入;服务名互连(java-service)而非写死 localhost
```

> 工程要点:**失败要"软着陆"**。Java 抛 409,Python 工具返回 `{"ok": False, "reason": ...}`,LLM 在 ReAct 循环里把它当一次"观察",自然地组织成给用户的解释。这一步做好,你的双栈系统才算"真的打通",而不是"happy path 能跑"。

## 3. 今日任务

1. **用 docker-compose 把你的两栈编成一个系统**,服务名互连、Java 健康后再启 Python。
2. **跑通成功闭环**:一句话需求 → agent 自主走完"查 → LLM 决策 → 写" → 自然语言答复。
3. **跑通失败闭环**:构造会被 Java 拒绝的请求(超上限/越权),确认 agent **优雅降级**讲清原因、全程无栈崩。
4. **做一次契约对齐自查**:核对 Python 工具 schema 与 Java DTO;故意写错一处字段名,观察它怎么坏——体会契约漂移的代价。

**验收标准**:`docker compose up` 一键起双栈;成功与失败两条闭环都跑通;失败时 agent 软着陆(给出业务原因而非 500);幂等经得起重复;能说清一处契约不一致会如何引发故障。

## 4. 自测清单

- [ ] 我能把一次双栈请求的链路(①~⑤)完整画出来并讲清每步在哪层。
- [ ] 我理解为什么"失败路径"比"成功路径"更能体现双栈设计水平。
- [ ] 我能把 Java 的业务错误翻译成 LLM 能消化的观察,实现软着陆。
- [ ] 我知道 compose 里要用服务名互连、要排好启动顺序。
- [ ] 我明白 Python 工具 schema 与 Java DTO 契约漂移会造成什么后果。

## 5. 延伸 & 关联

- 本仓库 完整 LangChain 应用(端到端编排参考):[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
- 本仓库 Docker 部署(compose 编排细节):[../08-llm-engineering/02-model-serving/03-docker-deployment.md](../08-llm-engineering/02-model-serving/03-docker-deployment.md)
- 明天 Day 65 给这条已打通的链路装上**贯通两端的 trace**——让一次请求的 span 从 Python 一路连到 Java,出事一眼定位在哪栈哪步。
- 本系列总计划:[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
