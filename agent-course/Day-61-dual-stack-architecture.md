# Day 61 · 双栈架构设计：Python 编排层 + Java 服务层

> **今日目标**：搞清**为什么**生产 AI 系统要拆成"Python 编排层 + Java 服务层"，并**定清两层之间的契约**——什么走 REST、什么走 MCP，画出架构图、定义接口 schema。这是 Phase 5 的地基。
> **时长**：~2h ｜ **前置**：Day 6~45（agent/LangGraph）、Day 37（MCP 概念）、你的 Java/Spring 底子
> **今日产出**：一张分层架构图（文字版即可）、一份接口契约（OpenAPI 片段 + MCP tool 定义）、一句话说清"哪些能力归 Java、哪些归 Python"。

## 1. 为什么 & 是什么（概念 + Java 类比）

到这儿你已经能用 Python 把 agent 送上线了。但**真实企业系统里，业务逻辑、数据、权限往往早就沉淀在 Java/Spring 体系里了**——你不可能、也不该用 Python 重写订单系统、风控规则、用户权限。

于是出现了 2024-2026 越来越主流的**双栈架构**：

```
┌─────────────────────────────────────────────────────────┐
│  Python 编排层 (Orchestration)  —— LangGraph / Agent      │
│   职责: LLM 推理 · 工具选择 · 多步规划 · RAG · 流式        │
│   "大脑": 决定下一步做什么                                 │
└───────────────┬─────────────────────────┬────────────────┘
                │ REST (同步业务调用)       │ MCP (工具发现/调用)
                ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│  Java 服务层 (Services)  —— Spring Boot                   │
│   职责: 业务规则 · 数据库 · 鉴权 · 事务 · 可观测 · 缓存    │
│   "肌肉": 可靠地把事情做对（ACID、权限、审计）             │
└─────────────────────────────────────────────────────────┘
```

**为什么这样分？** 一句话:**让 LLM 做它擅长的"模糊决策",让 Java 做它擅长的"确定性执行"。** 给你的精确类比:

| 关注点 | 该放哪层 | 为什么 |
|---|---|---|
| LLM 调用 / prompt / 多步规划 | **Python** | 生态在这（LangGraph/OpenAI SDK），迭代快 |
| 事务一致性（下单扣库存） | **Java** | `@Transactional` + ACID,LLM 给不了这个保证 |
| 鉴权 / 权限边界 | **Java** | Spring Security 成熟,且**绝不能让 LLM 决定谁有权限** |
| 业务规则 / 风控 / 计价 | **Java** | 已有代码资产 + 需可审计、可单测、确定性 |
| 数据库读写 / 缓存 | **Java** | 连接池、ORM、慢查询治理都在 Java 侧 |
| RAG 检索 / 向量 | Python（或调 Java 的检索服务） | 嵌入生态在 Python |

**核心心智(最重要的一条):把 LLM 当成一个"不可信的初级实习生"。** 它能提议"给用户 A 退款 500 元",但**真正执行退款的是 Java 服务,Java 会重新校验金额、权限、幂等**。Python 层只负责"想",Java 层负责"做对且做安全"。这正是你 Java 工程师的护城河——你比纯 AI 工程师更懂这层该怎么设计。

## 2. 跟着做（Hands-on）：定清两层契约

架构设计日的"hands-on"是**把接口契约写死**,而不是急着写实现。两种通道,各有所长:

**通道一:REST —— 用于明确的、点对点的业务调用**

```yaml
# order-service OpenAPI 片段 / contract for the Java side (REST)
# Python agent 通过它调用 Java 的查询/下单能力
paths:
  /api/v1/orders/{userId}:
    get:
      summary: 查用户订单 / list a user's orders
      security: [{ bearerAuth: [] }]          # 鉴权在 Java 侧强制 / auth enforced in Java
      parameters:
        - { name: userId, in: path, required: true, schema: { type: string } }
      responses:
        "200":
          description: 订单列表 / order list
          content:
            application/json:
              schema:
                type: object
                properties:
                  orders:
                    type: array
                    items: { $ref: "#/components/schemas/Order" }
  /api/v1/refunds:
    post:
      summary: 发起退款（强校验+幂等）/ issue a refund (validated, idempotent)
      security: [{ bearerAuth: [] }]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [orderId, amount, idempotencyKey]   # 幂等键防重复退款 / idempotency
              properties:
                orderId: { type: string }
                amount: { type: number, minimum: 0 }
                idempotencyKey: { type: string }            # LLM 重试不会重复扣钱 / safe retries
```

**通道二:MCP —— 用于把 Java 能力"暴露成 agent 可发现的工具"**

REST 是"你得提前知道有这个接口"。**MCP(Model Context Protocol)让 agent 在运行时"发现"有哪些工具可用**——更适合工具集会变、想让 agent 自主选工具的场景。Java 侧用 MCP server 暴露工具:

```json
// MCP tool 定义（Java MCP server 暴露给 Python agent）
// the Java side advertises this tool; the agent discovers & calls it
{
  "name": "query_inventory",
  "description": "查询某 SKU 的实时库存。只读、无副作用。/ read-only stock lookup",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sku": { "type": "string", "description": "商品编码 / product SKU" }
    },
    "required": ["sku"]
  }
}
```

**怎么选 REST 还是 MCP?** 决策原则:

| 选 REST | 选 MCP |
|---|---|
| 接口固定、调用方明确 | 工具集会增减、想让 agent 自主发现 |
| 高频、低延迟业务调用(下单/查询) | LLM 驱动的工具调用(让模型决定调哪个) |
| 已有大量 Spring REST 接口,直接复用 | 想把多个内部能力统一成"工具市场" |
| 严格契约 + 版本管理(OpenAPI) | 动态、自描述、即插即用 |

> 实战常见做法:**两者并存**。核心业务(下单、退款、鉴权)走 REST,严格可控;一批只读查询能力(查库存、查物流、查 FAQ)走 MCP,让 agent 灵活组合。今天先把这条"分界线"在你的场景里画清楚。

## 3. 今日任务

1. **选一个真实业务场景**(建议:电商客服 agent / 内部数据查询助手),写一句话说清它要做什么。
2. **画分层架构图**(文字版即可):标清 Python 编排层与 Java 服务层各自的职责,以及它们之间用哪条通道。
3. **定契约**:为你的场景写 ①至少 2 个 REST 接口的 OpenAPI 片段(其中一个是有副作用的写操作,必须带幂等键 + 鉴权);②至少 1 个 MCP 只读工具定义。
4. **画"信任边界"**:明确列出"哪些决定 LLM 可以提议,但必须由 Java 重新校验后才执行"(至少 3 条,如金额、权限、库存)。

**验收标准**:架构图能让人一眼看懂两层职责;每个写操作接口都带幂等键和鉴权;能清楚回答"为什么鉴权和事务必须在 Java 侧,而不是 Python 侧"。

## 4. 自测清单

- [ ] 我能用一句话讲清双栈架构"为什么"这么分(模糊决策 vs 确定性执行)。
- [ ] 我能判断一个能力该归 Python 还是 Java(用上面那张职责表)。
- [ ] 我理解"把 LLM 当不可信实习生"——它提议、Java 校验后才执行。
- [ ] 我能说出 REST 与 MCP 各自的适用场景,以及为什么常常并存。
- [ ] 我知道为什么写操作必须带幂等键(防 LLM 重试导致重复扣款)。

## 5. 延伸 & 关联

- 本仓库 LangChain Agents 与工具(Python 编排层的能力来源):[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- 本仓库 API 服务开发(REST 契约设计基础):[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 明天 Day 62 动手实现 **Java 服务层**(Spring Boot 提供业务/数据/鉴权/可观测),把今天的契约变成真代码。
- 本系列总计划:[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
