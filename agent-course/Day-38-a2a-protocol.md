# Day 38 · A2A 协议：Agent 间通信标准，解决什么问题

> **今日目标**：搞清 A2A（Agent-to-Agent）协议解决什么问题、和 MCP 的分工，跑通一个最小的 A2A client→server 调用。
> **时长**：~2h ｜ **前置**：Day 37（MCP）、Day 32（多 Agent 流水线）
> **今日产出**：一个 `day38_a2a_demo.py`，启动一个对外发布 Agent Card 的 A2A server，并用 client 发现它、给它派一个任务、拿回结果。

## 1. 为什么 & 是什么

昨天的 MCP 解决"一个 agent 怎么用工具"。但真实系统里，一个能力（比如"做财报分析"）往往本身就是**另一个团队、另一个进程、甚至另一家公司**的 agent。你不想知道它内部用什么模型、什么框架，你只想：**发现它 → 给它派活 → 拿回结果**。这就是 **A2A（Agent2Agent Protocol）** 解决的问题——2025 年由 Google 牵头、后捐给 Linux 基金会的开放协议，专门规范**异构 agent 之间**的通信。

一句话对比，这是面试高频考点：

| 维度 | **MCP** | **A2A** |
|---|---|---|
| 连接的是 | agent ↔ **工具/数据源** | agent ↔ **另一个 agent** |
| 对端是什么 | 被动的能力（函数、资源） | 自主的、可能多轮交互的对等体 |
| 类比 | JDBC（调数据库） | **微服务间的 REST/gRPC 调用** |
| 核心交换物 | tool 调用结果 | **Task**（带生命周期的任务）|
| 谁暴露能力 | server 列举 tools | server 发布 **Agent Card** |

A2A 的几个关键概念：

- **Agent Card**：一个 `/.well-known/agent-card.json`，相当于 agent 的"服务发现 + API 文档"——声明它叫什么、会什么技能（skills）、用什么鉴权。对应 Java 世界的 **服务注册中心里的一条目录项 + OpenAPI**。
- **Task**：A2A 的核心单元。一次委派是一个有状态的 Task（`submitted → working → input-required → completed/failed`），支持长耗时、流式、甚至中途要求补充输入。比"一次 RPC 返回"强，更像**异步作业**。
- **Message / Artifact**：交互中的消息和最终产物。

**何时用 A2A、何时不用**：如果对端是你自己进程里的一个函数/子 agent，别上 A2A，直接函数调用最省事（明天 Day 39 就是这种）。**只有当对端是独立部署、跨团队/跨语言/跨网络的自治 agent 时，A2A 的"标准化发现 + 任务生命周期"才值回成本。** 杀鸡别用牛刀。

## 2. 跟着做（Hands-on）

用官方 `a2a-sdk`。我们起一个最小 server（暴露一个"摘要"技能），再用 client 发现并委派任务。

```bash
pip install "a2a-sdk>=0.2"   # 官方 A2A SDK / official A2A SDK
```

**Server 端：发布 Agent Card + 一个技能**

```python
"""Day 38: 最小 A2A server / a minimal A2A server exposing one skill."""

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message


class SummarizerExecutor(AgentExecutor):
    """收到任务就返回一句"摘要"（演示桩）/ returns a fake summary per task."""

    async def execute(self, ctx: RequestContext, queue: EventQueue) -> None:
        user_text = ctx.get_user_input()  # 取入参 / pull the incoming text
        reply = f"摘要：{user_text[:20]}…（共 {len(user_text)} 字）"
        await queue.enqueue_event(new_agent_text_message(reply))

    async def cancel(self, ctx: RequestContext, queue: EventQueue) -> None:
        raise NotImplementedError  # 本例不支持取消 / cancel unsupported here


# Agent Card：对外的"服务目录项" / the public service descriptor
card = AgentCard(
    name="summarizer-agent",
    description="把长文本压成一句话 / summarize long text",
    url="http://localhost:9999/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="summarize",
            name="summarize",
            description="文本摘要 / text summarization",
            tags=["nlp"],
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
)

handler = DefaultRequestHandler(
    agent_executor=SummarizerExecutor(),
    task_store=InMemoryTaskStore(),  # 任务状态存储 / task lifecycle store
)
app = A2AStarletteApplication(agent_card=card, http_handler=handler)

if __name__ == "__main__":
    uvicorn.run(app.build(), host="0.0.0.0", port=9999)
```

**Client 端：发现 → 委派任务 → 拿结果**

```python
"""Day 38: A2A client 发现并委派任务 / discover an agent and delegate a task."""

import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import Message, Part, TextPart, Role


async def main() -> None:
    async with httpx.AsyncClient() as http:
        # 1) 拉取 Agent Card（服务发现）/ fetch the Agent Card
        resolver = A2ACardResolver(http, base_url="http://localhost:9999")
        card = await resolver.get_agent_card()
        print("发现 agent / discovered:", card.name, "skills:", [s.id for s in card.skills])

        # 2) 建客户端并发一条消息（委派任务）/ build client and send a task message
        client = ClientFactory(ClientConfig(httpx_client=http)).create(card)
        msg = Message(
            role=Role.user,
            message_id=uuid4().hex,
            parts=[Part(TextPart(text="A2A 让异构 agent 能互相委派任务，无需知道对方内部实现。"))],
        )
        # 3) 流式收结果 / stream back the result
        async for event in client.send_message(msg):
            print("收到 / got:", event)


if __name__ == "__main__":
    asyncio.run(main())
```

先跑 server，另开一个终端跑 client。你会看到 client 先打印发现到的 agent 名与技能，再收到"摘要：…"。

## 3. 今日任务

1. 两个终端分别跑通 server 与 client，确认能打印出 Agent Card 的 `name/skills` 并收到摘要回复。
2. **看协议本体**：浏览器打开 `http://localhost:9999/.well-known/agent-card.json`，对照它和你代码里 `AgentCard` 的字段——理解"这就是 agent 的服务发现文档"。
3. **画一张分工图**：用一句话各写 MCP 与 A2A 解决的问题，并举一个"该用 A2A"和一个"不该用 A2A（直接函数调用即可）"的具体例子。

**验收标准**：client 能发现 server 并完成一次任务往返；能在浏览器里看到 Agent Card JSON；能清楚说出 A2A vs MCP 的边界，以及何时不该上 A2A。

## 4. 自测清单

- [ ] 我能一句话区分 MCP（agent↔工具）和 A2A（agent↔agent）。
- [ ] 我理解 Agent Card 相当于"服务发现 + API 文档"。
- [ ] 我知道 A2A 的核心是带生命周期的 **Task**，不是一次性 RPC。
- [ ] 我能判断一个场景该不该上 A2A（对端是否独立自治）。
- [ ] 我能把 A2A 类比成微服务间的 REST/gRPC 调用并讲清差异（自治体、任务态）。

## 5. 延伸 & 关联

- 本课程 Day 37（MCP）：与今天互补，建议放一起记忆。
- 本课程 Day 32–33（多 Agent 流水线 / 通信与共享状态）：进程内的多 agent 协作；A2A 是它的"跨进程标准化版本"。
- API 服务化基础（A2A server 本质是个 HTTP 服务）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 明天 Day 39：在**一个进程内**做"工具 + 子 agent"的复杂编排——刻意不上 A2A，体会两者的取舍。
