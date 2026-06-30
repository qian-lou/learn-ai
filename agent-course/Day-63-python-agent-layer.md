# Day 63 · Python Agent 层：LangGraph 经 REST / MCP 调用 Java 服务

> **今日目标**：写出双栈的"大脑"——用 **LangGraph** 编排一个 agent,它把 Java 服务层的能力**包成工具**(REST 调用 + MCP 工具),让 LLM 自主决定何时调用、并把鉴权 token 与 trace 上下文一路透传给 Java。
> **时长**：~2h ｜ **前置**：Day 26~28(LangGraph)、Day 62(Java 服务已跑起来)
> **今日产出**：一个 LangGraph agent,问"我的订单里有超过 500 的吗,帮我退了",它能调 Java 查订单 → 让 LLM 决策 → 调 Java 幂等退款,全程带 bearer token。

## 1. 为什么 & 是什么(概念 + Java 类比)

Day 62 的 Java 服务是"肌肉",今天写"大脑"。Python 层的唯一职责:**把 Java 能力暴露成 LLM 能调用的工具,然后让 LLM 规划"先查再退"这种多步流程**。

关键设计点,给 Java 类比:

| Python Agent 层做的事 | Java 世界类比 | 说明 |
|---|---|---|
| 把 REST 接口包成 `@tool` | Feign Client + 注册成可调用能力 | agent 的"工具"= 对 Java 的一次 HTTP 调用 |
| LangGraph 的 state/node | 状态机 / Saga 编排 | 多步流程的状态由图持有 |
| LLM 决定调哪个工具 | (无直接对应) | agent 独有的"模糊路由" |
| 透传 bearer token | `RestTemplate` 拦截器加 Auth 头 | agent 调 Java 必须带身份 |
| 透传 trace 上下文 | Sleuth 跨服务传 traceId | 为 Day 65 端到端链路 |

**最重要的一条心智:工具(tool)就是 Python 层与 Java 层之间唯一的桥。** LLM 永远不直接碰数据库、不自己算权限——它只会"调用 `refund_order` 这个工具",而工具内部是一次对 Java `/api/v1/refunds` 的、带幂等键和 token 的 HTTP 请求。**Java 收到后会重新校验一切**(Day 62 的信任根)。这样即使 LLM 被注入诱导"退 100 万",Java 也会因超上限而拒绝——**LLM 只提议,Java 信任根说了算**。

## 2. 跟着做(Hands-on)

**Step 1 — 装依赖(2026 现代栈)**

```bash
pip install "langgraph>=0.2.50" "langchain-openai>=0.2" "httpx>=0.27"
# MCP 客户端,可选,用于 MCP 通道 / optional MCP client
pip install "mcp>=1.1" "langchain-mcp-adapters>=0.1"
```

**Step 2 — 把 Java REST 接口包成工具(带 token 透传)**

```python
"""Day 63: Python Agent 层 / the orchestration brain.
把 Java 服务包成工具,LLM 自主决定何时调用 / wrap Java endpoints as tools, LLM decides."""
import os
from typing import Any, Dict, List

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Java 服务层地址 + 机器身份 token / Java base URL and the agent's M2M bearer token
JAVA_BASE_URL: str = os.environ["JAVA_SERVICE_URL"]  # 如 http://localhost:8080
AGENT_TOKEN: str = os.environ["AGENT_JWT"]

# 复用单个 client,带超时,勿每次新建 / reuse one client, with timeout
_http: httpx.Client = httpx.Client(
    base_url=JAVA_BASE_URL,
    timeout=httpx.Timeout(10.0),
    headers={"Authorization": f"Bearer {AGENT_TOKEN}"},  # token 透传给 Java / pass identity
)

@tool
def list_orders(user_id: str) -> List[Dict[str, Any]]:
    """查询某用户全部订单,只读 / list a user's orders, read-only. 时间 O(1) 调用 + O(N) 解析;Raises HTTPStatusError(非 2xx)。"""
    resp = _http.get(f"/api/v1/orders/{user_id}")
    resp.raise_for_status()  # 错误显式抛出,交给 agent 处理 / surface errors to the agent
    return resp.json()["orders"]

@tool
def refund_order(order_id: str, amount: float, idempotency_key: str) -> Dict[str, Any]:
    """对订单发起退款,有副作用 / refund an order; Java re-validates amount/auth/idempotency. idempotency_key 重试用同一值;Raises HTTPStatusError(越界 400 / 额度不足 409)。"""
    # LLM 只"提议"退款,真正的安全校验在 Java(信任根)/ Java is the root of trust
    resp = _http.post(
        "/api/v1/refunds",
        json={"orderId": order_id, "amount": amount, "idempotencyKey": idempotency_key},
    )
    resp.raise_for_status()
    return resp.json()
```

**Step 3 — 用 LangGraph 装配 ReAct agent**

```python
def build_agent() -> Any:
    """装配一个能调用 Java 工具的 ReAct agent / build a runnable tool-using LangGraph app."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [list_orders, refund_order]  # Java 能力即工具集 / Java capabilities as tools
    # create_react_agent: 内置"思考→调工具→看结果→再思考"循环 / built-in ReAct loop
    system = (
        "你是订单助手。可调用工具查订单、发起退款。"
        "退款前必须先查到该订单、确认金额。每次退款用唯一 idempotency_key。"
    )
    return create_react_agent(llm, tools, prompt=system)

def main() -> None:
    """跑一个真实多步场景 / run a real multi-step scenario."""
    app = build_agent()
    # 这一句会触发:LLM 先调 list_orders → 自己筛 >500 → 再调 refund_order
    result = app.invoke(
        {"messages": [("user", "我是 u1,把我订单里金额超过 500 的都退了")]}
    )
    # 中间的工具调用 LangGraph 已自动编排 / print only the final answer
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
```

**Step 4 — (可选)走 MCP 通道:让 agent 自动发现 Java 工具**

```python
"""若 Java 侧起了 MCP server,可不手写 @tool,运行时自动发现工具。
If the Java side runs an MCP server, tools are discovered at runtime."""
from langchain_mcp_adapters.client import MultiServerMCPClient

async def build_agent_via_mcp() -> Any:
    """从 Java MCP server 自动加载工具 / auto-load tools from Java's MCP server."""
    # 指向 Java 暴露的 MCP server,工具集变化无需改 Python / tools auto-sync
    client = MultiServerMCPClient(
        {"orders": {"url": "http://localhost:8080/mcp", "transport": "streamable_http"}}
    )
    tools = await client.get_tools()  # 运行时发现,无需手写 schema / discovered at runtime
    return create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools)
```

**验证:**

```bash
export JAVA_SERVICE_URL=http://localhost:8080 AGENT_JWT="$JWT"
python day63_agent.py     # 应看到:它先查订单、再对 >500 的发起退款
# 幂等:再跑同样输入,Java 因 idempotencyKey 不重复扣款
# 防御:把输入改成"退 100 万",Java 返回 400,agent 优雅报告失败而非崩
```

> 决策原则(承 Day 61):**核心写操作(退款)走 REST**——契约严格、可控;**只读发现类能力**适合走 MCP——让 agent 灵活组合。两种通道这里都给了样板,实战常并存。无论走哪条,**安全底线都在 Java**,Python 层只管"想"。

## 3. 今日任务

1. **包工具**:把你 Day 62 的 Java 接口包成 LangGraph 工具(至少一个只读 + 一个写),token 透传到 Java。
2. **跑通多步场景**:让 agent 自主完成"先查 → LLM 决策 → 再写"的链路(如"退掉超过某金额的订单")。
3. **验证信任边界**:故意让 agent 提议越界操作(超上限退款 / 查他人订单),确认**被 Java 拦下**(400/403)且 agent 优雅讲给用户、不崩。
4. **验证幂等**:同一请求跑两次,确认 Java 侧没有重复执行写操作。

**验收标准**:多步场景端到端跑通且中间工具调用由 LLM 自主编排;越界操作被 Java 拒绝且 agent 优雅处理;幂等键生效。

## 4. 自测清单

- [ ] 我理解"工具是 Python 层与 Java 层唯一的桥",LLM 从不直接碰数据/权限。
- [ ] 我能把一个 Java REST 接口包成带 token 透传的 LangGraph 工具。
- [ ] 我能说清为什么即便 LLM 被注入,Java 信任根仍能兜住安全。
- [ ] 我知道 REST 工具与 MCP 工具的区别,以及各自何时用。
- [ ] 我的写工具传了幂等键,经得起编排层重试。

## 5. 延伸 & 关联

- 本仓库 LangChain Agents 与工具(工具机制基础):[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- 本仓库 完整 LangChain 应用(编排范式):[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
- 明天 Day 64 把 Java 服务层 + Python agent 层**端到端打通一个真实场景**,跑成完整闭环。
- 本系列总计划:[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
