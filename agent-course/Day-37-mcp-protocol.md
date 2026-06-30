# Day 37 · MCP 协议：接一个现成 MCP server 当工具源

> **今日目标**：搞懂 Model Context Protocol 解决什么问题，用官方 `mcp` Python SDK 连一个现成的 MCP server，把它的工具喂给你的 agent。
> **时长**：~2h ｜ **前置**：Day 6（Tool Calling 原理）、Day 8（多工具 Agent）
> **今日产出**：一个 `day37_mcp_client.py`，启动一个本地文件系统 MCP server 并列出/调用它暴露的工具；以及一个 30 行的自定义 MCP server。

## 1. 为什么 & 是什么

到目前为止，你的工具都是**写死在 agent 进程里**的 Python 函数。问题来了：每接一个新数据源（GitHub、Slack、Postgres、文件系统……），你都要重写一份 function schema + 调用胶水代码，而且只能给自己用。

**MCP（Model Context Protocol）** 是 Anthropic 2024 年底开源、2025 年成为业界事实标准的协议。一句话：**它把"工具/数据源"和"agent"解耦成 client–server 两端，用统一协议（JSON-RPC over stdio/HTTP）通信。**

| MCP 概念 | 作用 | Java 工程师类比 |
|---|---|---|
| **MCP Server** | 暴露一组工具/资源/提示，独立进程 | 一个**微服务**，对外提供标准接口 |
| **MCP Client** | 你的 agent，发现并调用 server 的能力 | 服务消费方（Feign client） |
| **协议本身** | JSON-RPC，规定握手、列举、调用 | 就像 **JDBC / ODBC**——换数据库不换上层代码 |
| **Tools / Resources / Prompts** | server 三类能力 | 接口的方法 / 只读数据 / 模板 |

为什么这对你重要：**生态复用**。社区已经有几百个现成 MCP server（GitHub、文件系统、Postgres、浏览器、Sentry……）。接 MCP = 一行配置接入一个完整工具集，不用自己写。这正是"换数据库不换上层代码"的 Agent 版本。

> 传输（transport）有两种：**stdio**（server 作为子进程，本地工具首选）和 **streamable HTTP**（远程 server）。今天用 stdio，最易上手。

## 2. 跟着做（Hands-on）

用官方 `mcp` SDK。先连一个**现成的**文件系统 server（社区维护，npm 一行起），再自己写一个最小 server。

```bash
pip install "mcp>=1.2"          # 官方 Python SDK / official MCP SDK
# 现成 server 用 npx 拉起，无需预装 / ready-made server via npx
```

**Part A — 客户端连现成 MCP server（文件系统）**

```python
"""Day 37: MCP 客户端连现成文件系统 server / connect a ready-made MCP server."""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 现成的文件系统 MCP server，限定可访问目录（安全边界）
# ready-made filesystem server; the path arg is its access boundary
server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)


async def main() -> None:
    """握手 → 列举工具 → 调用一个工具 / handshake, list tools, call one."""
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # 协议握手 / protocol handshake

            tools = await session.list_tools()  # 发现能力 / discover capabilities
            print("server 暴露的工具 / exposed tools:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            # 调用其中一个工具（写一个文件）/ invoke a tool
            result = await session.call_tool(
                "write_file",
                arguments={"path": "/tmp/mcp_hello.txt", "content": "hello from MCP"},
            )
            print("调用结果 / call result:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

**Part B — 30 行自定义 MCP server（用 `FastMCP`）**

`FastMCP` 是 SDK 内置的高层封装，用装饰器把普通函数变成 MCP 工具——和你写 FastAPI 路由几乎一样。

```python
"""Day 37: 最小自定义 MCP server / a minimal custom MCP server."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools")  # server 名称 / server name


@mcp.tool()
def word_count(text: str) -> int:
    """统计文本词数 / count words in text.

    Args:
        text: 输入文本 / input text.

    Returns:
        词数 / number of whitespace-separated words.
    """
    # 时间 O(n) 空间 O(1)
    return len(text.split())


@mcp.tool()
def fx_rate(base: str, quote: str) -> str:
    """查询汇率（演示桩）/ look up an FX rate (stub)."""
    return f"1 {base} = 0.92 {quote}（示例数据 / sample）"


if __name__ == "__main__":
    mcp.run(transport="stdio")  # 作为子进程被 client 拉起 / run over stdio
```

把 Part A 里的 `StdioServerParameters` 改成 `command="python", args=["你的_server.py"]`，就能让客户端连上你自己的 server。

## 3. 今日任务

1. 跑通 Part A，打印出文件系统 server 暴露的工具列表，并成功写出 `/tmp/mcp_hello.txt`。
2. 跑通 Part B 的自定义 server，再写一个客户端连它，调用 `word_count` 验证返回值正确。
3. **接进 agent**：把 MCP 工具的 `name/description/inputSchema` 转换成你 Day 8 agent 用的 tool schema，让模型自己决定调用 `word_count`（提示：`session.list_tools()` 返回的每个 tool 自带 `inputSchema`，可直接喂给 OpenAI 的 `tools=` 参数）。

**验收标准**：能列出至少一个现成 server 的工具并成功调用；自定义 server 的 `word_count` 能被独立客户端调用；能说清"为什么 MCP 让工具可跨 agent 复用"。

## 4. 自测清单

- [ ] 我能用一句话说清 MCP 解决的问题（工具与 agent 解耦、生态复用）。
- [ ] 我能区分 MCP 的 Tools / Resources / Prompts 三类能力。
- [ ] 我理解 stdio 与 HTTP 两种 transport 的适用场景。
- [ ] 我知道 `server-filesystem` 的路径参数其实是安全边界，不能随手给 `/`。
- [ ] 我能把 MCP 工具的 `inputSchema` 对接到模型的 tool calling。

## 5. 延伸 & 关联

- 工具调用的底层原理（MCP 之下仍是 function calling）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- 本课程 Day 8（多工具 Agent）：MCP 是它的"协议化升级版"。
- 安全视角：把外部 MCP server 当工具源时，等同引入第三方代码，需做权限边界（本课程 Day 54 工具权限与 guardrails 会展开）。
- 明天 Day 38（A2A）：MCP 解决"agent↔工具"，A2A 解决"agent↔agent"，互补不冲突。
