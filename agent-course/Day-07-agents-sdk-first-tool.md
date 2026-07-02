# Day 7 · 上手厂商 SDK：用 Agents SDK 定义第一个 tool

> **今日目标**：从"手写三段式循环"升级到"框架托管循环"——用 OpenAI Agents SDK 的 `@function_tool` 定义工具，一行 `Runner.run` 跑通完整调用。
> **时长**：~2h ｜ **前置**：Day 6（必须先理解裸机制）
> **今日产出**：一个 `day07_agent.py`，定义一个工具 + 一个 Agent，运行后能自动完成"思考→调工具→给答复"。

## 1. 为什么 & 是什么

Day 6 你**手写**了那个循环：append 消息、解析参数、回填、再请求……能跑，但啰嗦。真实项目里这套循环（含多轮、并行、错误重试）由 **SDK 托管**。今天的目标不是学新概念，而是看清：**框架到底替你省了哪几步**。

给 Java 工程师的类比：Day 6 像你用 `RestTemplate` 手撸 HTTP；今天用 Agents SDK 像换成 **Spring 声明式风格**——你只声明"有哪些工具"，循环编排交给框架。

| 你昨天手写的 | Agents SDK 替你做的 |
|---|---|
| 写 JSON Schema | `@function_tool` 自动从函数签名 + docstring 生成 schema |
| `json.loads(args)` 解析参数 | 框架自动解析并按类型校验 |
| 维护 `messages`、append `tool` 消息 | 框架内部维护，对你透明 |
| 手动"第 2 次请求" | 框架自动多轮，直到模型给出最终答复 |

> **选型说明**：本系列默认 **OpenAI Agents SDK**（`pip install openai-agents`，2025 GA），因为它和前 6 天的 OpenAI 生态无缝衔接。等价方案是 **Claude Agent SDK**（`pip install claude-agent-sdk`），用 `@tool` 装饰器 + `ClaudeSDKClient`，理念一致，文末给对照。

**一个心智锚点**：框架再方便，底下仍是 Day 6 那个三段式循环。SDK 只是把"样板代码"收进黑盒——你已经亲手拆过黑盒，所以现在敢用。

## 2. 跟着做（Hands-on）

```bash
pip install openai-agents   # 提供 Agent / Runner / function_tool
export OPENAI_API_KEY="sk-你的key"
```

**Step 1 — 用装饰器定义工具（schema 自动生成）**

```python
"""Day 7: 用 OpenAI Agents SDK 定义第一个工具 / first tool via Agents SDK."""

import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。需要天气信息时调用。

    Args:
        city: 城市名，如 北京 / city name.

    Returns:
        一句话的天气描述 / a one-line weather string.
    """
    # 关键：函数签名(city: str) + docstring 会被 SDK 自动转成 function schema
    # the signature + docstring are auto-converted into a JSON schema for the model
    fake = {"北京": "3℃ 晴", "上海": "11℃ 多云", "广州": "22℃ 小雨"}
    return f"{city}：{fake.get(city, '20℃ 未知')}"
```

注意：**没有一行 JSON Schema**。`@function_tool` 读取你的类型注解和 docstring，自动生成 Day 6 里手写的那坨 schema。参数名、类型、描述全从签名来——这也是为什么 docstring 要写清楚（它就是模型看到的工具说明）。

**Step 2 — 定义 Agent 并交给 Runner 跑**

```python
# Agent = 系统指令(人格/规则) + 它能用的工具清单
# an Agent = instructions (persona/rules) + the tools it may use
weather_agent = Agent(
    name="天气助手",
    instructions="你是简洁的中文助手。涉及天气时调用工具，再用一句话作答。",
    model="gpt-4o-mini",
    tools=[get_weather],  # 把工具挂上去 / attach the tool
)


async def main() -> None:
    """运行一次 Agent：框架自动完成调工具与回填的整个循环。"""
    # Runner.run 托管整个 Day 6 三段式循环：决定→调用→回填→最终答复
    # Runner.run drives the whole loop you hand-wrote on Day 6
    result = await Runner.run(
        weather_agent, "北京今天适合穿短袖吗？",
    )
    # final_output 是模型综合工具结果后的最终自然语言答复
    # final_output is the model's final answer after using the tool
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

运行 `python day07_agent.py`：框架内部自动走完"模型决定调 `get_weather('北京')` → 执行 → 回填 → 模型答复"，你只拿到最后一句话。对照 Day 6 的几十行手写循环，体会差距。

> **想看内部发生了什么？** Agents SDK 自带 tracing，`result` 里能拿到每一步（调了哪个工具、传了什么参数、返回了什么）。这正是 Day 6 你手动 `print` 的东西，现在框架结构化记录——Day 9 的 ReAct、后期的可观测性都靠它。

## 3. 今日任务

1. 跑通 `day07_agent.py`，确认输出是一句基于天气结果的中文答复。
2. **加输入对比**：把问题换成"你好呀"（无需工具），确认 Agent 不调工具直接寒暄——验证"模型自主决定"在框架里依然成立。
3. **改 docstring 做实验**：把 `get_weather` 的 docstring 删成一句空话，重跑几次刁钻问法，观察工具触发率/参数准确率下降——亲证"docstring 就是模型看到的工具契约"。
4. **看 trace**：打印 `result` 或 `result.new_items`，找到"工具被调用"那一步，对照 Day 6 的三段式。

**验收标准**：Agent 能在天气问题上自动调工具并给中文答复；闲聊不触发工具；你能从 `result` 里指认出"工具调用"发生在哪一步；并能说清 `@function_tool` 替代了 Day 6 的哪些手写代码。

## 4. 自测清单

- [ ] 我能说出 Agents SDK 相比 Day 6 手写循环省了哪 4 件事。
- [ ] 我会用 `@function_tool` + 类型注解 + docstring 定义工具（无需手写 schema）。
- [ ] 我理解 `Agent(instructions=..., tools=[...])` 和 `Runner.run` 的分工。
- [ ] 我知道 docstring/类型注解直接决定模型对工具的理解。
- [ ] 我能从 `result` 里找到工具调用的中间步骤。

## 5. 延伸 & 关联

- **Claude Agent SDK 对照**：`from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient`。接线比 Agents SDK 多一步：用 `@tool("name", "desc", {...})` 定义工具 → `create_sdk_mcp_server(name="...", tools=[...])` 把它们打包成进程内 MCP server → 经 `ClaudeAgentOptions(mcp_servers={...}, allowed_tools=[...])` 挂给 `ClaudeSDKClient`，再 `async with ClaudeSDKClient(options=...) as client: await client.query(...)`。装饰器生成 schema、框架托管循环——和今天一样的思路，只是工具要先经 MCP server 接入、API 命名不同。
- 框架虽方便，但**别跳过 Day 6**：线上排障时，你需要知道黑盒里那两次请求长什么样。
- 关联章节：
  - 手写版三段式（黑盒拆解）：[../agent-course/Day-06-tool-calling-basics.md](./Day-06-tool-calling-basics.md)
  - LangChain 的 Agent/Tool 封装（另一种框架口味）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - LangChain 基础（链与组件）：[../07-llm-applications/05-langchain/01-langchain-basics.md](../07-llm-applications/05-langchain/01-langchain-basics.md)
