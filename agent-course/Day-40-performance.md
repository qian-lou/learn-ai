# Day 40 · 性能：并行工具调用 + 流式中间结果

> **今日目标**：把串行的多工具调用改成并行（缩短墙钟时间），并用流式把中间进度实时推给用户（缩短首字延迟感知）。
> **时长**：~2h ｜ **前置**：Day 39（工具编排）、Day 2（流式输出）
> **今日产出**：一个 `day40_parallel.py`，对模型一次返回的多个 tool call 用 `asyncio.gather` 并行执行，并流式打印每个工具完成的中间结果。

## 1. 为什么 & 是什么

Agent 慢，主要慢在两处：**串行等待工具**（3 个各 1s 的工具串起来要 3s）和**用户盯着空白屏**（最终答案出来前毫无反馈）。两把利器对应解决：

| 优化 | 解决 | 指标 | Java 类比 |
|---|---|---|---|
| **并行工具调用** | 多个互不依赖的工具串行等待 | 降低**总墙钟时间** | `CompletableFuture.allOf(...)` / 线程池并发 |
| **流式中间结果** | 用户等最终答案时无反馈 | 降低**感知延迟（TTFB）** | SSE / WebSocket 推送进度 |

两个关键认知：

1. **现代模型一次就能返回多个 tool call**（parallel tool calling）。OpenAI/Anthropic 的响应里 `tool_calls` 本就是个**数组**——模型已经告诉你"这几件事彼此独立，可以一起做"。你只要别傻乎乎地 `for` 循环 `await` 它们，而是 `gather` 并发执行。**注意：这是 I/O 并发，用 `asyncio` 即可，不需要多线程**（工具大多是网络调用，受 GIL 影响小）。
2. **并行的前提是"无依赖"**。如果工具 B 的入参依赖工具 A 的输出，就必须串行——这本质是个 DAG 拓扑，独立节点并行、有依赖的串行。别为了并行而并行。

> 复杂度视角：N 个独立工具，串行是 `O(Σtᵢ)`（时间累加），并行是 `O(max tᵢ)`（取最慢的那个）。这就是为什么并行对"几个慢工具"收益巨大。

## 2. 跟着做（Hands-on）

用异步 OpenAI 客户端。演示：模型一次返回 3 个独立工具调用 → `asyncio.gather` 并发跑 → 每个完成就流式打印。

```bash
pip install "openai>=1.40"
```

```python
"""Day 40: 并行工具调用 + 流式中间结果 / parallel tool calls with streamed progress."""

from __future__ import annotations

import asyncio
import json
import time

from openai import AsyncOpenAI

client = AsyncOpenAI()
MODEL = "gpt-4o-mini"


# ---- 三个互相独立、各自有耗时的"慢工具" / three independent slow tools ----
async def get_weather(city: str) -> str:
    await asyncio.sleep(1.0)  # 模拟网络 I/O / simulate network I/O
    return f"{city} 晴 26℃"


async def get_stock(symbol: str) -> str:
    await asyncio.sleep(1.0)
    return f"{symbol} 收 ¥152.3"


async def get_news(topic: str) -> str:
    await asyncio.sleep(1.0)
    return f"{topic} 今日要闻：A2A 进入 CNCF 沙箱"


DISPATCH = {"get_weather": get_weather, "get_stock": get_stock, "get_news": get_news}

TOOLS = [
    {"type": "function", "function": {"name": "get_weather",
        "description": "查城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_stock",
        "description": "查股票收盘价",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "get_news",
        "description": "查主题新闻",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
]


async def run_one(call) -> dict:
    """执行单个 tool call，完成即返回回填消息 / run one tool call, return its tool message."""
    args = json.loads(call.function.arguments)
    result = await DISPATCH[call.function.name](**args)
    print(f"  ✓ [{call.function.name}] 完成 -> {result}")  # 流式中间结果 / live progress
    return {"role": "tool", "tool_call_id": call.id, "content": result}


async def main() -> None:
    messages = [
        {"role": "system", "content": "用户问多件独立的事，请一次性发起所有需要的工具调用。"},
        {"role": "user", "content": "同时告诉我北京天气、AAPL 收盘价、还有『多 agent』的新闻。"},
    ]
    # 第一轮：让模型一次返回多个并行 tool call / get all parallel tool calls at once
    resp = await client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message
    messages.append(msg)
    print(f"模型一次请求了 {len(msg.tool_calls)} 个工具，开始并行执行…")

    # 关键：gather 并发，而非 for-await 串行 / concurrent, NOT sequential
    t0 = time.monotonic()
    tool_msgs = await asyncio.gather(*(run_one(c) for c in msg.tool_calls))
    print(f"并行总耗时 {time.monotonic() - t0:.2f}s（串行会是 ~{len(msg.tool_calls):.0f}s）")
    messages.extend(tool_msgs)

    # 第二轮：把结果交回模型，流式输出最终答案 / stream the final synthesized answer
    print("\n最终回答（流式）/ final answer (streaming):")
    stream = await client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
```

你会看到三个工具几乎**同时完成**，总耗时约 1s 而非 3s；随后最终答案逐字流式吐出。

## 3. 今日任务

1. 跑通 `day40_parallel.py`，确认"并行总耗时"约等于单个工具耗时（~1s），而非工具数之和。
2. **做对照实验**：把 `asyncio.gather` 改成 `for c in msg.tool_calls: await run_one(c)`（串行），对比耗时翻几倍——亲手量出并行收益。
3. **识别"不能并行"的情形**：构造一个任务，让工具 B 必须用工具 A 的输出当入参（如先 `get_stock` 拿价再 `compute` 算市值），思考为什么这两步**不能**放进同一个 `gather`，该怎么编排。

**验收标准**：并行版耗时显著低于串行版并能报出数字；最终答案为流式输出；能正确判断哪些工具可并行、哪些因数据依赖必须串行。

## 4. 自测清单

- [ ] 我知道模型响应里的 `tool_calls` 是数组，可一次返回多个独立调用。
- [ ] 我会用 `asyncio.gather` 并发执行而非 for-await 串行。
- [ ] 我理解并行收益是 `O(Σtᵢ)` 降到 `O(max tᵢ)`，且仅对独立工具成立。
- [ ] 我知道有数据依赖的工具必须串行，并行要看 DAG 拓扑。
- [ ] 我能用流式把中间进度与最终答案实时推给用户，降低感知延迟。
- [ ] 我清楚工具多为 I/O，`asyncio` 足够，不必上多线程。

## 5. 延伸 & 关联

- 流式输出基础：本课程 Day 2（模型参数与流式输出）。
- 本课程 Day 39（工具编排）：今天把它里面串行的 worker 调用并行化。
- 生产化的延迟/成本观测（怎么证明你的优化真有效）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 推理加速（模型侧而非编排侧的性能）：[../08-llm-engineering/01-model-optimization/02-inference-acceleration.md](../08-llm-engineering/01-model-optimization/02-inference-acceleration.md)
- 阶段项目预告 Day 41–45：研究 Agent 的"并行检索多个来源"将直接用上今天的并行模式。
