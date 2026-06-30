# Day 6 · Tool Calling 原理：模型如何"调用"你的函数

> **今日目标**：彻底搞懂 function calling 的三段式机制——模型怎么决定调谁、function schema 长什么样、结果怎么回填。
> **时长**：~2h ｜ **前置**：Day 1~5（尤其 Day 4 结构化输出）
> **今日产出**：一个 `day06_tool_loop.py`，手写一轮完整的"模型请求调用 → 本地执行 → 回填结果 → 模型给最终答复"。

## 1. 为什么 & 是什么

模型本身**不能**执行代码、查数据库、读时间——它只会生成文本。Tool calling（旧称 function calling）就是给它一个能力：**判断需要外部能力时，输出一个"我想调用某函数 + 参数"的结构化请求**，由**你的代码**真正去执行，再把结果喂回去让它继续。关键澄清：**模型从不自己执行函数**，它只产出"调用意图"，执行权 100% 在你手里——这点对安全至关重要（Day 10、Day 15 反复强调）。

给 Java 工程师的贴切类比：

| Tool calling | Java 世界类比 | 说明 |
|---|---|---|
| function schema（JSON Schema） | 接口契约 / OpenAPI 定义 | 声明函数名、参数名、类型、必填 |
| 模型返回 `tool_calls` | 一次 **RPC 调用请求**（还没执行） | 模型说"请帮我调 `get_weather(city='北京')`" |
| 你本地执行函数 | RPC 服务端真正跑业务逻辑 | 执行权在你这边，模型碰不到 |
| `{"role":"tool", ...}` 回填 | RPC 响应回传 | 把返回值作为一条消息塞回对话 |

**三段式机制**（后面一切 Agent 的核心循环）：① **你**把 `tools=[schema...]` 和问题一起发给模型；② **模型**若需要工具，返回 `finish_reason="tool_calls"` + 一个 `tool_calls` 列表（含函数名与 JSON 参数），否则直接回答；③ **你**解析参数 → 本地执行 → 把结果以 `role:"tool"` append → **再发一次**给模型 → 模型综合结果给出最终答复。注意第 3 步要再调一次模型——一次完整工具调用至少 **两次** LLM 请求。

## 2. 跟着做（Hands-on）

```bash
pip install "openai>=1.0"
```

**Step 1 — 定义工具的 schema 与本地实现**

```python
"""Day 6: 手写一轮完整 tool calling / a full manual tool-calling round."""

import json
from typing import Any

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"


def get_weather(city: str) -> dict[str, Any]:
    """查询某城市天气（此处用假数据模拟）/ fake weather lookup.

    Args:
        city: 城市名 / city name.

    Returns:
        含温度与天气的字典 / a dict with temperature and condition.
    """
    # 真实场景这里会调外部 API；今天先用假数据聚焦机制
    # a real impl would call an API; we mock to focus on the mechanism
    fake = {"北京": (3, "晴"), "上海": (11, "多云"), "广州": (22, "小雨")}
    temp, cond = fake.get(city, (20, "未知"))
    return {"city": city, "temp_c": temp, "condition": cond}


# 工具的 function schema —— 等价于"接口契约"，模型靠它知道怎么调
# the function schema — the contract the model reads to call correctly
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气。需要天气信息时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
```

**Step 2 — 跑完整的三段式循环**

```python
# 本地函数名 → 实现的映射表，便于按名分发 / dispatch table by name
TOOL_IMPL = {"get_weather": get_weather}


def run_once(question: str) -> str:
    """执行一轮完整 tool calling 并返回最终自然语言答复。

    Args:
        question: 用户问题 / the user's question.

    Returns:
        模型综合工具结果后的最终回答 / the final answer text.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    # 第 1 次请求：把工具清单交给模型，由它决定要不要调
    # 1st call: hand the model the tools; it decides whether to call
    first = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[WEATHER_TOOL],
    )
    msg = first.choices[0].message

    # 模型不需要工具就直接回答了 / no tool needed
    if not msg.tool_calls:
        return msg.content

    # 把模型这条"调用请求"消息原样放回历史（必须，否则下一步报错）
    # append the assistant's tool-call message back (required)
    messages.append(msg)

    # 逐个执行模型请求的工具调用 / execute each requested call
    for call in msg.tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments)  # 参数是 JSON 字符串 / args are JSON
        result = TOOL_IMPL[name](**args)            # 本地真正执行 / run locally
        # 回填：role=tool，并用 tool_call_id 与请求一一对应
        # feed back as a tool message, matched by tool_call_id
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 第 2 次请求：模型拿到工具结果，给出最终自然语言答复
    # 2nd call: model sees the results and writes the final answer
    second = client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content


if __name__ == "__main__":
    print(run_once("北京今天天气怎么样？适合穿短袖吗？"))
```

运行 `python day06_tool_loop.py`，你会看到模型先"决定"调 `get_weather(city="北京")`，拿到 `{temp_c: 3, ...}` 后，用自然语言回答"3℃、晴，不适合短袖"。把问题改成"1+1 等于几"，它不会调工具，直接答——**模型自己判断该不该用工具**，这就是 tool calling 的精髓。

## 3. 今日任务

1. 跑通 `run_once`，确认能看到完整的两次请求效果（天气问题走工具，闲聊不走）。
2. **打印调用细节**：在执行工具前 `print(name, args)`，亲眼看到模型生成的函数名与参数 JSON。
3. **试探边界**：问"北京和上海哪个暖和？"，观察 `msg.tool_calls` 是否一次返回**两个**调用（并行工具调用，Day 8 详谈）。
4. **故意缺契约**：把 schema 里 `city` 的 `description` 删掉或写得很模糊，看模型抽参数的准确率是否下降——体会"schema 即 prompt"。

**验收标准**：天气类问题能走完三段式并给出含温度的中文回答；闲聊问题不触发工具；你能指出代码里哪一步是"第 1 次请求 / 执行 / 回填 / 第 2 次请求"。

## 4. 自测清单

- [ ] 我能说清"模型只产出调用意图、执行权在我代码里"这件事。
- [ ] 我会写一个 function schema（name / description / parameters / required）。
- [ ] 我知道一次完整工具调用**至少两次** LLM 请求。
- [ ] 我会用 `tool_call_id` 把 `role:"tool"` 结果和请求对应起来。
- [ ] 我理解为什么必须把模型那条 `tool_calls` 消息也 append 回历史。

## 5. 延伸 & 关联

- function schema 和 Day 4 的结构化输出**同源**——都是用 JSON Schema 约束模型输出，只不过一个约束"参数"，一个约束"返回结构"。
- Anthropic 等价写法：`client.messages.create(..., tools=[{"name":..., "input_schema":{...}}])`，模型返回 `content` 里的 `tool_use` 块，你用 `tool_result` 块回填，三段式完全一致。
- 关联章节：
  - LangChain 里的 Agent 与工具（看框架如何封装这套循环）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - 结构化输出回顾（同源能力）：[../agent-course/Day-04-structured-output.md](./Day-04-structured-output.md)
  - 提示工程进阶（schema 描述写得好 = 工具调得准）：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
