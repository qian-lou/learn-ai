# Day 9 · ReAct 循环：思考 → 行动 → 观察 → 再思考

> **今日目标**：把"两次请求"扩展成一个**自驱动的多步循环**——让 Agent 连续调多个工具，自己决定何时收尾。
> **时长**：~2h ｜ **前置**：Day 6~8
> **今日产出**：一个 `day09_react.py`，手写一个带步数上限的 ReAct 主循环，能用多步工具调用完成一个复合任务。

## 1. 为什么 & 是什么

前几天的循环是"一来一回"：模型调一次工具就给最终答复。但真实任务常需**多步**：先查 A、根据 A 的结果再查 B、综合后才能答。这种"想一步、做一步、看结果、再想下一步"的范式就叫 **ReAct = Reasoning + Acting**。

核心是一个会**反复转圈**的循环：

```
Thought（想：下一步该干啥）
  → Action（做：调某个工具）
    → Observation（看：工具返回了什么）
      → Thought（再想：够了吗？还是继续？）
        → ... 直到 Thought 判断"信息够了" → Final Answer
```

给 Java 工程师的类比：这就是一个 **`while` 循环 + 状态机**。每轮迭代里，"模型"是那个根据当前状态决定"下一条边"的决策器；`messages` 历史是累积的状态；"给出最终答复"是循环的退出条件。**和你写业务状态机的唯一区别：转移函数是 LLM，不是你的 `switch`。**

为什么不能无限转？两个现实约束，必须工程化兜住：

- **死循环**：模型可能反复调同一个工具出不来 → 必须设**最大步数**（类比 Java 里给 `while` 加守卫，防活锁）。
- **成本**：每转一圈就是一次 LLM 请求，转 10 圈就是 10 次计费 → 步数上限同时是**成本闸门**。

> 注意：现代做法**不靠**让模型输出 "Thought:/Action:" 文本再正则解析（那是 2022 年的老 ReAct）。今天用**原生 tool calling** 实现同样的循环——`tool_calls` 就是 Action，`role:"tool"` 回填就是 Observation，模型的内部推理就是 Thought。更稳、更省 token。

## 2. 跟着做（Hands-on）

```bash
pip install "openai>=1.0"
```

复用 Day 8 的工具（这里精简为两个，便于看清多步链路）：

```python
"""Day 9: 手写 ReAct 主循环 / a hand-rolled ReAct loop via native tool calling."""

import json
from typing import Any

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

# 两个工具：先查城市人口，再按人均 GDP 算总量（必须分两步）
# two tools that force a multi-step chain
_POP = {"北京": 2184, "上海": 2487}        # 万人 / in 10k people
_GDP_PER = {"北京": 20.0, "上海": 18.8}      # 万元/人 / per-capita, 10k CNY


def get_population(city: str) -> dict[str, Any]:
    """查城市人口（万人）/ city population in 10k."""
    return {"city": city, "population_wan": _POP.get(city, 0)}


def get_gdp_per_capita(city: str) -> dict[str, Any]:
    """查城市人均 GDP（万元/人）/ per-capita GDP."""
    return {"city": city, "gdp_per_capita_wan": _GDP_PER.get(city, 0)}


TOOLS = [
    {"type": "function", "function": {"name": "get_population",
     "description": "查询城市常住人口（单位：万人）。",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}}, "required": ["city"],
         "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_gdp_per_capita",
     "description": "查询城市人均 GDP（单位：万元/人）。",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}}, "required": ["city"],
         "additionalProperties": False}}},
]
IMPL = {"get_population": get_population, "get_gdp_per_capita": get_gdp_per_capita}
```

**ReAct 主循环本体**（重点在 `for step in range(MAX_STEPS)`）：

```python
MAX_STEPS = 6  # 死循环/成本守卫：最多转 6 圈 / loop guard & cost cap


def react(question: str) -> str:
    """ReAct 主循环：反复 思考→行动→观察 直到模型给出最终答复。

    Args:
        question: 复合任务问题 / a multi-step question.

    Returns:
        最终自然语言答复；超步数则返回降级提示 / final answer or a fallback.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你会拆解任务，按需多次调用工具，信息足够后再作答。"},
        {"role": "user", "content": question},
    ]

    for step in range(MAX_STEPS):  # 守卫：绝不无限循环 / never loop forever
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 退出条件：模型不再要调工具 = 它认为信息够了 = Final Answer
        # exit when the model stops requesting tools (it's done)
        if not msg.tool_calls:
            print(f"  [第 {step+1} 步] 模型收尾 / final answer")
            return msg.content

        messages.append(msg)  # 把这轮的"行动意图"记进状态 / record the action
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  [第 {step+1} 步] Action: {name}({args})")  # 看每步行动
            obs = IMPL[name](**args)                              # Observation
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(obs, ensure_ascii=False)})

    # 兜底：转满 MAX_STEPS 还没收尾 → 优雅降级，别让用户等 / graceful fallback
    return "（已达最大步数仍未完成，请把问题拆得更具体 / step limit reached）"


if __name__ == "__main__":
    # 这个问题逼出多步：要分别查两城人口+人均GDP，再比较——至少 4 次工具调用
    print(react("北京和上海，哪个城市的 GDP 总量更高？给出推理过程。"))
```

跑起来看打印的每一步 `Action`：模型会**自己规划**——先 `get_population("北京")`、再 `get_gdp_per_capita("北京")`、再查上海两项，最后一步不再调工具、直接综合比较给出答案。整个多步链路**没有一行是你写死的**，全是模型在循环里逐步决策。

## 3. 今日任务

1. 跑通 `react`，数一数它为这道题转了几圈、每圈调了什么——确认是真正的多步链。
2. **逼出更长的链**：把问题改成"北京、上海、广州三城里 GDP 总量最高的是哪个"（广州数据缺失时它会怎么办？观察降级行为）。
3. **触发守卫**：把 `MAX_STEPS` 改成 `1`，跑那个复合问题，确认循环被步数上限切断并返回降级提示，而不是死循环。
4. **观察 Thought**：把 system 改成"每次调工具前，先用一句话说明你为什么调"，让模型把推理显式说出来——直观感受 Reasoning 部分。

**验收标准**：能看到 ≥3 步的工具调用链且最终给出比较结论；调小 `MAX_STEPS` 时循环被安全切断；你能在代码里指出 Thought / Action / Observation / 退出条件分别对应哪几行。

## 4. 自测清单

- [ ] 我能用"while 循环 + 状态机，转移函数是 LLM"解释 ReAct。
- [ ] 我知道 `tool_calls`=Action、`role:"tool"`回填=Observation、模型不再调工具=退出。
- [ ] 我的循环有 `MAX_STEPS` 守卫，能防死循环、控成本。
- [ ] 我理解现代 ReAct 用原生 tool calling，而非解析 "Thought:" 文本。
- [ ] 我见过一个需要 ≥3 步工具调用才能完成的任务。

## 5. 延伸 & 关联

- 今天手写的循环，Day 7 的 Agents SDK 其实**已内置**（`Runner.run` 默认就跑这个多步循环，`max_turns` 参数就是你的 `MAX_STEPS`）。手写一遍是为了让你看清框架的黑盒。
- 再往后（Phase 3）会用 **LangGraph** 把这个循环画成显式状态图（node/edge），那时"状态机"的类比会变得字面成真。
- 关联章节：
  - 多工具基础（本循环的工具来源）：[../agent-course/Day-08-multi-tool-agent.md](./Day-08-multi-tool-agent.md)
  - LangChain 里的 ReAct Agent（`create_react_agent`）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - 完整应用（多步编排落地）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
