# Day 5 · 复盘 + 小项目：流式对话 CLI

> **今日目标**：把前 4 天合成一个可运行的「流式对话 CLI」——多轮记忆 + 流式输出 + 识别 `/json` 结构化指令。
> **时长**：~2h ｜ **前置**：Day 1~4
> **今日产出**：单文件 `chat_cli.py`，能多轮对话、流式打字机输出，输入 `/json 描述` 时切换为 Pydantic 结构化输出。

## 1. 为什么 & 是什么

前 4 天的零件，今天组装成一台能用的"机器"。这一步的价值：**多轮对话的状态到底存在哪、怎么传给模型**——这正是后面所有 Agent 的地基（Agent 的"记忆"就是从这里长出来的）。

复盘四天，对应到本项目的零件：

| 来自 | 零件 | 在 CLI 里的角色 |
|---|---|---|
| Day 1 | `messages` 数组 + `usage` | 整个对话历史就是这个数组 |
| Day 2 | `stream=True` | 回答逐 token 冒出，体验好 |
| Day 3 | system 角色定调 | 开场设定助手人格 |
| Day 4 | `parse` + Pydantic | `/json` 指令时切到强类型输出 |

**关键心智：模型是无状态的（stateless）。** 它不会"记得"上一句——是**你每次都把完整历史 `messages` 重发过去**，它才显得有记忆。给 Java 工程师：模型像个**纯函数 / 无状态的 RESTful 接口**，会话状态由**调用方维护**。所谓"对话记忆"= 客户端持有的一个不断 append 的 list。

## 2. 跟着做（Hands-on）

```python
"""Day 5 小项目：流式对话 CLI / a streaming chat CLI.

支持 / supports:
  - 多轮记忆 / multi-turn memory (history kept client-side)
  - 流式输出 / streaming output
  - /json <描述> 切换结构化输出 / structured output via Pydantic
  - /reset 清空历史, /exit 退出 / reset & exit
"""

from typing import Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()
MODEL = "gpt-4o-mini"


class Task(BaseModel):
    """从一句话描述里抽出的待办 / a task parsed from free text."""

    title: str = Field(description="任务标题 / short title")
    priority: int = Field(ge=1, le=5, description="优先级 1~5 / priority")
    due: str = Field(description="截止时间，自然语言即可 / due date in plain text")


def stream_reply(history: List[Dict[str, str]]) -> str:
    """把完整历史发给模型并流式打印回答，返回完整文本。

    Args:
        history: 全量对话历史（含 system）/ full message history.

    Returns:
        助手本轮完整回答 / the assistant's full reply this turn.
    """
    # 每次都重发完整 history —— 这就是“记忆”的真相
    # we resend the WHOLE history every time — that IS the "memory"
    stream = client.chat.completions.create(
        model=MODEL, messages=history, stream=True, temperature=0.7,
    )
    pieces: List[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            print(delta, end="", flush=True)  # 实时刷新 / flush immediately
            pieces.append(delta)
    print()
    return "".join(pieces)


def handle_json(description: str) -> str:
    """把一句描述解析为强类型 Task 并格式化展示 / parse into a typed Task.

    Args:
        description: /json 后面的自由文本 / free text after /json.

    Returns:
        给用户看的结果字符串 / a display string for the user.
    """
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "把用户描述解析为一个任务对象。"},
            {"role": "user", "content": description},
        ],
        response_format=Task,  # Day 4 的结构化输出 / structured output
    )
    msg = completion.choices[0].message
    if msg.refusal:  # 安全：模型可能拒答 / the model may refuse
        return f"(模型拒答 / refused) {msg.refusal}"
    t: Task = msg.parsed
    return f"[结构化] 标题={t.title} | 优先级={t.priority} | 截止={t.due}"


def main() -> None:
    """CLI 主循环：读输入 → 分发 → 维护历史 / the REPL loop."""
    # 历史以 system 开头，全程只 append 不重置（除非 /reset）
    # history starts with system; we only append (until /reset)
    history: List[Dict[str, str]] = [
        {"role": "system", "content": "你是简洁友好的中文助手。"}
    ]
    print("流式对话 CLI 已启动。命令: /json <描述>  /reset  /exit")

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C 优雅退出 / graceful exit
            print("\n再见 / bye")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("再见 / bye")
            return
        if user_input == "/reset":
            del history[1:]  # 保留 system，清空对话 / keep system, drop the rest
            print("(已清空历史 / history cleared)")
            continue
        if user_input.startswith("/json"):
            # 切换到结构化分支；注意：不污染对话 history
            # branch to structured mode; do NOT pollute the chat history
            desc = user_input[len("/json"):].strip()
            if not desc:
                print("用法 / usage: /json 明天下午前交季度报告")
                continue
            print(handle_json(desc))
            continue

        # 普通对话：append 用户消息 → 流式回答 → append 助手消息
        # normal turn: append user → stream reply → append assistant
        history.append({"role": "user", "content": user_input})
        print("助手 > ", end="")
        reply = stream_reply(history)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
```

跑起来：`python chat_cli.py`，先正常聊两句（验证它记得上文），再输入 `/json 下周一前把预算表发给财务`，看它返回 `[结构化] 标题=... | 优先级=... | 截止=...`。

> 设计要点：`/json` 分支**单独成链**、不写进对话 `history`，避免把结构化指令污染成普通上下文——这种"主对话 vs 旁路工具调用"的分离，正是 Agent 工程里反复出现的模式。

## 3. 今日任务

1. 跑通 `chat_cli.py`：验证多轮记忆（问"我刚才说了啥？"它能答上来）、流式输出、`/json`、`/reset`、`/exit` 全部生效。
2. **加一个命令**：实现 `/tokens`，打印当前 `history` 大致占用（可用 `len(str(history))` 粗估，或接入 Day 1 的 `usage` 累加）——为 Day 11 的"上下文超长截断"埋下直觉。
3. **挑一个增强做掉**（任选）：给 `/json` 支持多任务返回 `List[Task]`；或给普通对话加 `/system <新人格>` 动态改 system。
4. **写定义**：在文件顶部注释或单独一行，写下**你自己对 "Agent" 的一句话定义**（下面有引子）。

**验收标准**：CLI 五类命令全部可用；`/json` 返回的是结构化结果且未污染对话历史；完成至少一个增强；并写下你的 Agent 一句话定义。

## 4. 自测清单

- [ ] 我能解释"模型无状态、记忆靠客户端重发 history"这件事。
- [ ] 我的 CLI 能跨轮引用上文（真正的多轮）。
- [ ] 流式输出可用，且 `/json` 走的是结构化分支。
- [ ] 我理解为什么 `/json` 不该写进对话 `history`。
- [ ] 我用一句话定义了 "Agent"，并能说清它和"裸调用 LLM"的区别。

## 5. 延伸 & 关联

**✍️ 留给你的思考**：跑完这个 CLI，用一句话写下你心里的 "Agent" 是什么。一个参考方向（不必照抄）：

> "Agent = 一个能**自己决定下一步做什么**的 LLM 循环——它不只回答，还会调用工具、读取结果、再决策，直到任务完成。"

明天起的 Phase 1 会把这句话变成代码：今天的 `/json` 旁路调用，下周就长成真正的 **tool calling**；今天手维护的 `history`，就是 Agent **记忆**的雏形。

- 衔接下一步——LangChain 里的 Agent 与工具：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- 本系列总计划（看看你在 70 天里的位置）：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
