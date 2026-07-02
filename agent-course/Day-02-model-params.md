# Day 2 · 模型参数 + 流式输出

> **今日目标**：搞懂 temperature / top_p / max_completion_tokens / stop 四个旋钮，并实现"逐 token 打印"的流式输出。
> **时长**：~2h ｜ **前置**：Day 1
> **今日产出**：一个能流式打字机式输出的脚本，外加一份"同 prompt × 不同 temperature"的对比观察。

## 1. 为什么 & 是什么

Day 1 我们用的是**默认参数**。今天学会调旋钮，就能控制模型的"性格"——是稳定严谨，还是天马行空。

类比：这些参数就像 Java 里 client 的 **`RequestConfig`**（连接超时、重试次数那一套）——同一个接口，配置不同，行为不同。

四个最常用的旋钮：

| 参数 | 作用 | 取值直觉 |
|---|---|---|
| `temperature` | 随机性/创造力，越高越"放飞" | `0`≈确定性（适合抽取、分类、代码）；`0.7` 通用对话；`1.0+` 头脑风暴/创意 |
| `top_p` | 核采样，只从累计概率前 p 的词里挑 | 与 temperature **二选一调**，别同时大改；`0.9` 是常见值 |
| `max_completion_tokens` | **输出**长度上限（不含输入） | 防止跑飞 + 控成本；要完整结果就给足，否则会被硬截断。旧参数 `max_tokens` 已被它取代（推理模型只认新参数） |
| `stop` | 命中即停的字符串（最多几个） | 让模型在某标记处收口，如 `["\n\n", "###"]`；常用于自定义格式 |

两个易错点：

- **`temperature` 和 `top_p` 不要同时大幅调**——两者都在改采样分布，叠加后行为难预测。日常**只动 temperature** 就够。
- **`max_completion_tokens` 截断 ≠ 模型答完了**。如果回答在句子中间断掉，多半是它给小了（看 `finish_reason` 是否为 `"length"`）。注：Chat Completions 里旧的 `max_tokens` 已被 `max_completion_tokens` 取代，Day 3 引入的 o 系列推理模型**只认新参数**，新代码一律用它。

**流式（streaming）是什么**：默认调用要等模型把整段生成完才一次性返回；流式则像 SSE/WebSocket 一样**一片一片（chunk）地推**，每来一小段就能立刻显示。体验上就是 ChatGPT 那种打字机效果，**首字延迟**大幅降低。对 Java 工程师：约等于把"一次性 `ResponseEntity`"换成 `Flux<String>` / SSE 流。

## 2. 跟着做（Hands-on）

**Step 1 — 显式传参，观察 `finish_reason`**

```python
"""Day 2: 模型参数 / model parameters."""

from openai import OpenAI

client = OpenAI()


def call_with_params(prompt: str, temperature: float, max_completion_tokens: int) -> None:
    """用显式参数调用，并打印结束原因 / call with explicit params, show finish_reason.

    Args:
        prompt: 用户提示 / user prompt.
        temperature: 随机性 0.0~2.0 / randomness.
        max_completion_tokens: 输出上限 / output cap.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,                       # 随机性 / randomness
        max_completion_tokens=max_completion_tokens,   # 输出长度上限（旧 max_tokens 的替代）
        stop=["###"],                                  # 命中即停 / stop sequence
    )
    choice = response.choices[0]
    # finish_reason: "stop"=正常收口, "length"=被 max_tokens 截断
    print(f"[T={temperature}] finish={choice.finish_reason}")
    print(choice.message.content, "\n")
```

**Step 2 — 流式输出（核心）**

```python
def stream_call(prompt: str, temperature: float = 0.7) -> str:
    """流式逐 token 打印，并返回拼接后的完整文本。

    Args:
        prompt: 用户提示 / user prompt.
        temperature: 随机性 / randomness, 默认 0.7.

    Returns:
        完整回答文本 / the fully assembled answer.
    """
    # stream=True 时返回的是一个可迭代的 chunk 流
    # with stream=True the call returns an iterable stream of chunks
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )

    pieces: list[str] = []
    for chunk in stream:
        # 增量内容在 delta.content；流末尾几帧可能为 None，需判空
        # the delta carries the increment; trailing frames may be None
        delta: str | None = chunk.choices[0].delta.content
        if delta is not None:
            print(delta, end="", flush=True)  # 实时刷新到终端 / flush immediately
            pieces.append(delta)
    print()  # 收尾换行 / final newline
    return "".join(pieces)
```

**Step 3 — 同 prompt × 不同 temperature 对比**

```python
if __name__ == "__main__":
    topic = "给一家做手冲咖啡的小店起 3 个中文名字。"

    print("=== 流式输出演示 / streaming ===")
    stream_call(topic)

    print("\n=== temperature 对比 / comparison ===")
    for t in (0.0, 0.7, 1.3):
        # 同一 prompt、仅改 temperature，观察多样性差异
        # same prompt, vary only temperature, observe diversity
        call_with_params(topic, temperature=t, max_completion_tokens=120)
```

运行后你会看到：`T=0.0` 多次运行结果几乎一样（确定性强），`T=1.3` 则每次都更花哨、更发散。

## 3. 今日任务

1. 跑通 `stream_call`，确认是**逐字冒出来**而不是一次性蹦出整段。
2. 把 `max_completion_tokens` 故意设成 `10` 调用一次较长的问题，观察 `finish_reason` 变成 `"length"`，亲眼确认"被截断"。
3. 用 `0.0 / 0.7 / 1.3` 三档各跑 2~3 次同一个创意类 prompt，记录"确定性 vs 多样性"的差异。
4.（可选）给一个需要固定格式的任务加上 `stop`，让模型在你指定的标记处停下。

**验收标准**：终端出现打字机式流式输出；能复现 `finish_reason="length"` 的截断；能用一句话总结"什么任务该用低 temperature、什么任务该用高 temperature"。

## 4. 自测清单

- [ ] 我能说出 temperature / top_p / max_completion_tokens / stop 各自管什么（并知道旧 `max_tokens` 已被前者取代）。
- [ ] 我知道为什么不建议同时大改 temperature 和 top_p。
- [ ] 我能通过 `finish_reason` 判断回答是"答完了"还是"被截断了"。
- [ ] 我实现了流式输出，并理解它如何降低首字延迟。
- [ ] 我能从对比实验里直观说出 temperature 对多样性的影响。

## 5. 延伸 & 关联

- 流式在做 CLI / Web 聊天时几乎是标配（Day 5 的小项目会用上）。生产里还要考虑：流式如何与"结构化输出"共存、如何在中途取消。
- Anthropic 等价写法：`with client.messages.stream(...) as stream: for text in stream.text_stream: ...`。
- 进阶提示技巧（含对输出格式的控制）：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
