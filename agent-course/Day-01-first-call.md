# Day 1 · 环境搭建 + 第一次模型调用

> **今日目标**：装好 SDK，拿到 API key，跑通人生第一个 LLM `chat` 调用。
> **时长**：~2h ｜ **前置**：无
> **今日产出**：一个 `day01_hello.py`，运行后能打印出模型回答 + 本次调用消耗的 token 数。

## 1. 为什么 & 是什么

学 Agent 的第一步不是"框架"，而是先把一次**裸调用**摸清楚。Agent 再花哨，底层都是一次次对模型的 HTTP 请求。

几个核心概念，给 Java 工程师的贴切类比：

| LLM 世界 | Java 世界类比 | 说明 |
|---|---|---|
| API key | 配置中心里的凭证（如 Nacos 的 access-key） | 身份 + 计费凭证，**绝不写进代码**，从环境变量读 |
| SDK（`openai` 包） | 封装好的 HTTP client（类似 `RestTemplate` / Feign） | 帮你处理鉴权、重试、序列化，省得手撸 `requests` |
| `messages` 数组 | 请求 DTO | 一个有序列表，每项 `{"role": ..., "content": ...}`，就是你发给服务端的请求体 |
| `response` 对象 | 响应 DTO（反序列化后的 POJO） | 强类型对象，字段点出来即可，不用手解析 JSON |

三个必须建立的"心智模型"：

- **Token**：模型不是按"字"而是按 **token**（子词片段）计费和计数的。英文里约 1 token ≈ 4 个字符 ≈ 0.75 个单词；中文一个汉字通常 1~2 token。**输入 + 输出 token 总量直接决定你花多少钱。**
- **Context window（上下文窗口）**：模型单次能"看到"的 token 上限（如 128k）。它像一个**定长的滑动缓冲区**——历史对话、系统提示、你的问题、模型的回答全挤在这一个窗口里。超了就得截断（Day 11 细讲）。类比：一个 `byte[128*1024]` 的固定缓冲，写满即溢出。
- **计费心智**：按 token 计价，且**输入价 ≠ 输出价**（输出通常更贵）。`gpt-4o-mini` 这类小模型极便宜，最适合学习期反复试错。养成每次调用都看一眼 `usage` 的习惯，对成本就有数了。

> 选型说明：本系列默认用 **OpenAI Python SDK（≥1.x）** + `gpt-4o-mini`，因为便宜、生态全、文档好。Anthropic 的 Claude 用 `pip install anthropic`、`client.messages.create(...)`，概念一一对应，后续 Phase 1 会用到。

## 2. 跟着做（Hands-on）

**Step 1 — 装 SDK（建议先建虚拟环境）**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install "openai>=1.0"                            # 主力 SDK / primary SDK
# 备选 Anthropic / optional: pip install anthropic
```

**Step 2 — 拿 key 并设为环境变量**

到 https://platform.openai.com/api-keys 创建一个 key（形如 `sk-...`）。**不要硬编码**，写进环境变量：

```bash
export OPENAI_API_KEY="sk-你的key"   # 加进 ~/.zshrc 持久化；Windows 用 setx
```

**Step 3 — 第一个调用**

```python
"""Day 1: 第一个 LLM chat 调用 / first LLM chat call."""

import os

from openai import OpenAI

# 客户端默认自动读取环境变量 OPENAI_API_KEY
# Client auto-reads the OPENAI_API_KEY env var by default
client = OpenAI()


def first_call(question: str) -> None:
    """发起一次最小 chat 调用并打印回答与 token 用量。

    Args:
        question: 用户问题文本 / the user's question.

    Returns:
        None. 结果直接打印到标准输出 / prints to stdout.
    """
    # messages 是有序角色列表，等价于请求 DTO
    # messages is an ordered list of roles, akin to a request DTO
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个简洁的中文助手。"},  # 系统设定 / system persona
            {"role": "user", "content": question},                      # 用户输入 / user input
        ],
    )

    # 取回答：choices[0] 是首个候选，.message.content 是文本
    # choices[0] is the first candidate; .message.content is the text
    answer: str = response.choices[0].message.content
    usage = response.usage  # token 计数对象 / token-usage object

    print("回答 / Answer:\n", answer)
    print("\n--- Token 用量 / usage ---")
    print(f"输入 / prompt     : {usage.prompt_tokens}")
    print(f"输出 / completion : {usage.completion_tokens}")
    print(f"合计 / total      : {usage.total_tokens}")


if __name__ == "__main__":
    # 健壮性：缺 key 时给出明确提示，而不是抛一长串栈
    # Robustness: fail fast with a clear message if the key is missing
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("缺少 OPENAI_API_KEY 环境变量 / OPENAI_API_KEY is not set")
    first_call("用一句话解释什么是 LLM 的 context window。")
```

运行：

```bash
python day01_hello.py
```

预期会看到一段中文回答，外加三行 token 计数（合计通常几十到一百出头）。

## 3. 今日任务

1. 跑通上面的 `day01_hello.py`，确认能打印回答 + `usage`。
2. **观察 token**：把问题从一句话改成"写一首 8 行的短诗"，再跑一次，对比 `completion_tokens` 的变化——直观感受"输出越长越贵"。
3. **试错一次鉴权**：临时 `unset OPENAI_API_KEY` 后运行，确认你的报错提示是友好的那一行，而不是裸栈。

**验收标准**：终端同时出现「中文回答」和「三行 token 计数」；改长输出后 `completion_tokens` 明显变大；缺 key 时打印的是你写的中文提示。

## 4. 自测清单

- [ ] 我能说清 token、context window、计费三者的关系。
- [ ] 我知道 API key 为什么要放环境变量，而不是写进代码。
- [ ] 我能指出代码里哪部分对应 Java 的「请求 DTO / 响应 DTO」。
- [ ] 我跑通了调用，并能从 `usage` 读出输入/输出 token。
- [ ] 我理解为什么学习期优先用 `gpt-4o-mini` 而不是大模型。

## 5. 延伸 & 关联

- 想直观看"一段文本被切成多少 token"：搜索 OpenAI 的 tokenizer 可视化页面（输入中文/英文对比，体会汉字更"吃" token）。
- Anthropic 等价写法：`client = anthropic.Anthropic()` → `client.messages.create(model="claude-...", max_tokens=512, messages=[...])`，注意它的 `max_tokens` 是**必填**。
- 本仓库已有的相关章节：
  - 提示工程基础（明天会深入）：[../07-llm-applications/02-prompt-engineering/01-prompt-basics.md](../07-llm-applications/02-prompt-engineering/01-prompt-basics.md)
  - LLM 核心技术总览：[../06-llm-core-technology/README.md](../06-llm-core-technology/README.md)
