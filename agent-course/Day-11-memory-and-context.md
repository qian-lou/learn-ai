# Day 11 · 记忆与会话状态：上下文超长时怎么办

> **今日目标**：搞懂 Agent 的"记忆"到底存在哪、为什么会超长，并实现 2~3 种**上下文截断/压缩**策略。
> **时长**：~2h ｜ **前置**：Day 5（手维护 history）、Day 9（多步累积消息）
> **今日产出**：一个 `day11_memory.py`，一个会管理上下文窗口的对话器——超长时自动按策略裁剪，且不丢系统设定。

## 1. 为什么 & 是什么

回顾 Day 5 的铁律：**模型无状态，"记忆"=你每次重发的 `messages` 列表**。问题来了——Agent 跑久了（多轮对话 + 多步工具调用），这个列表会**无限膨胀**，迟早撑爆 context window（也越来越贵、越来越慢）。所以"记忆管理"本质是**一个定长缓冲区的淘汰策略问题**。给 Java 工程师：这几乎就是**缓存淘汰**——context window 像固定容量的 LRU 缓存，满了必须淘汰旧条目；且要按 **token**（用字节算太糙）来计量。

**两类记忆**先分清：**短期记忆**=当前会话的 `messages`（今天的主角，存内存里）；**长期记忆**=跨会话持久化的知识，通常落到**向量库/数据库**（Phase 2 的 RAG 就是它的一种实现）。

今天只攻**短期记忆的截断**，三种主流策略（按复杂度递增）：① **滑动窗口**：只留最近 K 轮，老的直接丢——简单零成本，但会失忆；② **摘要压缩**：把超出的旧消息用一次 LLM 调用总结成一条摘要挂在前面——省 token 又保要点，但多花一次调用；③ **混合**：最近 K 轮原文 + 更早内容的滚动摘要，生产常用。注意：**system 永不淘汰**，否则 Agent 人格崩塌。

## 2. 跟着做（Hands-on）

**Step 1 — 按 token 计量（而非数条数）**（依赖 `pip install "openai>=1.0" tiktoken`）

```python
"""Day 11: 上下文窗口管理 / context-window management with truncation."""

import tiktoken
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"
_ENC = tiktoken.get_encoding("o200k_base")  # gpt-4o 系列的编码 / encoding for 4o


def count_tokens(messages: list[dict[str, str]]) -> int:
    """粗略统计一组消息的 token 数（够做截断判断）。

    Args:
        messages: 消息列表 / the message list.

    Returns:
        近似 token 总数 / approximate total tokens.
    """
    # 时间 O(总字符数) 空间 O(1) —— 每条内容编码后累加，含少量角色开销
    # encode each content; add a small per-message overhead for roles
    total = 0
    for m in messages:
        total += len(_ENC.encode(m.get("content") or "")) + 4
    return total
```

**Step 2 — 三种截断策略**

```python
def truncate_sliding(messages: list[dict[str, str]], max_tokens: int,
                     ) -> list[dict[str, str]]:
    """滑动窗口：钉住 system，从最旧的非 system 消息开始丢，直到不超限。

    Args:
        messages: 含 system 的完整历史 / full history (system first).
        max_tokens: token 预算 / token budget.

    Returns:
        裁剪后的消息列表 / a trimmed list within budget.
    """
    # system 永不淘汰（否则人格丢失）；其余按"最旧先淘汰" / pin system, drop oldest
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    # 时间 O(N)：从队首逐条丢弃直到预算内 / pop oldest until within budget
    while rest and count_tokens(system + rest) > max_tokens:
        rest.pop(0)
    return system + rest


def summarize_old(messages: list[dict[str, str]], keep_recent: int,
                  ) -> list[dict[str, str]]:
    """摘要压缩：保留最近 keep_recent 轮原文，更早的用一次 LLM 调用压成摘要。

    Args:
        messages: 含 system 的完整历史 / full history.
        keep_recent: 保留原文的最近消息条数 / how many recent msgs to keep verbatim.

    Returns:
        [system, 摘要, ...最近原文] 的新历史 / compacted history.
    """
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) <= keep_recent:
        return messages  # 还没多到需要压缩 / nothing to compact yet

    old, recent = rest[:-keep_recent], rest[-keep_recent:]
    # 用一次便宜调用把旧对话压成要点 / one cheap call to summarize the old turns
    digest = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "把下面对话压成要点，保留关键事实与决定。"},
                  {"role": "user", "content": str(old)}],
    ).choices[0].message.content
    summary_msg = {"role": "system", "content": f"[早期对话摘要] {digest}"}
    return system + [summary_msg] + recent
```

**Step 3 — 装进对话器，发请求前自动裁剪**

```python
def chat(history: list[dict[str, str]], user_msg: str,
         budget: int = 800, strategy: str = "sliding") -> str:
    """一轮对话：先按策略把历史压进预算，再发请求。

    Args:
        history: 会持续 append 的历史（原地修改）/ mutated in place.
        user_msg: 本轮用户输入 / this turn's input.
        budget: token 预算（演示用故意调小）/ token budget (small for demo).
        strategy: "sliding" 或 "summary" / truncation strategy.

    Returns:
        助手回答 / the assistant reply.
    """
    history.append({"role": "user", "content": user_msg})
    # 关键：发请求前先裁剪，保证永不超 context window
    # trim BEFORE sending so we never blow the context window
    sent = (truncate_sliding(history, budget) if strategy == "sliding"
            else summarize_old(history, keep_recent=4))
    print(f"  [发送 {count_tokens(sent)} tokens / 历史共 {count_tokens(history)}]")
    reply = client.chat.completions.create(model=MODEL, messages=sent
                                           ).choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    hist = [{"role": "system", "content": "你叫小通，只说中文，回答简短。"}]
    # 故意多聊几轮把历史撑大，观察发送的 token 被压在 budget 内
    for q in ["我叫张三，记住我。", "1+1?", "讲个冷笑话", "再讲一个", "我叫什么名字？"]:
        print("Q:", q, "→ A:", chat(hist, q, budget=300, strategy="sliding"))
```

跑 `sliding` 策略：注意最后一句"我叫什么名字？"——如果窗口太小把"我叫张三"挤掉了，模型会**答不上来**（这就是滑动窗口的代价，直观感受"失忆")。把 `strategy` 换成 `"summary"` 再跑，摘要里若保住了"用户叫张三"，模型就还能答对——体会两种策略的取舍。

## 3. 今日任务

1. 跑通 `sliding` 策略，故意把 `budget` 调到很小，复现"模型忘记早期信息"的现象。
2. **对比策略**：同样的对话改用 `"summary"`，确认摘要保住关键事实（名字），模型仍能答对——量化两者差异。
3. **接回工具循环**：把 Day 9/10 的多步循环里的 `messages`，在每次请求前过一遍 `truncate_sliding`——验证工具型 Agent 也不会因长链路爆窗口。
4. **测 token 计量**：打印每轮"发送 token vs 历史总 token"，确认发送量始终被压在预算内。

**验收标准**：能复现滑动窗口的失忆现象；摘要策略下关键事实得以保留；工具循环接入截断后长对话不爆窗口；你能讲清两种策略各自的成本与代价。

## 4. 自测清单

- [ ] 我能解释"短期记忆=客户端的 messages 列表"，以及它为何会膨胀。
- [ ] 我会按 **token**（用 tiktoken）而非条数来做截断判断。
- [ ] 我实现了滑动窗口，并保证 **system 永不被淘汰**。
- [ ] 我理解摘要压缩用一次额外 LLM 调用换取"省 token + 保要点"。
- [ ] 我能区分短期记忆（messages）与长期记忆（向量库/DB，Phase 2）。

## 5. 延伸 & 关联

- **长期记忆预告**：今天的"记不住跨会话信息"正是 RAG 要解决的——把事实存进向量库，需要时检索回来塞进上下文。Phase 2（Day 16 起）专门讲。
- 生产常用**混合策略** + 给摘要也设上限（摘要本身也会越滚越长）。框架层面，Agents SDK / LangGraph 都有内置的会话状态与裁剪机制。
- 关联章节：
  - 第一次手维护 history（记忆的雏形）：[../agent-course/Day-05-chat-cli.md](./Day-05-chat-cli.md)
  - RAG 基础（长期记忆的实现）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
  - 向量数据库（长期记忆的存储）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
