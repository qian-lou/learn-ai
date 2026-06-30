# Day 51 · 成本优化：缓存、token 管理、控制上下文膨胀

> **今日目标**：把 Agent 的钱花得明白——用 **prompt 缓存**、**token 管理**、**上下文裁剪**三招，在不掉质量的前提下显著降本。
> **时长**：~2h ｜ **前置**：Day 48（看 trace 找成本黑洞）、Day 11（记忆/截断）
> **今日产出**：一个 `cost_opt.py`，演示读取缓存命中、对比"裁剪上下文前后"的 token 与成本，并给出一份省钱清单。

## 1. 为什么 & 是什么

Day 48 你已亲眼看到"上下文膨胀"怎么把账单悄悄翻倍。今天系统地解决它。成本优化不是抠门——**它是生产 Agent 能不能规模化的生死线**：一个每天百万次调用的 Agent，单次省 30% 就是省一大笔真金白银。

给 Java 工程师的对照——这套打法你全见过：

| LLM 成本优化 | Java / 后端世界 | 说明 |
|---|---|---|
| **Prompt 缓存** | Redis / 二级缓存 | 重复的长前缀（系统提示/示例）命中缓存，**不重复计费** |
| **Token 管理** | 控制 payload 大小 | 输入越短越便宜，砍掉无用上下文 |
| **上下文裁剪** | 分页 / 滑动窗口 | 只带"够用"的历史，而非全量 |
| **小模型分流** | 服务降级 / 分级处理 | 简单活给便宜模型（Day 52 专讲） |

**三招的优先级（投入产出比从高到低）：**

1. **Prompt 缓存——白捡的省钱**。2026 的好消息：OpenAI 对 `gpt-4o` 及更新模型**自动开启** prompt 缓存，重复的长前缀（≥1024 token）命中后**便宜约 90%**，你几乎不用改代码，只要**把静态内容放最前、可变内容放最后**。Anthropic 则是显式 `cache_control`。
2. **控制上下文膨胀——最大的隐形浪费**。就是 Day 48 那个坑：别把工具原始大块、全量历史无脑塞进每轮。
3. **Token 管理——精打细算**：压缩系统提示、裁剪 few-shot、对历史做摘要。

## 2. 跟着做（Hands-on）

### Step 1 — 读出缓存命中（OpenAI 自动缓存）

```bash
pip install "openai>=1.0"
```

```python
"""Day 51: 观测 prompt 缓存命中 / observe automatic prompt caching."""

from openai import OpenAI

client = OpenAI()
# 一段足够长(≥1024 token)的静态系统提示，才能命中缓存
# a long (>=1024 tok) STATIC prefix is what gets cached
LONG_SYSTEM = "你是资深客服助手。以下是公司政策与常见问答：\n" + ("政策条款示例。 " * 500)


def ask(question: str) -> None:
    """同一长前缀连问两次，第二次应命中缓存 / second call should hit cache."""
    resp = client.chat.completions.create(
        model="gpt-4o",  # 自动缓存对 gpt-4o 及更新模型生效 / auto-cache on gpt-4o+
        messages=[
            {"role": "system", "content": LONG_SYSTEM},  # 静态长前缀放最前 / static prefix FIRST
            {"role": "user", "content": question},        # 可变内容放最后 / variable part LAST
        ],
        prompt_cache_key="kb/customer-v1",  # 2026 可选：提升共享前缀的命中率 / optional hint
    )
    u = resp.usage
    # 2026 字段路径：缓存命中的 token 数 / cached tokens field path
    cached = u.prompt_tokens_details.cached_tokens
    print(f"prompt={u.prompt_tokens}  cached={cached}  (<1024 前缀时 cached=0)")


if __name__ == "__main__":
    ask("怎么退货？")   # 第一次：写入缓存 / first: populate cache
    ask("怎么换货？")   # 第二次：长前缀命中，cached>0 / second: cache hit
```

跑两次，第二次的 `cached_tokens` 会 > 0——**那部分 token 计费打了一折**。配合 Day 47 的 trace，你能在面板上直接看到缓存省了多少。

> **Anthropic 侧（显式缓存）**：在 system / message 内容块上加 `cache_control: {"type": "ephemeral"}`，用 `usage.cache_creation_input_tokens`（首次写入）与 `usage.cache_read_input_tokens`（命中读取）观测。2026 当前 Claude 模型用 `claude-sonnet-4-6` / `claude-opus-4-8`（**不要**再写 `claude-3-5-sonnet-*`）。

```python
# Anthropic 显式缓存 / explicit caching with Anthropic
resp = anthropic.Anthropic().messages.create(
    model="claude-sonnet-4-6", max_tokens=512,
    system=[{"type": "text", "text": LONG_SYSTEM,
             "cache_control": {"type": "ephemeral"}}],  # 缓存断点 / cache breakpoint
    messages=[{"role": "user", "content": "怎么退货？"}],
)
print(resp.usage.cache_creation_input_tokens, resp.usage.cache_read_input_tokens)
```

### Step 2 — 控制上下文膨胀：裁剪 + 摘要

```python
from typing import Dict, List

MAX_TURNS = 6  # 只保留最近 N 轮，更早的压成摘要 / keep last N turns, summarize older


def trim_history(history: List[Dict[str, str]], max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    """滑动窗口裁剪：保留 system + 最近若干轮 / sliding-window trim.

    Args:
        history: 完整对话历史（含 system 在 [0]）/ full history, system at [0].
        max_turns: 保留的最近消息条数 / how many recent messages to keep.

    Returns:
        裁剪后的历史 / trimmed history. 时间 O(1) 切片 空间 O(k)。
    """
    if len(history) <= max_turns + 1:
        return history
    system = history[:1]                  # 永远保留 system / always keep system
    recent = history[-max_turns:]         # 只带最近窗口 / keep the recent window
    return system + recent                # 中间的可另行摘要后再插回 / older → summarize


def estimate_cost(prompt_tok: int, completion_tok: int) -> float:
    """按 gpt-4o-mini 价目粗算单次成本（美元）/ rough cost estimate.

    价格示例：$0.15/1M 输入, $0.60/1M 输出 / illustrative pricing.
    """
    return round(prompt_tok / 1e6 * 0.15 + completion_tok / 1e6 * 0.60, 6)


if __name__ == "__main__":
    # 模拟一段 20 轮的长对话 / a fake 20-turn conversation
    fake = [{"role": "system", "content": "S"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"} for i in range(20)
    ]
    print("裁剪前条数 / before:", len(fake))
    print("裁剪后条数 / after :", len(trim_history(fake)))
    # 直观感受：每轮输入 token 从“随轮数线性增长”变成“恒定上限”
    print("成本对比示例 / cost @ 8000 vs 1500 输入 tok:",
          estimate_cost(8000, 200), "→", estimate_cost(1500, 200))
```

**核心策略总结（省钱清单）：**

- **静态内容前置 + 稳定**：系统提示、工具定义、few-shot 放最前且不变，最大化缓存命中。
- **滑动窗口 + 摘要**：长对话只带最近 N 轮，更早的压成一条摘要（Day 48 的滚动摘要法）。
- **别把工具原始大块塞回历史**：用完即弃，只留提炼结果。
- **压缩 prompt**：删冗余措辞、合并示例；用 `max_tokens` 给输出设上限。
- **小模型分流**：能用便宜模型办的，别上大模型（Day 52）。
- **观测驱动**：每条优化都回 Day 47 的 trace 看 token/成本曲线，用数字验证。

## 3. 今日任务

1. 跑 Step 1，连问两次，确认第二次 `cached_tokens > 0`——亲眼看到缓存生效。
2. 把 Day 48 的 `buggy_research` 接上 `trim_history`（或滚动摘要），回 trace 对比优化前后的总 token 与成本，**记录降幅**。
3. **压缩 prompt 实验**：把一个啰嗦的 system prompt 砍掉 40% 字数，用 Day 49 的 eval 确认**质量没掉**、但 token 降了——体会"省钱不等于降质"。
4. **算一笔账**：估算你的 Agent 在"每天 10 万次调用"下，优化前后的月成本差（用 `estimate_cost` + 真实 token 数）。

**验收标准**：能读出缓存命中数；接入裁剪后 token/成本曲线明显下降并有记录；prompt 压缩后 eval 分数不掉；能给出一份"优化前后月成本对比"的估算。

## 4. 自测清单

- [ ] 我理解 prompt 缓存的原理，知道要"静态前置、可变后置"。
- [ ] 我知道 OpenAI 自动缓存看 `usage.prompt_tokens_details.cached_tokens`。
- [ ] 我知道 Anthropic 用 `cache_control` + `cache_read_input_tokens` 观测。
- [ ] 我会用滑动窗口 / 摘要控制上下文膨胀，避免每轮线性增长。
- [ ] 我坚持"观测驱动优化"——每次降本都回 trace 用数字验证，且用 eval 守住质量。

## 5. 延伸 & 关联

- 成本优化的另一半——**按难度路由到不同大小的模型**：[Day-52-model-routing.md](./Day-52-model-routing.md)。
- 优化效果靠 Day 47 的 trace 验证，质量底线靠 Day 49~50 的 eval 守住。
- 本仓库相关章节：
  - 推理加速（KV cache / 批处理等服务端降本）：[../08-llm-engineering/01-model-optimization/02-inference-acceleration.md](../08-llm-engineering/01-model-optimization/02-inference-acceleration.md)
  - 量化（用更小的模型权重换更低的推理成本）：[../08-llm-engineering/01-model-optimization/01-quantization.md](../08-llm-engineering/01-model-optimization/01-quantization.md)
