# Day 3 · Prompt 基础：角色、few-shot、CoT

> **今日目标**：用好 system/user/assistant 三角色，掌握 few-shot 与思维链（CoT），并认清 2026 年的反模式。
> **时长**：~2h ｜ **前置**：Day 1、Day 2
> **今日产出**：一个对比脚本——同一任务，zero-shot vs few-shot vs CoT，肉眼看出质量差异。

## 1. 为什么 & 是什么

模型的输出质量，**七分靠 prompt**。Prompt engineering = 不动模型参数，只靠"怎么问"把效果拉满。给 Java 工程师：这就像**精心设计 API 的请求参数**——接口（模型）没变，但入参组织得好不好，结果天差地别。

**三种角色（messages 里的 `role`）**：

| 角色 | 作用 | Java 类比 |
|---|---|---|
| `system` | 设定身份、规则、输出风格——全程生效的"宪法" | 全局配置 / 拦截器里设的规则 |
| `user` | 用户的实际输入 | 请求参数 |
| `assistant` | 模型的回答；**也可由你手写**，用来给"示例" | 既是响应，也是 mock 数据 |

**Few-shot（少样本）**：在 `messages` 里**手写几组 user→assistant 示例**，再抛真正的问题。模型会照着示例的"格式和口吻"作答。本质是**用例子定义任务**，比纯文字描述更稳。类比：给模型几条"输入/期望输出"的单元测试样例，让它照着对齐。

**CoT（Chain-of-Thought，思维链）**：引导模型"先推理、再给结论"。对**多步推理/数学/逻辑**类任务，让它分步思考能显著提升正确率。经典触发语是 "let's think step by step"。

## ⚠️ 2026 关键认知：CoT 已分情况

**对 2026 年的推理模型（OpenAI 的 o 系列、Claude 的 extended thinking 等），再手写 "let's think step by step" 已经是反模式。**

为什么：这类模型在**训练阶段已内置了推理过程**，调用时会自动在内部"想"。你再手动塞一句"一步步想"，往往是：

- **多余**——它本来就在分步推理；
- **甚至有害**——可能干扰它原生的推理策略，或让你为重复的推理 token 多付钱。

**正确做法（2026）**：

- 用**推理模型**（o 系列 / Claude thinking）时：**只描述任务和约束，把"怎么想"交给模型**。需要时通过 `reasoning_effort` 之类的参数调推理强度，而不是在 prompt 里手写咒语。
- 用**普通模型**（如 `gpt-4o-mini`）做复杂推理时：CoT 手法**依然有效**，可以继续用。

一句话记心里：**"Let's think step by step" 是给非推理模型的拐杖；给推理模型用，多半是帮倒忙。**

## 2. 跟着做（Hands-on）

```python
"""Day 3: Prompt 三件套对比 / roles, few-shot, CoT."""

from openai import OpenAI

client = OpenAI()


def ask(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """发送一组 messages 并返回回答 / send messages, return the answer.

    Args:
        messages: 角色消息列表 / list of role messages.
        model: 模型名 / model id.

    Returns:
        模型回答文本 / the model's text answer.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 对比实验用 0，保证可复现 / 0 for reproducibility
    )
    return response.choices[0].message.content


# --- 1) 仅 system + user：定调 / set the tone ---
def demo_system() -> str:
    return ask([
        {"role": "system", "content": "你是严谨的法律顾问，只用要点回答，不寒暄。"},
        {"role": "user", "content": "签合同前最该确认的 3 件事？"},
    ])


# --- 2) few-shot：手写示例对，让模型对齐格式 ---
def demo_few_shot() -> str:
    return ask([
        {"role": "system", "content": "把用户输入解析为『情感: 标签』一行输出。"},
        # 下面三轮是“示例”，由我们手写 assistant 充当样例
        # these turns are hand-written exemplars acting as few-shot samples
        {"role": "user", "content": "这手机续航太顶了"},
        {"role": "assistant", "content": "情感: 正面"},
        {"role": "user", "content": "客服半天不回，差评"},
        {"role": "assistant", "content": "情感: 负面"},
        # 真正要解决的问题 / the real query
        {"role": "user", "content": "包装一般，但东西还行"},
    ])


# --- 3) CoT：仅对“非推理模型”有效 ---
def demo_cot() -> str:
    # gpt-4o-mini 不是推理模型，CoT 仍然有用
    # gpt-4o-mini is NOT a reasoning model, so CoT still helps
    return ask([
        {"role": "user", "content": (
            "一个水池进水管 6 小时注满，出水管 8 小时放空。"
            "两管同时开，多久注满？请一步步推理后再给最终答案。"
        )},
    ])


if __name__ == "__main__":
    print("=== system 定调 ===\n", demo_system(), "\n")
    print("=== few-shot ===\n", demo_few_shot(), "\n")
    print("=== CoT(非推理模型) ===\n", demo_cot())
```

**对比实验**：把 `demo_cot` 里的"请一步步推理后再给最终答案"删掉，跑一次直接要答案的版本，对比正确率/稳定性——你会感受到 CoT 对非推理模型的增益。

## 3. 今日任务

1. 跑通三个 demo，确认 few-shot 的输出严格贴合你给的"情感: 标签"格式。
2. **改写练习**：拿 `demo_system` 的 system prompt 连改 3 版（如"换成幽默风/限定 50 字内/必须给出反例"），观察输出风格随之变化。
3. **CoT A/B**：对 `demo_cot` 做"有 CoT vs 无 CoT"对照，记下差异。
4. **认知确认**：用一句话写下——如果今天换成 o 系列推理模型做这道水池题，你会怎么写 prompt（提示：别再手写"一步步想"）。

**验收标准**：few-shot 输出格式稳定可控；能展示一组 system prompt 改写带来的风格变化；能说清 CoT 在"推理模型 vs 非推理模型"上的不同处置。

## 4. 自测清单

- [ ] 我能说清 system / user / assistant 各自的职责。
- [ ] 我知道 few-shot 的示例是写在 `messages` 里的 user/assistant 对。
- [ ] 我能解释 CoT 为什么能提升非推理模型的多步推理质量。
- [ ] **我能讲清：为什么对 2026 推理模型手写 "let's think step by step" 是反模式。**
- [ ] 我会优先用任务描述 + 模型原生推理，而不是堆砌"思考咒语"。

## 5. 延伸 & 关联

- 进阶技巧：角色扮演、输出约束、自一致性（self-consistency）、提示模板化等。
- 本仓库现成章节，强烈建议配合阅读：
  - 提示工程基础：[../07-llm-applications/02-prompt-engineering/01-prompt-basics.md](../07-llm-applications/02-prompt-engineering/01-prompt-basics.md)
  - 进阶提示技巧：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
