# 高级提示技巧 / Advanced Prompt Techniques

## 1. 背景（Background）

> **为什么要学这个？**
>
> 基础 Prompt 技巧解决简单任务，但复杂推理需要**高级策略**。Chain-of-Thought（思维链）让 GPT-4 的数学能力提升 30%，ReAct 让模型能够调用工具完成真实任务。
>
> 这些技巧是构建 AI Agent 的理论基础。

## 2. 知识点（Key Concepts）

| 技巧 | 核心思想 | 适用场景 |
|------|---------|---------|
| CoT | 分步推理 | 数学、逻辑 |
| Zero-shot CoT | "让我们一步步思考" | 通用推理 |
| Self-Consistency | 多次采样取多数 | 需要高准确率 |
| ToT | 搜索推理树 | 复杂规划 |
| ReAct | 思考+行动交替 | Agent |
| Structured Output | 约束输出格式 | API 集成 |

## 3. 内容（Content）

### 3.1 Chain-of-Thought (CoT)

```
# ============================================================
# 标准 CoT（给推理示例）
# ============================================================
问题：一个商店有 15 个苹果，卖了 8 个，又进了 12 个，现在有多少？

思考过程：
1. 初始数量：15 个苹果
2. 卖出后：15 - 8 = 7 个
3. 进货后：7 + 12 = 19 个
答案：19 个

问题：小明有 23 元，买了 3 本 5 元的书，找回多少钱？

思考过程：


# ============================================================
# Zero-shot CoT（不需要示例，一句话触发）
# ============================================================
问题：如果一列火车以 60km/h 速度行驶 2.5 小时能走多远？

Let's think step by step.（让我们一步步思考。）


# ============================================================
# CoT 的效果（GPT-4 在 GSM8K 数学基准上）:
# ============================================================
# 无 CoT: ~55% 准确率
# 有 CoT: ~92% 准确率 → 提升 37%！
```

### 3.2 ReAct（Reasoning + Acting）

```
# ============================================================
# ReAct 模式：思考 → 行动 → 观察 → 重复
# ============================================================

问题：2024 年奥运会在哪里举行？获得金牌最多的国家是？

Thought 1: 我需要查询 2024 年奥运会的举办地点
Action 1: search("2024 Olympics host city")
Observation 1: 2024 年夏季奥运会在法国巴黎举行

Thought 2: 现在我需要查询金牌数最多的国家
Action 2: search("2024 Paris Olympics most gold medals")
Observation 2: 美国以 40 枚金牌位居第一

Thought 3: 我已经有了所有需要的信息
Answer: 2024 年奥运会在法国巴黎举行，美国获得最多金牌（40 枚）。
```

### 3.3 Self-Consistency（自洽性）

```python
# ============================================================
# Self-Consistency：多次采样，多数投票
# Self-Consistency: Multiple sampling + majority vote
# ============================================================

def self_consistency(prompt, n_samples=5, temperature=0.7):
    """多次采样取多数投票提高准确率."""
    answers = []
    for _ in range(n_samples):
        response = llm.invoke(prompt, temperature=temperature)
        answer = extract_answer(response)
        answers.append(answer)
    
    # 多数投票
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]
    return most_common[0], most_common[1] / n_samples

# 单次采样准确率: ~80%
# Self-Consistency (5次): ~92%
# Self-Consistency (10次): ~95%
```

### 3.4 Structured Output（结构化输出）

```python
# ============================================================
# OpenAI Function Calling / JSON Mode
# ============================================================
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",  # gpt-4-turbo 已换代
    # 更强约束可用 Structured Outputs：response_format={"type":"json_schema","json_schema":{...}}
    response_format={"type": "json_object"},
    messages=[{
        "role": "user",
        "content": """分析以下评论，以 JSON 输出：
        {"sentiment": "positive/negative", "topics": [...], "score": 1-5}
        
        评论：这家餐厅环境优雅，服务周到，但价格偏高。"""
    }]
)
# 输出保证是合法 JSON
```

### 3.5 推理模型与 Structured Outputs（2024-2025）

```python
# ============================================================
# (A) 推理模型：把"思考"交给模型，prompt 只给任务与约束
#     Reasoning models: delegate the thinking; prompt = task + constraints
# 代表：OpenAI o 系列 / Claude extended thinking / Gemini thinking / DeepSeek-R1
#   它们内置长思维链（自回归生成 reasoning tokens），无需手写 CoT。
#   Built-in long chain-of-thought — no hand-written CoT needed.
# ============================================================
# ❌ 反模式：对推理模型再加 "Let's think step by step"
#    既浪费 token，又干扰其内置推理（容易打断原生 reasoning）。
#    Anti-pattern: it wastes tokens AND disrupts the native reasoning.
# ✅ 正确：清晰陈述任务 + 约束，让模型自己规划推理路径。
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="o4-mini",  # 推理模型 / reasoning model（按需选 o4-mini 等）
    reasoning_effort="medium",  # low/medium/high：控制思考深度与成本
    messages=[{"role": "user",
               "content": "证明：任意 5 个整数中必有 3 个之和被 3 整除。给出严谨步骤。"}],
)
print(resp.choices[0].message.content)

# 何时用推理模型 vs 普通模型 / When to use which:
#   推理模型: 多步数学/算法证明/复杂规划/代码调试 —— 难但可验证的任务。
#   普通模型(gpt-4o 等): 抽取/分类/改写/对话/RAG —— 快、便宜、低延迟。
#   经验：能一步答出的别上推理模型（贵且慢）；需要"打草稿"才上。

# ============================================================
# (B) Structured Outputs：用 json_schema 强约束输出（替代旧 json_object）
#     旧 {"type":"json_object"} 只保证「是合法 JSON」，不保证字段/类型。
#     新 json_schema 在解码层强制贴合 schema —— 字段、类型、枚举全约束。
# ============================================================
from pydantic import BaseModel
from typing import List, Literal

class Review(BaseModel):  # Pydantic 即 schema / schema-as-code
    sentiment: Literal["positive", "negative", "neutral"]
    topics: List[str]
    score: int  # 1-5

# parse() 直接回传已校验的 Pydantic 实例 / returns a validated instance
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "评论：环境优雅、服务周到，但价格偏高。"}],
    response_format=Review,  # 等价于 {"type":"json_schema", "json_schema":{...}}
)
review: Review = completion.choices[0].message.parsed
print(review.sentiment, review.score)  # 直接当对象用，无需再 json.loads
```

## 4. 详细推理（Deep Dive）

### 4.1 CoT 为什么有效？

```
假设：CoT 让模型"使用更多的计算资源"

Without CoT:
  输入 → 一步推理 → 答案
  计算量: N tokens 的 attention

With CoT:
  输入 → 中间步骤 1 → ... → 中间步骤 K → 答案
  计算量: (N + K × M) tokens 的 attention
  
  → 中间步骤的 token 相当于"草稿纸"
  → 让模型在自回归框架中进行多步推理
  → 类似于人类"打草稿"的过程

这也是为什么 CoT 在小模型上不工作：
  小模型的 attention 容量不够容纳复杂推理链
```

### 4.2 技巧选择指南

```
简单任务（分类/提取）  → Few-shot 即可
中等任务（数学/逻辑）  → CoT
高准确率要求          → Self-Consistency
复杂多步任务          → ReAct (Agent)
需要结构化输出        → JSON Mode / Function Calling
```

## 5. 例题（Worked Examples）

```python
# CoT prompt 实战
cot_prompt = """
解决以下数学问题，请展示你的推理步骤：

问题：一个班有 45 名学生，男生比女生多 7 人，请问男生和女生各有多少人？

**推理步骤：**
"""
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 CoT 解决 5 道小学数学题，对比有 / 无 CoT 的答案准确率。

**练习 2：** 实现 Self-Consistency，对比单次采样和 5 次投票的效果。

### 进阶题

**练习 3：** 实现一个简单的 ReAct Agent：集成搜索工具和计算器。

**练习 4：** 设计 Tree-of-Thought prompt，解决 24 点游戏。
