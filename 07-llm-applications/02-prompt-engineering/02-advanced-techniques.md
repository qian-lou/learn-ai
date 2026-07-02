# 高级提示技巧 / Advanced Prompt Techniques

## 1. 背景（Background）

> **为什么要学这个？**
>
> 基础 Prompt 技巧解决简单任务，但复杂推理需要**高级策略**。Chain-of-Thought（思维链）在 GSM8K 数学基准上把准确率从约 ~55% 提到 ~92%（约 37 个百分点，示例数据），ReAct 让模型能够调用工具完成真实任务。
>
> 这些技巧是构建 AI Agent 的理论基础。
>
> 对于 Java 工程师来说：Self-Consistency ≈ 集群多副本"多数表决"（跑 N 次取众数提高可靠性）；ReAct ≈ Controller 的请求循环——调用 Service(工具)、看返回、再决定下一步，直到给出最终响应。

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
# CoT 的效果（GPT-4 在 GSM8K 数学基准上，示例数据）:
# ============================================================
# 无 CoT: ~55% 准确率
# 有 CoT: ~92% 准确率 → 约提升 37 个百分点！
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

# parse() 直接回传已校验的 Pydantic 实例；新版 SDK 用去 beta 的稳定路径 / stable path
completion = client.chat.completions.parse(
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

*参考答案*：同一批题目跑两版 prompt——直接问 vs 追加"一步步思考"，各自抽取末尾数字与标准答案比对，统计命中率。

```python
import re
from openai import OpenAI
client = OpenAI()

# (题目, 标准答案) / (question, gold)
problems = [
    ("15 个苹果卖了 8 个又进 12 个，现在多少个？", 19),
    ("小明有 23 元，买 3 本 5 元的书，找回多少元？", 8),
    ("一列火车 60km/h 跑 2.5 小时走多远(km)？", 150),
    ("班里 45 人，男生比女生多 7 人，男生几人？", 26),
    ("一箱 24 瓶，喝掉四分之三，还剩几瓶？", 6),
]

def ask(q: str, cot: bool) -> str:
    # 有 CoT 追加触发句，无 CoT 强制只给数字 / with/without the trigger sentence
    suffix = "\n请一步步思考，最后一行只写数字答案。" if cot else "\n只输出数字答案。"
    return client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": q + suffix}]
    ).choices[0].message.content

def last_int(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text)          # 取末尾数字作最终答案 / final number
    return int(nums[-1]) if nums else None

def accuracy(cot: bool) -> float:
    hits = sum(last_int(ask(q, cot)) == gold for q, gold in problems)
    return hits / len(problems)

print(f"无 CoT: {accuracy(False):.0%} | 有 CoT: {accuracy(True):.0%}")
# 预期：有 CoT 明显更高，尤其多步题 / CoT wins on multi-step problems
```

**练习 2：** 实现 Self-Consistency，对比单次采样和 5 次投票的效果。

*参考答案*：调高 temperature 采样多条推理路径，抽取各自答案后取众数——比单条采样更抗随机波动。

```python
import re
from collections import Counter
from openai import OpenAI
client = OpenAI()

question = ("一个水池有进水管和出水管。进水管 6 小时注满，出水管 8 小时放空。"
            "两管同开，多少小时注满？请一步步思考，最后一行只写数字答案。")

def sample_answer(temperature: float) -> float | None:
    text = client.chat.completions.create(
        model="gpt-4o-mini", temperature=temperature,
        messages=[{"role": "user", "content": question}]
    ).choices[0].message.content
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None

# 单次采样：一条路径，受随机性影响 / single path
single = sample_answer(temperature=0.7)

# Self-Consistency：多条路径投票取众数 / majority vote over n paths
# 时间 O(n) 采样 空间 O(n) 存答案
votes = [a for _ in range(5) if (a := sample_answer(0.7)) is not None]
final = Counter(votes).most_common(1)[0][0] if votes else None
print(f"单次: {single} | 5 次投票: {final} (票数分布 {Counter(votes)})")
# 正确答案 24；投票通常比单次更稳定命中 / voting is more robust
```

### 进阶题

**练习 3：** 实现一个简单的 ReAct Agent：集成搜索工具和计算器。

*参考答案*：手写最小 ReAct 循环——system prompt 约定 `Thought/Action/Observation` 协议，代码解析模型输出的 Action、执行工具、把结果作为 Observation 回灌，直到出现 `Answer`。

```python
import re
from openai import OpenAI
client = OpenAI()

# ---- 工具集 / tools ----
def search(q: str) -> str:  # 这里用 mock，真实场景接搜索 API / mock search
    kb = {"埃菲尔铁塔高度": "330 米", "光速": "约 300000 km/s"}
    return next((v for k, v in kb.items() if k in q), "未找到")

def calc(expr: str) -> str:
    # 仅解析数字与四则运算，绝不用 eval / safe arithmetic, never eval
    if not re.fullmatch(r"[\d\.\s\+\-\*\/\(\)]+", expr):
        return "非法表达式"
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def ev(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):    return ops[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError
    return str(ev(ast.parse(expr, mode="eval").body))

TOOLS = {"search": search, "calc": calc}

SYSTEM = """你是 ReAct Agent，按如下协议逐步作答，每次只输出一行：
Thought: <推理>
Action: <search 或 calc>[<参数>]
拿到 Observation 后继续；得出结论时输出：
Answer: <最终答案>"""

def react(question: str, max_steps: int = 5) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    for _ in range(max_steps):                       # 限制步数防死循环 / cap loops
        out = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0, messages=msgs
        ).choices[0].message.content.strip()
        msgs.append({"role": "assistant", "content": out})
        if out.startswith("Answer:"):
            return out
        m = re.search(r"Action:\s*(\w+)\[(.*?)\]", out)   # 解析工具调用 / parse action
        obs = TOOLS[m.group(1)](m.group(2)) if m and m.group(1) in TOOLS else "无效 Action"
        msgs.append({"role": "user", "content": f"Observation: {obs}"})
    return "超出最大步数"

print(react("埃菲尔铁塔的高度乘以 3 是多少米？"))
```

**练习 4：** 设计 Tree-of-Thought prompt，解决 24 点游戏。

*参考答案*：ToT 让模型显式"广度搜索"——每步生成多个候选中间状态、自评保留有希望的分支，而非一条路走到黑。用一个 prompt 引导它枚举分支并剪枝。

```python
from openai import OpenAI
client = OpenAI()

# ToT prompt：要求模型分层展开候选、自评剪枝、回溯 / branch, self-evaluate, backtrack
TOT_PROMPT = """用 Tree-of-Thought 方法解 24 点：给定 4 个数，用 + - * / 和括号算出 24，每个数用一次。

按以下步骤显式搜索，不要一步到位：
1. 【展开】从 4 个数中任选两个做一次运算，列出 3-5 个有希望的中间结果（3 个数的新状态）。
2. 【自评】给每个中间状态标注 "可能/不太可能" 到达 24，剪掉不太可能的分支。
3. 【递归】对保留的分支重复步骤 1-2，直到只剩一个数。
4. 若某分支等于 24，输出完整算式；若走不通，回溯换分支。

数字：{numbers}
最后一行给出：答案：<完整算式> = 24"""

def solve_24(numbers: list[int]) -> str:
    return client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": TOT_PROMPT.format(numbers=numbers)}]
    ).choices[0].message.content

print(solve_24([4, 6, 8, 2]))   # 如 (8-6)*(4+8)? 由模型搜索给出 / model searches a valid expr
```
