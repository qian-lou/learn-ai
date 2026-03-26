# Prompt Engineering 基础 / Prompt Engineering Basics

## 1. 背景（Background）

> **为什么要学这个？**
>
> Prompt Engineering 是**不改变模型参数**，仅通过设计输入提示来引导模型输出的技术。它是使用大模型**最重要的实用技能**——同一个模型，好的 prompt 和差的 prompt 效果可以天壤之别。
>
> 对于 Java 工程师来说，Prompt Engineering 就像是**设计 API 的请求参数**——接口不变（模型不变），但通过精心设计的输入参数（prompt），获得完全不同的输出。

## 2. 知识点（Key Concepts）

| 技巧 | 描述 | 效果 |
|------|------|------|
| Zero-shot | 直接提问 | 基础 |
| Few-shot | 给示例引导 | 显著提升 |
| System Prompt | 设定角色和规则 | 控制风格 |
| 输出约束 | 指定格式（JSON） | 结构化输出 |
| 分步指令 | 拆解复杂任务 | 提升准确性 |

## 3. 内容（Content）

### 3.1 基础 Prompt 模式

```
# ============================================================
# 1. Zero-shot（零样本）
# ============================================================
将以下文本分类为正面或负面情感：
文本：这家餐厅的服务太棒了！
分类：


# ============================================================
# 2. Few-shot（少样本）— 效果显著提升
# ============================================================
将以下文本分类为正面或负面情感：

文本：这个产品太棒了！ → 正面
文本：服务态度很差。 → 负面
文本：价格合理，值得购买。 → 正面
文本：今天的晚餐真难吃。 → 负面
文本：客服响应很快，问题解决了。 →


# ============================================================
# 3. System Prompt（系统提示）
# ============================================================
你是一位专业的 Java 架构师，精通 Spring Boot 和微服务。
用户会问你技术问题，请用以下格式回答：
1. 先用一句话总结答案
2. 给出详细解释
3. 提供代码示例
4. 指出常见陷阱


# ============================================================
# 4. 输出格式约束
# ============================================================
分析以下文本的情感，以 JSON 格式输出：
{
  "text": "原文",
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键词列表"]
}

文本：这家餐厅环境优雅，但菜品一般。
```

### 3.2 Prompt 模板设计

```python
# ============================================================
# Python 中的 Prompt 模板
# Prompt template in Python
# ============================================================

def create_classification_prompt(text: str, categories: list[str], examples: list[dict] = None) -> str:
    """创建分类 Prompt / Create classification prompt."""
    prompt = f"将以下文本分类为以下类别之一：{', '.join(categories)}\n\n"
    
    if examples:
        for ex in examples:
            prompt += f"文本：{ex['text']} → {ex['label']}\n"
        prompt += "\n"
    
    prompt += f"文本：{text} →"
    return prompt


def create_extraction_prompt(text: str, fields: list[str]) -> str:
    """创建信息提取 Prompt / Create extraction prompt."""
    return f"""从以下文本中提取信息，以 JSON 格式输出。
需要提取的字段：{', '.join(fields)}

文本：{text}

JSON 输出："""


# 使用
prompt = create_classification_prompt(
    "这部电影太精彩了！",
    categories=["正面", "负面", "中性"],
    examples=[
        {"text": "很棒", "label": "正面"},
        {"text": "太差了", "label": "负面"},
    ]
)
```

### 3.3 Prompt 设计原则

```
六大原则：

1. 明确具体: "分析这段文本" ❌  →  "从这段文本中提取人名和地名" ✅
2. 提供示例: Few-shot 几乎总是比 Zero-shot 好
3. 指定格式: 明确告诉模型用什么格式输出
4. 分步拆解: 复杂任务拆成多个简单步骤
5. 设定角色: "你是一位资深数据分析师..."
6. 添加约束: "回答不超过 100 字"，"只用中文"

常见错误：
  ❌ Prompt 过长（模型容易忽略中间部分）
  ❌ 指令矛盾（"简短回答" + "详细解释"）
  ❌ 没有示例就期待复杂格式输出
```

## 4. 详细推理（Deep Dive）

### 4.1 Prompt 为什么有效？

```
原因：大模型在预训练时学到了语言的模式匹配能力

Few-shot 的本质:
  模型看到 "A→B, C→D, E→?" 的模式
  利用 In-Context Learning 推断 "?" 应该遵循同样的映射

System Prompt 的本质:
  大模型在预训练时见过大量 "角色扮演" 文本
  "你是一位医生" 激活了与医学知识相关的参数

格式约束的本质:
  模型学习了 JSON/XML 等格式的规律
  在 prompt 中指定格式 → 模型延续这个格式
```

## 5. 例题（Worked Examples）

```python
# 对比 Zero-shot vs Few-shot
from openai import OpenAI
client = OpenAI()

# Zero-shot
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "翻译成英文：机器学习"}]
)

# Few-shot（通常更好）
messages = [
    {"role": "system", "content": "你是专业翻译，保持技术术语准确性。"},
    {"role": "user", "content": "深度学习"},
    {"role": "assistant", "content": "Deep Learning"},
    {"role": "user", "content": "机器学习"},
]
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 设计 Prompt 让 LLM 做情感分析，对比 zero-shot 和 few-shot 效果。

**练习 2：** 设计一个 System Prompt，让模型扮演 Java 代码审查专家。

### 进阶题

**练习 3：** 设计一个多步 Prompt 完成复杂任务：先提取关键信息，再生成摘要，最后翻译。

**练习 4：** 构建一个 Prompt 评估框架，量化比较不同 prompt 在 50 个测试用例上的准确率。
