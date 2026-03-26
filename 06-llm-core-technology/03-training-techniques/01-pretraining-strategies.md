# 预训练策略 / Pretraining Strategies

## 1. 背景（Background）

> **为什么要学这个？**
>
> 预训练是大模型**获取知识**的阶段——模型在海量文本上学习语言规律、世界知识和推理能力。不同的预训练策略决定了模型的能力特长：MLM（BERT）擅长理解，CLM（GPT）擅长生成，Span Corruption（T5）兼顾两者。
>
> 对于 Java 工程师来说，预训练就像是**框架的初始化**——Spring Boot 启动时自动扫描、注入依赖；大模型预训练时自动从文本中"注入"语言知识和世界知识。
>
> **在整个体系中的位置：** 预训练 → SFT（监督微调）→ RLHF（对齐），三步构成大模型的完整训练流程。

## 2. 知识点（Key Concepts）

| 策略 | 代表模型 | 训练目标 | 适合任务 |
|------|---------|---------|---------|
| MLM | BERT | 预测 [MASK] | 理解（分类/NER） |
| CLM | GPT | 预测下一个 token | 生成（对话/写作） |
| Span Corruption | T5 | 预测被替换的片段 | 通用 |
| Prefix LM | GLM/U-PaLM | 前缀双向 + 后缀因果 | 通用 |

## 3. 内容（Content）

### 3.1 三种预训练策略详解

```
1. MLM (Masked Language Model) — BERT:

  输入: "The [MASK] sat on the [MASK]"
  目标: 预测被遮盖的词 → "cat", "mat"
  
  遮盖策略（15% 的 token）:
    80% → [MASK]（标准遮盖）
    10% → 随机词（增加鲁棒性）
    10% → 保持不变（减少训练/推理差异）
  
  优势: 双向上下文，理解能力强
  劣势: 不自然的 [MASK] token，不擅长生成


2. CLM (Causal Language Model) — GPT:

  输入: "The cat sat on"
  目标: 预测 → "the"
  
  P(x₁, x₂, ..., xₙ) = Π P(xᵢ | x₁, ..., xᵢ₋₁)
  
  优势: 自然的自回归生成
  劣势: 只能看到左边上下文


3. Span Corruption — T5:

  原文: "Thank you for <X> me to your party <Y> week"
  输入: "Thank you <X> me to your party <Y> week"
  目标: "<X> for inviting <Y> last </s>"
  
  随机遮盖连续片段（spans），而非单个词
  → 更好地学习短语和句法结构
```

### 3.2 预训练数据

```python
# ============================================================
# 大模型预训练数据组成（以 LLaMA-2 为例）
# Pretraining data composition (LLaMA-2 example)
# ============================================================

data_mix = {
    "Common Crawl (Web)": 67.0,    # 网页数据（最大来源）
    "C4": 15.0,                     # 清洗后的 Common Crawl
    "Wikipedia": 4.5,               # 百科知识
    "Books": 4.5,                   # 书籍
    "ArXiv": 2.5,                   # 学术论文
    "GitHub": 4.5,                  # 代码
    "StackExchange": 2.0,           # 问答
}

# 数据质量 > 数据数量
# LLaMA-1: 1.4T tokens → 非常好的效果
# GPT-3:   300B tokens → 效果也不错（但数据不够只能靠模型大）
```

### 3.3 预训练的统一视角

```
所有预训练策略的本质都是：
  给模型一些上下文 → 让模型预测缺失/后续的内容

区别在于"给多少上下文"和"预测什么":

MLM:   上下文 = 完整句子（但挖了洞）, 预测 = 洞里的 token
CLM:   上下文 = 前缀,               预测 = 下一个 token
Span:  上下文 = 挖了片段的文本,      预测 = 片段内容

这个统一视角让我们理解为什么 CLM 在规模够大后也能做理解任务：
  如果你能完美预测下一个词，你必须理解前面所有内容
  → "预测下一个 token" 本身就要求深度理解
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 CLM 成为主流？

```
MLM (BERT) vs CLM (GPT) 的竞争结果：

2018-2020: BERT 全面领先
  → 理解任务（GLUE/SuperGLUE）几乎都是 BERT 系列

2020-2022: GPT-3 展示了 CLM 的潜力
  → 规模够大时，CLM 的 few-shot 也能做理解任务

2023+: CLM 全面胜出
  → 理解 + 生成 + 推理 + 代码 + 工具使用
  → 一个模型搞定所有任务

根本原因：
  CLM 的自回归训练天然支持生成
  而 BERT 的 MLM 无法自然地生成长文本
  在"理解"vs"生成"的选择中，大模型时代选择了"生成"
```

### 4.2 预训练 vs 继续预训练

```
预训练 (From Scratch):
  从随机初始化开始训练 → 需要巨大算力
  只有少数公司/Lab 能做到（OpenAI, Google, Meta）

继续预训练 (Continual Pretraining):
  从已有模型开始 → 在特定领域数据上继续训练
  用途: 领域适配（法律、医疗、金融）
  成本: 预训练的 1/10 ~ 1/100
  
  例: LLaMA → 用中文数据继续预训练 → Chinese-LLaMA
```

## 5. 例题（Worked Examples）

### 例题：体验 MLM 预测

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")
results = fill_mask("The [MASK] of France is Paris.")
for r in results[:5]:
    print(f"  {r['token_str']:15s} score: {r['score']:.4f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 BERT 的 MLM 和 GPT-2 的 CLM 分别预测同一个句子中缺失的词，对比结果。

**练习 2：** 解释为什么 MLM 的遮盖策略中有 10% 保持不变。

### 进阶题

**练习 3：** 设计一个实验：分别用 MLM 和 CLM 预训练一个 Transformer，在下游分类任务上对比效果。

**练习 4：** 用 LLaMA 的 tokenizer 分析训练数据的 token 分布，理解 "1T tokens" 到底有多少文本。
