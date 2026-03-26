# T5 及其他模型 / T5 and Other Models

## 1. 背景（Background）

> **为什么要学这个？**
>
> T5（Text-to-Text Transfer Transformer，2019，Google）提出了一个优雅的统一范式：**所有 NLP 任务都是"文本到文本"**。分类、翻译、摘要、问答——统统转换为输入一段文本、输出一段文本。这个思想深刻影响了后续的 FLAN、ChatGPT 等模型。
>
> 同时，开源大模型生态正在爆发：LLaMA（Meta）、Qwen（阿里）、Mistral、DeepSeek 等模型提供了 GPT-4 的替代方案。理解这些模型的异同对技术选型至关重要。
>
> **在整个体系中的位置：** T5 代表 Encoder-Decoder 路线，LLaMA/Qwen 代表开源 Decoder-only 路线。

## 2. 知识点（Key Concepts）

| 模型 | 架构 | 参数量 | 特点 |
|------|------|--------|------|
| T5 | Encoder-Decoder | 220M-11B | 统一 text-to-text 格式 |
| FLAN-T5 | Encoder-Decoder | 80M-11B | 指令微调版 T5 |
| LLaMA-2 | Decoder-only | 7B-70B | Meta 开源标杆 |
| LLaMA-3 | Decoder-only | 8B-70B | 性能大幅提升 |
| Qwen-2 | Decoder-only | 0.5B-72B | 中文优化，阿里开源 |
| Mistral | Decoder-only | 7B | Sliding Window Attention |
| DeepSeek-V2 | Decoder-only (MoE) | 236B (21B active) | MLA + MoE 效率优化 |

## 3. 内容（Content）

### 3.1 T5 的统一范式

```
T5 将所有任务统一为 text-to-text:

分类:
  输入: "sentiment: This movie is great!"
  输出: "positive"

翻译:
  输入: "translate English to German: That is good."  
  输出: "Das ist gut."

摘要:
  输入: "summarize: [长文本...]"
  输出: "[摘要文本]"

问答:
  输入: "question: What is AI? context: AI is..."
  输出: "Artificial Intelligence"

优势: 一个模型解决所有任务，统一训练和推理接口
```

### 3.2 使用 T5 和 FLAN-T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ============================================================
# FLAN-T5: 经过指令微调的 T5
# FLAN-T5: Instruction-tuned T5
# ============================================================
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# 翻译
inputs = tokenizer("Translate to French: How are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 摘要
text = "summarize: Transformers have revolutionized NLP. They use attention mechanisms..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 开源大模型生态

```
2024-2025 开源大模型景观:

Meta LLaMA 系列:
  LLaMA-1 (2023): 7B-65B, 开源先驱
  LLaMA-2 (2023): 7B-70B, 加入 GQA, 4K→32K 上下文
  LLaMA-3 (2024): 8B-70B, 15T tokens 训练, 128K 上下文

阿里 Qwen 系列:
  Qwen-2 (2024): 0.5B-72B, 中英双语优化
  Qwen-2.5 (2024): 数学/代码能力增强

Mistral/Mixtral:
  Mistral-7B: Sliding Window Attention, 32K 上下文
  Mixtral-8x7B: MoE 架构, 仅 13B active 参数

DeepSeek:
  DeepSeek-V2: MLA + MoE, 极致效率
  DeepSeek-Coder: 专注代码生成
```

### 3.4 模型选型指南

```
场景                推荐模型              理由
──────────────────────────────────────────────────
中文 NLP            Qwen-2.5             中文优化最好
代码生成            DeepSeek-Coder       代码专精
通用对话            LLaMA-3              社区生态最好
资源受限(单卡)      Qwen-2.5-7B          效果/成本平衡
翻译/摘要           FLAN-T5              Encoder-Decoder 适合
文本分类(传统)      BERT/RoBERTa         推理快、效果好
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 Decoder-only 成为主流？

```
Encoder-Decoder (T5):
  ✅ 翻译/摘要等任务效果好
  ❌ 参数利用率低（Encoder + Decoder 各一半参数）
  ❌ 不擅长自由对话

Decoder-only (GPT/LLaMA):
  ✅ 参数利用率高（所有参数都用于生成）
  ✅ 统一的自回归范式
  ✅ 规模够大后理解+生成都好
  ✅ 更适合对话/指令跟随
  
结论: 在大模型时代，Decoder-only + 指令微调 是最优路线
```

## 5. 例题（Worked Examples）

### 例题：对比不同模型在同一任务上的表现

```python
from transformers import pipeline

# 对比不同模型的零样本文本分类
classifiers = {
    "FLAN-T5": pipeline("text2text-generation", model="google/flan-t5-base"),
}

text = "Classify sentiment: This restaurant has amazing food!"
for name, clf in classifiers.items():
    result = clf(text, max_length=10)
    print(f"{name}: {result[0]['generated_text']}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 FLAN-T5 完成翻译、摘要、分类三种任务，体验统一 text-to-text 范式。

**练习 2：** 在 HuggingFace 上对比 LLaMA-3-8B 和 Qwen-2.5-7B 的中文理解能力。

### 进阶题

**练习 3：** 下载 Qwen-2.5-7B，用 `transformers` 进行本地推理，测量推理速度（tokens/sec）。

**练习 4：** 研究 MoE 架构的原理：为什么 Mixtral-8x7B 有 47B 参数但只需要 13B 的计算量？
