# 机器翻译实战 / Machine Translation Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 机器翻译（Machine Translation）是 **Transformer 诞生的背景**——"Attention Is All You Need" 就是在机器翻译任务上验证的。通过实战完整的翻译项目，串联 Tokenization → Embedding → Encoder-Decoder → Attention → 生成 的全部知识。
>
> **在整个体系中的位置：** 机器翻译是 Seq2Seq 的经典应用。从 RNN+Attention 到 Transformer，翻译见证了 NLP 最重要的架构变革。

## 2. 知识点（Key Concepts）

| 组件 | 传统方案 | 现代方案 |
|------|---------|---------|
| 分词 | 词级分词 | BPE/SentencePiece |
| 编码器 | BiLSTM | Transformer Encoder |
| 解码器 | LSTM + Attention | Transformer Decoder |
| 评估指标 | BLEU | BLEU + COMET |
| 预训练模型 | - | mBART, NLLB, opus-mt |

## 3. 内容（Content）

### 3.1 使用预训练翻译模型

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# 方式 1: pipeline（最简单）
# Method 1: Pipeline (simplest)
# ============================================================
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
result = translator("Hello, how are you today?")
print(f"翻译结果: {result[0]['translation_text']}")


# ============================================================
# 方式 2: 手动加载模型（更灵活）
# Method 2: Manual loading (more flexible)
# ============================================================
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Machine learning is transforming every industry."
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,           # Beam Search
    length_penalty=1.0,
    early_stopping=True,
)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"翻译: {translation}")
```

### 3.2 Beam Search 解码

```
解码策略对比：

Greedy（贪心）：
  每步选概率最大的 token
  快速但可能错过更好的全局翻译

Beam Search（束搜索）：
  保留 top-K 个候选序列（beam_width=K）
  每步扩展所有候选，保留总概率最高的 K 个
  
  beam_width=1 → 退化为 Greedy
  beam_width=4 → 常用设置
  beam_width=10 → 质量更高但更慢

示例 (beam_width=2):
  Step 1: "I" → ["I love" (0.6), "I like" (0.3)]
  Step 2: 
    "I love" → ["I love you" (0.36), "I love it" (0.18)]
    "I like" → ["I like it" (0.21), "I like you" (0.06)]
  保留 top-2: ["I love you" (0.36), "I like it" (0.21)]
```

### 3.3 翻译评估指标 BLEU

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# ============================================================
# BLEU Score（翻译质量评估标准指标）
# BLEU Score (standard translation quality metric)
# ============================================================

reference = [["机器", "学习", "正在", "改变", "每个", "行业"]]
candidate = ["机器", "学习", "正在", "改变", "所有", "行业"]

# 句子级 BLEU
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score:.4f}")

# BLEU 原理:
# 计算 n-gram 精确率的几何平均值 + 短句惩罚
# BLEU-1: 1-gram 精确率（词匹配）
# BLEU-2: 2-gram 精确率（词对匹配）
# BLEU-4: 4-gram 精确率（常用指标）
#
# BLEU ∈ [0, 1]，越高越好
# 人工翻译: ~0.4-0.6（因为翻译有多种正确方式）
# 现代 NMT: ~0.3-0.5
```

### 3.4 批量翻译

```python
# ============================================================
# 批量翻译 / Batch translation
# ============================================================
texts = [
    "The weather is nice today.",
    "I want to learn artificial intelligence.",
    "Deep learning has revolutionized NLP.",
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_length=128, num_beams=4)

translations = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
for src, tgt in zip(texts, translations):
    print(f"EN: {src}")
    print(f"ZH: {tgt}")
    print()
```

## 4. 详细推理（Deep Dive）

### 4.1 从 RNN 翻译到 Transformer 翻译

```
RNN Seq2Seq + Attention (2015-2017):
  - 顺序处理，训练慢
  - 长序列效果差
  - BLEU ~25-30

Transformer (2017+):
  - 完全并行，训练快 3-5x
  - 长序列效果好
  - BLEU ~35-45

大模型翻译 (GPT-4, 2023+):
  - 零样本翻译，无需专门训练
  - 理解上下文和习语
  - 某些语言对上接近人工水平
```

### 4.2 翻译中的常见问题

```
1. 长度控制:
   生成太短或太长 → length_penalty 参数
   
2. 重复问题:
   "the the the..." → repetition_penalty / no_repeat_ngram_size
   
3. 多译问题:
   一个句子有多种正确翻译 → BLEU 天花板较低
   
4. 低资源语言:
   训练数据少的语言翻译质量差 → 多语言预训练（mBART/NLLB）
```

## 5. 例题（Worked Examples）

### 例题：对比不同模型的翻译质量

```python
models = [
    "Helsinki-NLP/opus-mt-en-zh",
    # "facebook/nllb-200-distilled-600M",
]

text = "Artificial intelligence will transform education fundamentally."

for model_name in models:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inp = tok(text, return_tensors="pt")
    out = mdl.generate(**inp, max_length=128)
    result = tok.decode(out[0], skip_special_tokens=True)
    print(f"{model_name.split('/')[-1]}: {result}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 `Helsinki-NLP/opus-mt-en-zh` 翻译 10 个英文句子，人工评判翻译质量。

**练习 2：** 对比 `num_beams=1`（Greedy）和 `num_beams=4`（Beam Search）的翻译质量差异。

### 进阶题

**练习 3：** 用 BLEU 评估翻译质量：收集 50 个英中平行句对，对比不同模型的 BLEU 分数。

**练习 4：** 尝试用 GPT 做翻译（通过 prompt），对比专门的翻译模型和通用 LLM 的翻译效果。

> **提示：** Prompt 如 `"Translate the following English text to Chinese:\n\n{text}\n\nTranslation:"`
