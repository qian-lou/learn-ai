# BERT 详解 / BERT in Depth

## 1. 背景（Background）

> **为什么要学这个？**
>
> BERT（2018, Google）开创了 **"预训练 + 微调"** 范式，是 NLP 领域的分水岭。它第一次证明：用大量无标注文本预训练一个通用模型，然后用少量标注数据微调，就能在几乎所有 NLP 任务上达到 SOTA。
>
> 对于 Java 工程师来说，BERT 就像 **Spring Boot 的自动配置**——你不需要从零构建，只需要在预训练好的模型上做少量定制（微调）即可获得优秀效果。
>
> **在整个体系中的位置：** BERT 是 Encoder-only 架构的代表。GPT 是 Decoder-only 的代表。理解 BERT 有助于理解为什么 GPT 选择了不同的路线。

## 2. 知识点（Key Concepts）

| 特性 | BERT-base | BERT-large |
|------|-----------|------------|
| 层数 | 12 | 24 |
| 隐藏维度 | 768 | 1024 |
| 注意力头数 | 12 | 16 |
| 参数量 | 110M | 340M |
| 预训练数据 | BooksCorpus + Wikipedia | 同左 |
| 预训练任务 | MLM + NSP | 同左 |

## 3. 内容（Content）

### 3.1 预训练任务

```
BERT 的两个预训练任务：

1. MLM (Masked Language Model) — 完形填空:
   输入: "I [MASK] machine [MASK]"
   目标: 预测 [MASK] = "love", "learning"
   
   随机遮盖 15% 的 token:
     80% → 替换为 [MASK]
     10% → 替换为随机词
     10% → 保持不变
   
   为什么不全用 [MASK]？
   → 微调时没有 [MASK]，训练和推理不一致（Exposure Bias）

2. NSP (Next Sentence Prediction) — 下一句预测:
   输入: "[CLS] 句子A [SEP] 句子B [SEP]"
   目标: 预测 B 是否是 A 的下一句
   
   ⚠️ 后续研究发现 NSP 贡献很小，RoBERTa 去掉了它
```

### 3.2 BERT 使用方式

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# ============================================================
# 1. 特征提取（不微调模型参数）
# Feature extraction (freeze model parameters)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

text = "BERT revolutionized natural language processing."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# [CLS] 向量作为句子表示
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
# 所有 token 的表示
all_embeddings = outputs.last_hidden_state  # [1, seq_len, 768]


# ============================================================
# 2. 微调做分类（Fine-tuning）
# Fine-tuning for classification
# ============================================================
classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# 微调训练
texts = ["This movie is great!", "Terrible film."]
labels = torch.tensor([1, 0])
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = classifier(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")
```

### 3.3 BERT 微调实战

```python
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                    max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# 训练循环
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)  # BERT 微调用小学习率

model.train()
for epoch in range(3):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 BERT 选择 Encoder，GPT 选择 Decoder？

```
BERT (Encoder-only, 双向):
  优势: 每个 token 都能看到完整上下文 → 理解能力强
  劣势: 不擅长生成（需要 [MASK] 才能预测）
  适合: 分类、NER、问答、句子相似度

GPT (Decoder-only, 单向):
  优势: 自然支持生成（自回归）
  劣势: 只能看到左边上下文 → 理解能力相对弱
  适合: 文本生成、对话、代码生成

后续证明: 当模型足够大时（GPT-3+），
  Decoder-only 在理解任务上也能追上甚至超越 BERT
  → 大模型时代 GPT 路线胜出
```

### 4.2 BERT 的变体

```
RoBERTa (2019): 去掉 NSP, 更多数据, 更大 batch
ALBERT (2019): 参数共享, 更小模型
DistilBERT (2019): 知识蒸馏, 参数减少 40%
DeBERTa (2020): 解耦注意力, 效果比 BERT 更好
```

## 5. 例题（Worked Examples）

### 例题：用 BERT 做 MLM 预测

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")
results = fill_mask("The capital of France is [MASK].")
for r in results[:3]:
    print(f"{r['token_str']:10s} 概率: {r['score']:.4f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 BERT 微调一个情感分类模型（IMDB 数据集）。

*参考答案*：复用 3.3 的训练框架，数据换成 IMDB（`datasets` 库一行加载）。

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
ds = load_dataset("imdb")                       # train/test 各 25000 条
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
args = TrainingArguments(output_dir="out", per_device_train_batch_size=16,
                         num_train_epochs=2, learning_rate=2e-5)  # 微调用小 lr
Trainer(model, args, train_dataset=enc["train"], eval_dataset=enc["test"]).train()
```
要点：微调 lr 取 `2e-5` 量级、1–3 个 epoch 足够，BERT-base 在 IMDB 上准确率约 92%–94%。`[CLS]` 经分类头输出 2 类 logits，损失为交叉熵。

**练习 2：** 对比 BERT-base 和 DistilBERT 在分类任务上的准确率和推理速度。

*参考答案*：把练习 1 的 checkpoint 换成 `distilbert-base-uncased` 再跑一遍即可对比。DistilBERT 通过知识蒸馏得到，**6 层（BERT-base 为 12 层）、参数减少约 40%（66M vs 110M）、推理快约 60%**，而在 GLUE/情感分类上保留约 **97% 的准确率**（典型仅低 1–2 个百分点）。结论：对延迟/成本敏感的生产场景，DistilBERT 是高性价比选择；追求极致精度才用 BERT-base/large。测速时注意固定 batch、序列长度并预热，用 `time` 统计 tokens/sec。

### 进阶题

**练习 3：** 用 BERT 做命名实体识别（NER），使用 `BertForTokenClassification`。

*参考答案*：NER 是 **token 级**分类，每个 token 预测 BIO 标签（如 `B-PER/I-PER/B-LOC/O`）。

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
ds = load_dataset("conll2003")                       # 标准 NER 数据集
tok = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)
```
关键难点：BERT 用子词（subword）分词，一个词可能切成多个 token，需用 `word_ids()` 做**标签对齐**——只给每个词的首个 subword 打真实标签，其余 subword 标为 `-100`（交叉熵忽略）。输出 logits 形状 `[B, N, num_labels]`，对每个 token 取 argmax。BERT-base-cased 在 CoNLL-2003 上 F1 约 91%。

**练习 4：** 分析 BERT 不同层的表示：浅层偏语法，深层偏语义。用 probing task 验证。

*参考答案*：探针（probing）做法：冻结 BERT，取某一层的隐藏状态作为特征，训一个**极简线性分类器**预测某种语言学属性，用该探针的准确率衡量"这层编码了多少此类信息"。

```python
model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
hs = model(**inputs).hidden_states   # tuple(len=13): 第0层为嵌入, 1..12为各层
# 对每层 hs[i] 取 token 表示 -> 训练线性探针预测 POS / 句法深度 / 共指
```
典型结论（Tenney et al. 2019 "BERT Rediscovers the Classical NLP Pipeline"、Jawahar et al. 2019）：**底层**编码表层与词性/句法信息（POS、成分边界），**中层**编码句法依存，**高层**编码语义与共指、语义角色。即 BERT 各层近似复刻了经典 NLP 流水线，从浅层语法到深层语义逐级抽象。
