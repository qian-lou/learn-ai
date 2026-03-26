# 上下文嵌入 / Contextual Embeddings

## 1. 背景（Background）

> **为什么要学这个？**
>
> 上下文嵌入是 NLP 从"静态词向量"到"预训练大模型"的**转折点**。ELMo（2018）首次提出"用上下文动态决定词向量"，BERT（2018）将这个思想推到了极致——同一个词在不同句子中会产生完全不同的向量。
>
> 这个转变类比：静态嵌入就像 Java 的 `final` 常量（编译时确定），上下文嵌入就像运行时通过反射获取的值（取决于运行时上下文）。
>
> **在整个体系中的位置：** Word2Vec（静态）→ ELMo（浅层上下文）→ BERT（深层双向上下文）→ GPT（自回归上下文）。上下文嵌入是现代大模型的基础。

## 2. 知识点（Key Concepts）

| 模型 | 年份 | 上下文方式 | 预训练任务 | 典型用途 |
|------|------|-----------|-----------|----------|
| ELMo | 2018 | BiLSTM | 双向语言模型 | 特征提取 |
| BERT | 2018 | Transformer Encoder | MLM + NSP | 理解任务 |
| GPT | 2018 | Transformer Decoder | 自回归 LM | 生成任务 |
| T5 | 2019 | Encoder-Decoder | Span Corruption | 通用 |

## 3. 内容（Content）

### 3.1 从静态到动态

```
静态嵌入 vs 上下文嵌入：

Word2Vec（静态）：
  "bank" → [0.3, -0.1, 0.8] （始终相同）
  
  "I went to the bank to deposit money."  bank = [0.3, -0.1, 0.8]
  "I sat on the river bank."              bank = [0.3, -0.1, 0.8]  ← 相同！

BERT（上下文）：
  "I went to the bank to deposit money."  bank = [0.5, 0.2, -0.1]  ← 金融语义
  "I sat on the river bank."              bank = [-0.3, 0.7, 0.4]  ← 自然语义
  
  同一个词，不同上下文 → 不同向量！
```

### 3.2 使用 BERT 提取上下文嵌入

```python
import torch
from transformers import AutoModel, AutoTokenizer

# ============================================================
# 用 BERT 提取上下文嵌入
# Extract contextual embeddings with BERT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

# 编码两个不同语境的 "bank"
sentences = [
    "I went to the bank to deposit money.",  # bank = 银行
    "I sat on the river bank watching fish.", # bank = 河岸
]

# 获取上下文嵌入
with torch.no_grad():
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
        
        # 找 "bank" 的位置
        tokens = tokenizer.tokenize(sent)
        bank_idx = tokens.index("bank") + 1  # +1 因为 [CLS]
        bank_vector = embeddings[0, bank_idx]
        
        print(f"句子: {sent}")
        print(f"bank 向量前 5 维: {bank_vector[:5].tolist()}")
        print()

# 计算两个 "bank" 向量的余弦相似度
# 如果模型理解了多义词，相似度应该较低
```

### 3.3 BERT 嵌入的三种使用方式

```python
# ============================================================
# 方式 1: 词级嵌入 / Word-level embedding
# 用于需要词级别表示的任务（NER、词性标注）
# ============================================================
word_embeddings = outputs.last_hidden_state  # [B, T, 768]

# ============================================================
# 方式 2: [CLS] 嵌入 / [CLS] embedding  
# 用于句子分类任务
# ============================================================
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

# ============================================================
# 方式 3: 句子嵌入（Mean Pooling）
# Sentence embedding (Mean Pooling)
# 用于句子相似度、检索任务
# ============================================================
attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
token_embeddings = outputs.last_hidden_state  # [B, T, 768]
# 只对非 padding token 做平均
sentence_embedding = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
```

### 3.4 Sentence-BERT 与语义搜索

```python
from sentence_transformers import SentenceTransformer

# ============================================================
# Sentence-BERT: 专门优化的句子嵌入模型
# Sentence-BERT: Optimized sentence embedding model
# ============================================================

model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级

sentences = [
    "How to learn Python programming?",
    "Best way to study Python coding?",
    "Where is the nearest coffee shop?",
]

embeddings = model.encode(sentences)
print(f"嵌入维度: {embeddings.shape}")  # [3, 384]

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print(f"Python 两句相似度: {sim_matrix[0][1]:.4f}")  # 高
print(f"Python vs Coffee: {sim_matrix[0][2]:.4f}")    # 低
```

## 4. 详细推理（Deep Dive）

### 4.1 从 ELMo 到 BERT 的关键区别

```
ELMo (2018):
  架构: 2 层 BiLSTM
  用法: 预训练后提取特征，作为下游模型的输入
  向量: concat(前向LSTM, 后向LSTM) → 拼接而非融合
  缺点: 双向信息没有真正交互

BERT (2018):
  架构: 12 层 Transformer Encoder
  用法: 预训练后 fine-tune 整个模型
  向量: 每层都做完全的双向 self-attention
  优势: 真正的深度双向上下文理解

关键区别:
  ELMo:  h = [LSTM_forward; LSTM_backward]  ← 拼接
  BERT:  h = Attention(所有位置)            ← 交互

这就是为什么 BERT 在几乎所有 NLP 任务上都超越了 ELMo
```

## 5. 例题（Worked Examples）

### 例题：可视化多义词的上下文嵌入

```python
# 收集 "apple" 在不同语境下的向量
contexts = [
    "I bought an Apple iPhone yesterday.",   # 苹果公司
    "I ate a delicious apple for lunch.",      # 水果
    "Apple announced new products today.",     # 苹果公司
    "The apple fell from the tree.",          # 水果
]
# 提取向量后用 PCA 降维，看它们是否聚类
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 BERT 提取 "bank" 在 5 个不同语境中的向量，计算两两余弦相似度，验证多义词区分能力。

**练习 2：** 对比 `[CLS]` 嵌入和 Mean Pooling 在句子相似度任务上的效果。

### 进阶题

**练习 3：** 用 Sentence-BERT 实现一个简单的语义搜索引擎：输入查询，返回最相似的文档。

**练习 4：** 对比 BERT 不同层的嵌入在 NER 任务上的表现。提示：浅层更擅长语法，深层更擅长语义。
