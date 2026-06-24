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

*参考答案*：

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()

sents = [
    "I deposited money at the bank.",          # 金融
    "The bank approved my loan.",              # 金融
    "We sat on the river bank.",               # 河岸
    "The boat reached the left bank.",         # 河岸
    "She works as a bank teller.",             # 金融
]

def bank_vec(sentence):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state[0]        # [seq_len, 768]
    ids = enc["input_ids"][0].tolist()
    pos = ids.index(tok.convert_tokens_to_ids("bank")) # 找 "bank" 的位置
    return out[pos]                                    # [768]

vecs = torch.stack([bank_vec(s) for s in sents])       # [5, 768]
sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
print(sim.round(decimals=2))
```

预期结论：**同义语境的 "bank" 之间相似度更高，跨义语境更低**。即句 0/1/4（金融）两两相似度明显高于"金融 vs 河岸"对（如句 0 与句 2）。这验证了 BERT 的上下文嵌入能区分多义词——同一个词在不同上下文产生不同向量，这是静态嵌入（Word2Vec/GloVe）做不到的。注意：BERT 各向量本身余弦相似度普遍偏高（各向异性），所以看**相对差异**而非绝对值。

**练习 2：** 对比 `[CLS]` 嵌入和 Mean Pooling 在句子相似度任务上的效果。

*参考答案*：

对每句话分别取 `[CLS]` 向量和 mean-pooling 向量，在 STS（语义文本相似度）数据上算与人工分的 Spearman 相关。

```python
import torch

def cls_embed(out):
    return out.last_hidden_state[:, 0, :]            # [B, 768]

def mean_embed(out, mask):
    m = mask.unsqueeze(-1).float()                   # [B, T, 1]
    summed = (out.last_hidden_state * m).sum(1)      # 只对非 padding 求和
    return summed / m.sum(1).clamp(min=1e-9)         # [B, 768]
```

结论：**对未经句向量微调的原生 BERT，Mean Pooling 通常优于 `[CLS]`**。

- 原因：BERT 的 `[CLS]` 是为预训练的 NSP 任务服务的，**没有专门优化成"句子语义摘要"**；直接拿来做相似度往往偏弱，甚至不如简单的词向量平均。Mean Pooling 综合了所有 token 信息，更稳。
- 重要前提：两者**原生效果都不理想**——直接用 BERT 做句子相似度普遍较差。要做好句向量应使用 **Sentence-BERT**（用孪生网络 + 对比/回归目标微调），它默认就用 mean pooling 并显著超过原生 BERT 的任一种取法。所以这道题的实践启示是：句子相似度别直接用 `[CLS]`，要么 mean pooling，要么直接上 SBERT。

### 进阶题

**练习 3：** 用 Sentence-BERT 实现一个简单的语义搜索引擎：输入查询，返回最相似的文档。

*参考答案*：

离线把文档编码成向量库，查询时编码 query 再算余弦相似度取 top-k。

```python
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    """基于 Sentence-BERT 的语义搜索 / SBERT semantic search."""
    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        # 文档向量库（归一化便于点积=余弦）/ encode corpus once
        self.doc_emb = self.model.encode(docs, convert_to_tensor=True,
                                         normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3):
        q = self.model.encode(query, convert_to_tensor=True,
                              normalize_embeddings=True)            # [384]
        scores = util.cos_sim(q, self.doc_emb)[0]                  # [N_docs]
        top = scores.topk(top_k)
        return [(self.docs[i], float(s)) for s, i in zip(top.values, top.indices)]

engine = SemanticSearch([
    "How to learn Python programming?",
    "Best framework for deep learning",
    "Where to buy fresh coffee beans",
])
print(engine.search("study python coding", top_k=2))
```

关键点：语义搜索的优势在于**匹配的是含义而非字面**——查询 "study python coding" 能命中 "learn Python programming"，即使没有一个词完全相同，这是 TF-IDF 关键词检索做不到的。文档向量应**离线预计算并缓存**（甚至存入向量数据库如 FAISS/Milvus）；归一化后用点积即等价余弦相似度。规模大时用 ANN 索引把检索从 O(N) 降到近似 O(log N)。

**练习 4：** 对比 BERT 不同层的嵌入在 NER 任务上的表现。提示：浅层更擅长语法，深层更擅长语义。

*参考答案*：

用 `output_hidden_states=True` 拿到全部 13 层（含 embedding 层）的隐状态，逐层冻结取特征 + 训练一个轻量分类头，比较各层在 NER 上的 F1。

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-cased",
                                  output_hidden_states=True).eval()

with torch.no_grad():
    out = model(**enc)
# hidden_states: 长度 13 的元组，每个 [B, T, 768]
# 第 0 层 = 词嵌入，第 1~12 层 = 各 Transformer 层输出
all_layers = out.hidden_states
layer_k = all_layers[k]                       # 取第 k 层做 NER 特征
# 对每个 k：冻结 BERT，仅用 layer_k 训练一个线性/CRF 头，记录 dev F1
```

预期结论（与 BERT 探针研究一致）：
- **浅层（靠近输入）**编码更多**表层/句法信息**（词形、词性、局部结构）。
- **深层**编码更多**语义/上下文信息**（指代、语义角色、词义消歧）。
- NER 既需要词形线索又需要上下文语义，因此**最佳单层通常落在中高层（约第 9~12 层附近）**，而非最顶层或最底层；纯顶层有时反而偏弱（顶层更偏向预训练目标 MLM）。
- 工程上更常用的是**拼接或加权多层**（如经典做法：concat 最后 4 层），效果优于任一单层——这也解释了为什么 ELMo 要对各层做加权求和。
