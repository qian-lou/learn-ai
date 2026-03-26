# GloVe 与 FastText

## 1. 背景（Background）

> **为什么要学这个？**
>
> GloVe（2014，斯坦福）和 FastText（2016，Facebook）是 Word2Vec 之后最重要的两个词嵌入方法。GloVe 通过**全局共现矩阵**弥补了 Word2Vec 只看局部窗口的不足；FastText 通过**子词（subword）信息**解决了 Word2Vec 无法处理未登录词（OOV）的问题。
>
> FastText 的子词思想直接影响了现代分词器（BPE/SentencePiece）的设计。
>
> **在整个体系中的位置：** Word2Vec → GloVe → FastText → ELMo → BERT，是词表示从静态到动态的演进路线。

## 2. 知识点（Key Concepts）

| 特性 | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| 训练方式 | 预测式（窗口） | 统计+预测（共现矩阵） | 预测式（子词） |
| 全局信息 | ❌ 只有局部窗口 | ✅ 全局共现 | ❌ 局部窗口 |
| OOV 处理 | ❌ | ❌ | ✅ 子词拆解 |
| 形态学 | ❌ | ❌ | ✅ n-gram |
| 训练速度 | 快 | 中 | 中 |

## 3. 内容（Content）

### 3.1 GloVe 原理

```
GloVe = Global Vectors for Word Representation

核心思想：利用词-词共现矩阵中的统计信息

共现矩阵 X（以窗口大小 2 为例）：
  语料: "the cat sat on the mat"
  
       the  cat  sat  on  mat
  the   0    1    0   0    1
  cat   1    0    1   0    0
  sat   0    1    0   1    0
  on    0    0    1   0    1
  mat   1    0    0   1    0

目标函数：
  让 vec(i) · vec(j) + bias_i + bias_j ≈ log(X_ij)
  
  即：词向量的内积应该反映共现频率

优势：
  1. 利用全局统计 → 词类比效果更好
  2. 训练效率高（只需遍历非零共现对）
  3. 大规模预训练向量广泛可用
```

### 3.2 FastText 原理

```
FastText 的核心创新：子词（subword）表示

Word2Vec:  "where" → vec("where")  （整词一个向量）
FastText:  "where" → vec("<wh") + vec("whe") + vec("her") + vec("ere") + vec("re>")
                     （多个字符 n-gram 向量之和）

添加 < > 标记词的边界

优势：
  1. OOV 处理: "whereabouts" 虽未见过，可以从其子词推断
  2. 形态学:   "running","runner","runs" 共享 "run" 子词
  3. 拼写错误容忍: "teh" 与 "the" 共享 "th" 子词
```

```python
import gensim.downloader as api

# ============================================================
# 使用预训练 GloVe 向量
# Using pretrained GloVe vectors
# ============================================================
# glove = api.load("glove-wiki-gigaword-100")  # 100 维 GloVe
# print(glove.most_similar("king", topn=5))

# ============================================================
# 使用 FastText
# Using FastText
# ============================================================
from gensim.models import FastText

# 训练 FastText 模型
sentences = [
    ["i", "love", "machine", "learning"],
    ["deep", "learning", "is", "great"],
] * 100

ft_model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,           # Skip-gram
    min_n=3,        # 最短子词长度
    max_n=6,        # 最长子词长度
)

# FastText 可以为 OOV 词生成向量！
oov_vector = ft_model.wv["machinelearning"]  # 未见过的词也行
print(f"OOV 词向量维度: {oov_vector.shape}")
```

### 3.3 三种词嵌入对比

```python
# ============================================================
# 实验对比 / Experimental comparison
# ============================================================

# 词相似度任务
test_pairs = [("king", "queen"), ("cat", "dog"), ("good", "bad")]

# 词类比任务
# vec(king) - vec(man) + vec(woman) ≈ ?
analogies = [
    (["king", "woman"], ["man"]),    # → queen
    (["paris", "japan"], ["france"]),  # → tokyo
]
```

## 4. 详细推理（Deep Dive）

### 4.1 GloVe 为什么叫"全局向量"？

```
Word2Vec 只看局部窗口:
  "cat sat on the mat" → (cat, sat), (sat, on) ...
  每个窗口独立训练，信息利用不充分

GloVe 看全局共现:
  先统计整个语料的共现频率
  然后用共现频率指导训练
  
  → 同样的数据量，GloVe 学到的信息更充分
  → 在词类比任务上通常优于 Word2Vec
```

### 4.2 静态词嵌入的根本局限

```
所有静态嵌入（Word2Vec/GloVe/FastText）的共同问题：

  "bank" →  [0.3, -0.1, 0.8, ...]  （固定向量）
  
  但 "bank" 有多个含义：
    "river bank"  → 河岸
    "bank account" → 银行
  
  静态嵌入给它同一个向量！无法区分多义词。

这个问题催生了上下文嵌入（ELMo → BERT）：
  相同的词在不同上下文中产生不同的向量
```

## 5. 例题（Worked Examples）

### 例题：用 GloVe 初始化 PyTorch Embedding

```python
import torch
import torch.nn as nn
import numpy as np

def load_glove_embeddings(path, word2idx, embed_dim=100):
    """加载 GloVe 到 PyTorch Embedding."""
    matrix = np.random.normal(0, 0.1, (len(word2idx), embed_dim))
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2idx:
                matrix[word2idx[word]] = np.array(parts[1:], dtype=float)
    return torch.FloatTensor(matrix)

# embedding = nn.Embedding.from_pretrained(weights, freeze=False)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 对比 Word2Vec、GloVe 和 FastText 在词相似度任务上的表现。

**练习 2：** 测试 FastText 对 OOV 词的处理能力。

### 进阶题

**练习 3：** 用 GloVe 预训练向量初始化 LSTM 文本分类模型，对比随机初始化的效果。

**练习 4：** 分析 FastText 的 subword 分解：查看 "unbelievably" 被分解为哪些子词。
