# Word2Vec（CBOW / Skip-gram）

## 1. 背景（Background）

> **为什么要学这个？**
>
> Word2Vec（2013，Google）开创了**用神经网络学习词向量**的先河，让 NLP 从稀疏表示（One-Hot/TF-IDF）迈入了稠密向量时代。它的核心洞察——**"一个词的含义由它的上下文决定"**（分布式假设）——至今仍是所有语言模型的基础。
>
> Transformer 中的 `nn.Embedding` 层，本质上就是 Word2Vec 思想的延续。理解 Word2Vec 对理解大模型的嵌入层至关重要。
>
> **在整个体系中的位置：** Word2Vec 是从 TF-IDF 到 BERT 的关键桥梁。它首次证明了"词嵌入空间中的线性关系可以编码语义"。

## 2. 知识点（Key Concepts）

| 概念 | CBOW | Skip-gram |
|------|------|-----------|
| 任务 | 上下文预测中心词 | 中心词预测上下文 |
| 输入 | 周围的 K 个词 | 一个中心词 |
| 输出 | 预测中心词 | 预测周围的词 |
| 适合 | 高频词效果好 | 低频词效果好 |
| 速度 | 快 | 慢（但效果更好）|

**核心公式：**
```
Skip-gram 目标函数：
  最大化 P(context | center) = softmax(v_context · v_center)
  
"king" - "man" + "woman" ≈ "queen"
  因为词向量空间中编码了"性别"方向
```

## 3. 内容（Content）

### 3.1 Skip-gram 原理

```
Skip-gram 训练过程：

输入文本: "The cat sat on the mat"
窗口大小 = 2

中心词 "sat" 的训练样本:
  (sat, The)    → 正样本
  (sat, cat)    → 正样本
  (sat, on)     → 正样本
  (sat, the)    → 正样本

目标: 让 sat 的向量与 cat, on 等上下文词的向量相似
      让 sat 的向量与随机词（如 banana）的向量不相似

→ 经过足够多的训练，语义相似的词向量会自然聚在一起
```

### 3.2 使用 Gensim 训练 Word2Vec

```python
from gensim.models import Word2Vec

# ============================================================
# 训练 Word2Vec 模型
# Train Word2Vec model
# ============================================================

# 准备语料（每个句子是词列表）
sentences = [
    ["i", "love", "machine", "learning"],
    ["deep", "learning", "is", "amazing"],
    ["natural", "language", "processing", "is", "fun"],
    ["word", "embeddings", "capture", "semantics"],
] * 100  # 重复以增加训练量

# 训练 Skip-gram 模型
model = Word2Vec(
    sentences,
    vector_size=100,    # 向量维度
    window=5,           # 上下文窗口
    min_count=1,        # 最低词频
    sg=1,               # 1=Skip-gram, 0=CBOW
    workers=4,          # 并行训练线程
    epochs=20,
)

# 查看词向量
vector = model.wv["learning"]
print(f"'learning' 向量维度: {vector.shape}")  # (100,)

# 相似词查询
similar = model.wv.most_similar("learning", topn=5)
print(f"与 'learning' 最相似的词: {similar}")


# ============================================================
# 使用预训练词向量
# Using pretrained word vectors
# ============================================================
import gensim.downloader as api

# 加载 Google News 预训练向量（300 维，300万词）
# model = api.load('word2vec-google-news-300')

# 经典的词类比实验
# result = model.most_similar(positive=['king', 'woman'], negative=['man'])
# → [('queen', 0.71), ...]
```

### 3.3 负采样（Negative Sampling）

```
原始 Skip-gram 的问题：
  softmax 需要对整个词表（V 个词）计算分母
  V 可能是 10 万+  →  太慢！

负采样解决方案：
  不计算完整的 softmax
  只对 K 个随机"负样本"做二分类
  
  正样本: (center="sat", context="cat") → 标签 1
  负样本: (center="sat", random="banana") → 标签 0（随机采样 5-20 个）
  
  用 sigmoid 二分类代替 softmax 多分类
  训练速度提升 100x+
```

### 3.4 Word2Vec 与 nn.Embedding 的关系

```python
import torch
import torch.nn as nn

# ============================================================
# nn.Embedding 本质上就是一个查找表
# nn.Embedding is essentially a lookup table
# ============================================================

# Word2Vec 训练出的词向量可以用来初始化 nn.Embedding
vocab_size = 10000
embed_dim = 300

# 随机初始化（默认）
embedding = nn.Embedding(vocab_size, embed_dim)

# 用 Word2Vec 预训练向量初始化
# pretrained_weights = word2vec_model.wv.vectors  # numpy array
# embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_weights))

# 查找: token ID → 向量
token_ids = torch.tensor([0, 512, 1024])
vectors = embedding(token_ids)  # Shape: [3, 300]
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么词类比有效？

```
"king" - "man" + "woman" ≈ "queen"

本质：词向量空间中存在平行的语义关系方向

                 queen
                  /|
    "gender"方向 / |
                /  |
             king  |
              |    |
  "royalty"方向|    |
              |    |
            man    woman

  vec(king) - vec(man) ≈ vec(queen) - vec(woman)
  
  这个方向差 = "男→女" 的语义变换
  类似的：
  vec(Paris) - vec(France) ≈ vec(Tokyo) - vec(Japan)  （首都关系）
  vec(bigger) - vec(big) ≈ vec(smaller) - vec(small)  （比较级）
```

## 5. 例题（Worked Examples）

### 例题：训练 Word2Vec 并可视化

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 用 t-SNE 降维可视化词向量
words = list(model.wv.index_to_key[:50])
vectors = [model.wv[w] for w in words]

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(vectors)

plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.annotate(word, (coords[i, 0], coords[i, 1]))
plt.title("Word2Vec t-SNE Visualization")
plt.show()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 在中文语料上训练 Word2Vec，测试"国王 - 男人 + 女人 ≈ 女王"等类比关系。

**练习 2：** 对比 CBOW 和 Skip-gram 在相同语料上的训练速度和词相似度效果。

### 进阶题

**练习 3：** 从零实现 Skip-gram with Negative Sampling（不使用 Gensim）。

**练习 4：** 用 Word2Vec 作为特征输入到 LSTM 做文本分类，对比随机初始化 Embedding 的效果差异。
