# Word2Vec（CBOW / Skip-gram）

## 1. 背景（Background）

> **为什么要学这个？**
>
> Word2Vec（2013，Google）开创了**用神经网络学习词向量**的先河，让 NLP 从稀疏表示（One-Hot/TF-IDF）迈入了稠密向量时代。它的核心洞察——**"一个词的含义由它的上下文决定"**（分布式假设）——至今仍是所有语言模型的基础。
>
> Transformer 中的 `nn.Embedding` 层，本质上就是 Word2Vec 思想的延续。理解 Word2Vec 对理解大模型的嵌入层至关重要。
>
> 对于 Java 工程师来说，嵌入表就像一个 `HashMap<String, float[]>`——按词查一个稠密向量；区别是这些向量是训练学出来的，语义相近的词向量在空间中也相近（`king - man + woman ≈ queen`）。
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
  提速幅度 ≈ 词表 V / 负采样数 K（V 上万、K 取 5-20，故可达百倍量级）
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

*参考答案*：

中文需要先分词（用 jieba），把每句切成词列表，再喂给 Gensim。

```python
import jieba
from gensim.models import Word2Vec

# 1. 分词：每个句子 -> 词列表 / Tokenize each sentence into words
raw_sentences = [...]                                   # 你的中文语料
sentences = [list(jieba.cut(s)) for s in raw_sentences]

# 2. 训练 Skip-gram / Train Skip-gram
model = Word2Vec(sentences, vector_size=200, window=5,
                 min_count=5, sg=1, epochs=10, workers=4)

# 3. 词类比：国王 - 男人 + 女人 ≈ ?
result = model.wv.most_similar(positive=["国王", "女人"],
                               negative=["男人"], topn=3)
print(result)        # 期望 [('女王', 0.7x), ...]
```

要点：(1) **中文必须先分词**，词向量的"词"取决于分词结果；(2) 语料要足够大、且包含相关词的足够多上下文，类比关系才学得出来——玩具语料很可能得不到 "女王"；(3) `model.wv.most_similar(positive=[...], negative=[...])` 内部正是在做 `vec(国王) - vec(男人) + vec(女人)` 后找最近邻。

**练习 2：** 对比 CBOW 和 Skip-gram 在相同语料上的训练速度和词相似度效果。

*参考答案*：

只改 `sg` 参数（0=CBOW, 1=Skip-gram），其余完全相同，再计时和对比相似度。

```python
import time
from gensim.models import Word2Vec

def train(sg):
    t0 = time.time()
    m = Word2Vec(sentences, vector_size=100, window=5,
                 min_count=5, sg=sg, epochs=10, workers=4)
    return m, time.time() - t0

cbow, t_cbow = train(sg=0)
skip, t_skip = train(sg=1)
print(f"CBOW: {t_cbow:.1f}s  Skip-gram: {t_skip:.1f}s")
print("CBOW 相似词:", cbow.wv.most_similar("learning", topn=5))
print("SG   相似词:", skip.wv.most_similar("learning", topn=5))
```

对比结论（与本文知识点表一致）：
- **速度**：CBOW **更快**。CBOW 用上下文词的平均去预测 1 个中心词，每个窗口只算一次；Skip-gram 用中心词去分别预测每个上下文词，一个窗口产生多个训练对，计算量更大。
- **效果**：Skip-gram 对**低频词/罕见词**更好（每个词都被反复当作中心词训练，得到更多更新），词相似度和词类比常更优；CBOW 对**高频词**效果不错且训练快。
- 选择：语料大、追速度 → CBOW；语料相对小、在意稀有词质量 → Skip-gram（这也是原论文和实践的普遍建议）。

### 进阶题

**练习 3：** 从零实现 Skip-gram with Negative Sampling（不使用 Gensim）。

*参考答案*：

核心：两套 Embedding（中心词 `in_emb`、上下文词 `out_emb`），正样本标签 1、K 个负样本标签 0，用 BCE 做二分类。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    """Skip-gram with Negative Sampling / 负采样跳字模型."""
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)   # 中心词向量
        self.out_emb = nn.Embedding(vocab_size, embed_dim)  # 上下文词向量

    def forward(self, center, context, neg):
        """
        Args:
            center: 中心词 id. Shape: [B]
            context: 正样本上下文 id. Shape: [B]
            neg: 负样本 id. Shape: [B, K]
        Returns:
            标量损失 / scalar loss.
        """
        v = self.in_emb(center)                 # Shape: [B, D]
        u_pos = self.out_emb(context)           # Shape: [B, D]
        u_neg = self.out_emb(neg)               # Shape: [B, K, D]

        # 正样本得分 -> 希望 sigmoid 接近 1
        pos_score = torch.sum(v * u_pos, dim=1)             # Shape: [B]
        pos_loss = F.logsigmoid(pos_score)
        # 负样本得分 -> 希望 sigmoid(-score) 接近 1（即 score 越负越好）
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)     # Shape: [B]
        return -(pos_loss + neg_loss).mean()    # 最大化对数似然 = 最小化其负值
```

要点：(1) 负样本按词频的 3/4 次方分布采样（出现越多越可能被采，但被 3/4 次方压平，让稀有词也有机会）；(2) 损失用 `logsigmoid` 而非完整 softmax，把 V 类多分类降为 (1+K) 个二分类，复杂度从 O(V) 降到 O(K)，这正是负采样加速 100x 的来源；(3) 训练完一般取 `in_emb` 作为词向量。

**练习 4：** 用 Word2Vec 作为特征输入到 LSTM 做文本分类，对比随机初始化 Embedding 的效果差异。

*参考答案*：

把训练好的 Word2Vec 词向量按词表顺序拼成权重矩阵，用 `from_pretrained` 灌进 `nn.Embedding`。

```python
import torch
import torch.nn as nn
import numpy as np

def build_embedding(w2v_model, word2idx, embed_dim):
    """用 Word2Vec 构造预训练 Embedding 权重 / Build pretrained weights."""
    weights = np.random.normal(0, 0.1, (len(word2idx), embed_dim))
    for word, idx in word2idx.items():
        if word in w2v_model.wv:                       # OOV 保留随机初始化
            weights[idx] = w2v_model.wv[word]
    return torch.FloatTensor(weights)                  # Shape: [V, embed_dim]

class TextClassifier(nn.Module):
    def __init__(self, pretrained, hidden, n_cls, freeze=False):
        super().__init__()
        # freeze=True 冻结词向量；数据少时建议先冻结
        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=freeze)
        self.lstm = nn.LSTM(pretrained.size(1), hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):                              # x: [B, T]
        emb = self.embedding(x)                        # [B, T, D]
        _, (h, _) = self.lstm(emb)
        return self.fc(h[-1])                          # [B, n_cls]
```

效果对比结论：
- **数据量小时**，Word2Vec 预训练初始化通常**明显更好**——预训练向量已编码语义，相当于迁移了大规模无标注语料的知识，模型不必从零学词义，收敛更快、泛化更好。
- **数据量很大时**，随机初始化也能学到足够好的任务专属向量，两者差距缩小。
- 实践建议：小数据先 `freeze=True` 只训上层，再视情况解冻微调（`freeze=False`）让词向量适配具体任务；OOV 词保留随机初始化并随训练更新。
