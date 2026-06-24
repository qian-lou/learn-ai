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

*参考答案*：

用三种预训练向量在同一份"人工标注相似度"数据集（如 WordSim-353、SimLex-999）上算 Spearman 相关系数：模型给出的余弦相似度排序与人工打分排序越一致，相关系数越高。

```python
import gensim.downloader as api
from scipy.stats import spearmanr

models = {
    "word2vec": api.load("word2vec-google-news-300"),
    "glove": api.load("glove-wiki-gigaword-300"),
    "fasttext": api.load("fasttext-wiki-news-subwords-300"),
}

# pairs: List[(w1, w2, human_score)]，来自 WordSim-353 等
def evaluate(kv, pairs):
    sys, gold = [], []
    for w1, w2, h in pairs:
        if w1 in kv and w2 in kv:                 # 词表内才计入
            sys.append(kv.similarity(w1, w2)); gold.append(h)
    return spearmanr(sys, gold).correlation

for name, kv in models.items():
    print(f"{name}: Spearman = {evaluate(kv, pairs):.3f}")
```

结论：三者在标准词相似度基准上**表现接近，差距通常很小**，具体高低取决于训练语料和维度，没有绝对赢家。经验上：GloVe 利用全局共现，在词类比/相似度上常与 Word2Vec 相当或略优；FastText 因子词信息在**形态丰富的语言和含罕见词的测试**上更稳，且能给 OOV 词打分（Word2Vec/GloVe 遇到 OOV 直接缺失）。用 Spearman 而非准确率，是因为相似度任务关心的是排序一致性。

**练习 2：** 测试 FastText 对 OOV 词的处理能力。

*参考答案*：

关键验证点：一个训练时**从未出现**的词，FastText 仍能由其字符 n-gram 拼出向量，且与形近词相似。

```python
from gensim.models import FastText

sentences = [["natural", "language", "processing", "is", "fun"],
             ["i", "love", "machine", "learning"]] * 100
ft = FastText(sentences, vector_size=100, window=5, min_count=1,
              sg=1, min_n=3, max_n=6)

word = "learnings"                                  # 训练集没有这个词
print(word in ft.wv.key_to_index)                   # False —— 确实是 OOV
vec = ft.wv[word]                                   # 但仍能得到向量！
print(vec.shape)                                    # (100,)
# OOV 词与其形近词相似：learnings 与 learning 子词大量重叠
print(ft.wv.similarity("learnings", "learning"))    # 相似度较高
```

原理与结论：FastText 把词表示为其字符 n-gram 向量之和（`<le`,`lea`,...,`ngs>`）。OOV 词 `learnings` 与已知词 `learning` 共享绝大多数子词，因此推断出的向量与 `learning` 高度相似。对比 Word2Vec/GloVe：它们以整词为单位，遇到 OOV 只能返回 KeyError 或 UNK，**完全无法处理**。这正是 FastText 对拼写变体、形态变化、未登录词鲁棒的根本原因。注意：若 OOV 词的所有子词都没在训练中见过，向量质量会下降。

### 进阶题

**练习 3：** 用 GloVe 预训练向量初始化 LSTM 文本分类模型，对比随机初始化的效果。

*参考答案*：

复用本文 5 节的 `load_glove_embeddings` 把 GloVe 灌进 `nn.Embedding`，再接 LSTM 分类头：

```python
import torch
import torch.nn as nn

class GloveLSTM(nn.Module):
    def __init__(self, glove_weights, hidden, n_cls, freeze=False):
        super().__init__()
        # glove_weights: [V, 100]，OOV 行为随机初始化
        self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=freeze)
        self.lstm = nn.LSTM(glove_weights.size(1), hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):                          # x: [B, T]
        _, (h, _) = self.lstm(self.embedding(x))   # h[-1]: [B, hidden]
        return self.fc(h[-1])

# 对照组：纯随机初始化
# rand_emb = nn.Embedding(vocab_size, 100)  # 其余结构相同
```

对比结论：
- 在**中小规模标注数据**上，GloVe 初始化通常**更好**——预训练向量已编码全局共现得到的语义，模型起点更高、收敛更快、对训练集未充分覆盖的词泛化更好。
- 随机初始化要从零学词义，小数据下容易过拟合或学不充分；数据量很大时差距会缩小。
- 实践：小数据先 `freeze=True`（保护预训练语义、只训上层），再视情况 `freeze=False` 微调；OOV 词用随机行初始化并随训练更新。这与 Word2Vec 初始化的结论一致。

**练习 4：** 分析 FastText 的 subword 分解：查看 "unbelievably" 被分解为哪些子词。

*参考答案*：

FastText 先给词加边界标记 `<` 和 `>`，再对 `<unbelievably>` 滑窗取长度 min_n~max_n 的所有字符 n-gram。

```python
def char_ngrams(word: str, min_n: int = 3, max_n: int = 6):
    """复现 FastText 的子词切分 / Reproduce FastText subword split.

    Time: O(len * (max_n - min_n))  Space: O(子词数)
    """
    w = f"<{word}>"                                # 加边界标记
    grams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(w) - n + 1):
            grams.append(w[i:i + n])
    return grams

for g in char_ngrams("unbelievably", 3, 6):
    print(g)
```

以 min_n=3, max_n=6 为例，`<unbelievably>`（含边界共 14 个字符）会被分解为：
- 3-gram：`<un`, `unb`, `nbe`, `bel`, `eli`, `lie`, `iev`, `eva`, `vab`, `abl`, `bly`, `ly>` …
- 4/5/6-gram：`<unb`, `unbe`, `nbel` … 直到 `<unbel`, `unbeli` …

**外加整词本身** `<unbelievably>` 也作为一个特殊 token。最终词向量 = 所有这些子词向量之和。关键观察：它包含了 `bel`/`elie`（来自 believe）、`<un`（否定前缀）、`bly`/`ly>`（副词后缀）等有意义的形态片段，所以 FastText 能让 `unbelievable`、`believe`、`believably` 这些同源词通过共享子词获得相近表示。
