# 文本表示 / Text Representation

## 1. 背景（Background）

> **为什么要学这个？**
>
> 计算机无法直接理解文本，必须将文本转换为**数值向量**才能进行计算。文本表示的演进路线——One-Hot → BoW → TF-IDF → Word2Vec → BERT——反映了 NLP 领域对"语义理解"的不断深入。
>
> 对于 Java 工程师来说，文本表示就像是**对象序列化**——将一个抽象的"文本对象"转换为可以存储和计算的数值格式。不同的表示方法就像不同的序列化协议（JSON vs Protobuf vs Avro），各有优劣。
>
> **在整个体系中的位置：** 文本表示是连接原始文本和模型输入的桥梁。传统方法（TF-IDF）用于 baseline，深度学习方法（Embedding/BERT）用于高精度任务。

## 2. 知识点（Key Concepts）

| 表示方法 | 维度 | 语义信息 | 稀疏/稠密 | 时代 |
|----------|------|----------|-----------|------|
| One-Hot | V（词表大小）| ❌ 无 | 极度稀疏 | ~2000 |
| BoW (词袋) | V | ❌ 词频 | 稀疏 | ~2005 |
| TF-IDF | V | 部分（词重要性）| 稀疏 | ~2005 |
| Word2Vec | 100-300 | ✅ 语义 | 稠密 | 2013 |
| BERT Embedding | 768 | ✅ 上下文语义 | 稠密 | 2018 |

## 3. 内容（Content）

### 3.1 One-Hot 编码

```python
import numpy as np

# ============================================================
# One-Hot：最简单但最浪费的表示
# One-Hot: Simplest but most wasteful representation
# ============================================================
vocab = {"I": 0, "love": 1, "NLP": 2, "and": 3, "AI": 4}

# "love" 的 One-Hot 表示
one_hot = np.zeros(len(vocab))
one_hot[vocab["love"]] = 1
print(f"love = {one_hot}")  # [0, 1, 0, 0, 0]

# 问题 / Problems:
# 1. 维度 = 词表大小（GPT: 100K 维）→ 极度浪费
# 2. 所有词之间距离相等 → "cat" 和 "dog" 不比 "cat" 和 "table" 更近
# 3. 无法捕捉语义关系
```

### 3.2 词袋模型（Bag of Words）与 TF-IDF

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = [
    "I love machine learning",
    "Machine learning is great",
    "I love deep learning and NLP",
]

# ============================================================
# 词袋模型 / Bag of Words
# ============================================================
bow = CountVectorizer()
X_bow = bow.fit_transform(texts)
print(f"词表: {bow.get_feature_names_out()}")
print(f"BoW 矩阵:\n{X_bow.toarray()}")
# 只记录词频，忽略词序 → "dog bites man" ≡ "man bites dog"

# ============================================================
# TF-IDF / Term Frequency - Inverse Document Frequency
# ============================================================
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(texts)
print(f"\nTF-IDF 矩阵:\n{X_tfidf.toarray().round(2)}")

# TF-IDF 原理：
# TF(t,d)  = 词 t 在文档 d 中的频率
# IDF(t)   = log(总文档数 / 包含词 t 的文档数)
# TF-IDF   = TF × IDF
# → 在该文档中频繁出现但在其他文档中不常见的词，得分更高
```

### 3.3 TF-IDF 文本分类 Baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# TF-IDF + 逻辑回归 = 强大的文本分类 baseline
# TF-IDF + Logistic Regression = Strong text classification baseline
# ============================================================

# 模拟数据 / Mock data
texts = [
    "great movie, loved it", "terrible film, waste of time",
    "amazing performance", "boring and slow",
    "best movie ever", "worst experience",
] * 50
labels = [1, 0, 1, 0, 1, 0] * 50

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# TF-IDF 特征提取
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 训练逻辑回归
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# 评估
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# 在很多任务上，TF-IDF + LR 的表现出乎意料地好
# 是大模型方案的重要 baseline
```

### 3.4 从稀疏到稠密的转变

```
文本表示的核心转变：

稀疏表示（One-Hot/BoW/TF-IDF）:
  "cat"  = [0, 0, 1, 0, 0, ..., 0]  (维度 = 词表大小，几万维)
  "dog"  = [0, 0, 0, 1, 0, ..., 0]
  cos_sim(cat, dog) = 0  ← 无法区分语义

稠密表示（Word2Vec/BERT）:
  "cat"  = [0.2, -0.1, 0.8, ..., 0.3]  (维度 = 256~768)
  "dog"  = [0.3, -0.2, 0.7, ..., 0.4]
  cos_sim(cat, dog) = 0.85  ← 语义相似！

这个转变是 NLP 从"基于规则"到"基于学习"的核心标志
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 TF-IDF 仍然有价值？

```
优势：
  1. 无需训练模型，直接计算
  2. 可解释性强（哪些词重要一目了然）
  3. 在小数据集上表现不错
  4. 是所有深度学习方案的 baseline

局限：
  1. 忽略词序（"not good" ≈ "good not"）
  2. 无法处理同义词（"happy" ≠ "joyful"）
  3. 高维稀疏，计算效率低
  4. 需要大量文本才能计算可靠的 IDF
```

## 5. 例题（Worked Examples）

### 例题：对比不同表示方法的分类效果

```python
# 对比 BoW、TF-IDF、TF-IDF+bigram 在同一任务上的 F1 分数
from sklearn.pipeline import Pipeline

pipelines = {
    "BoW": Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=1000))]),
    "TF-IDF": Pipeline([("vec", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))]),
    "TF-IDF+Bigram": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f"{name:20s}: Accuracy = {score:.4f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 TF-IDF + 逻辑回归在 20 Newsgroups 数据集（`sklearn.datasets.fetch_20newsgroups`）上做多分类。

**练习 2：** 解释为什么 n-gram（bigram、trigram）能部分解决 BoW 忽略词序的问题。

### 进阶题

**练习 3：** 对比 TF-IDF + LR 和 BERT 在 IMDB 情感分类上的效果差异，分析各自的优劣势。

**练习 4：** 实现一个简单的文档搜索引擎：输入查询语句，用 TF-IDF 余弦相似度返回最相关的文档。
