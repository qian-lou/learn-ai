# 文本表示 / Text Representation

## 1. 背景（Background）
> 从 One-Hot → BoW → TF-IDF → Word2Vec → BERT，文本表示不断进化。

## 2-3. 知识点与内容
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
texts = ["I love NLP", "NLP is great"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
# TF-IDF 衡量词在文档中的重要性
# 深度学习后被 Embedding 替代
```

## 4-6. 推理/例题/习题
**练习：** 用 TF-IDF + 逻辑回归做文本分类 baseline。
