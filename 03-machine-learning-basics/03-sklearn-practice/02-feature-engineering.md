# 特征工程 / Feature Engineering

## 1. 背景（Background）
> "特征工程决定模型的上限，算法只是逼近这个上限。" TF-IDF 文本特征提取是 NLP 预处理的基础。

## 2-3. 知识点与内容
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# 数值特征标准化 / Numeric feature scaling
scaler = StandardScaler()  # (x - mean) / std
X_scaled = scaler.fit_transform(X)

# 文本特征提取 / Text feature extraction
tfidf = TfidfVectorizer(max_features=5000)
text_features = tfidf.fit_transform(texts)  # 稀疏矩阵
```

## 4. 详细推理
- TF-IDF = 词频(TF) × 逆文档频率(IDF)，衡量词在文档中的重要性
- 深度学习时代用 Embedding 替代了 TF-IDF，但后者在小数据场景仍有价值

## 5-6. 例题/习题
**练习：** 对比 BoW、TF-IDF、Word2Vec 在文本分类任务上的效果。
