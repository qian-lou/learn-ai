# 聚类算法 / Clustering Algorithms

## 1. 背景（Background）
> 聚类是无监督学习的代表。NLP 中用于文本聚类、主题发现，向量数据库的索引也用到聚类思想（如 IVF 索引）。

## 2-3. 知识点与内容
```python
from sklearn.cluster import KMeans, DBSCAN

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# DBSCAN 基于密度，不需要预设 K
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

## 4. 详细推理
- K-Means: 反复计算中心 → 重新分配 → 计算中心，直到收敛。Time: O(N*K*I*D)
- 肘部法则（Elbow Method）确定最优 K
- DBSCAN 可发现任意形状的簇，自动检测噪声点

## 5-6. 例题/习题
**练习：** 用 K-Means 对词嵌入向量聚类，发现语义相似的词组。
