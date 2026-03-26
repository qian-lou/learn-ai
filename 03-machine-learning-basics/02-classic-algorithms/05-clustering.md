# 聚类算法
# Clustering Algorithms

## 1. 背景（Background）

> **为什么要学这个？**
>
> 聚类是无监督学习的代表。在大模型开发中，聚类用于数据去重（MinHash + 聚类）、数据多样性分析、Embedding 可视化（t-SNE + KMeans）。

## 2. 知识点（Key Concepts）

| 算法 | 特点 |
|------|------|
| KMeans | 快速，需指定 K |
| DBSCAN | 自动确定簇数，处理噪声 |
| 层次聚类 | 树状结构，无需指定 K |

## 3. 内容（Content）

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 生成聚类数据
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)

# ============================================================
# KMeans
# ============================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
print(f"聚类中心: {kmeans.cluster_centers_.shape}")

# 肘部法则选择 K
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
# 画图找"肘部"拐点

# ============================================================
# DBSCAN（基于密度）
# ============================================================
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"DBSCAN 找到 {n_clusters} 个簇")
```

## 4. 详细推理（Deep Dive）

```
聚类在 LLM 中的应用:
  - 训练数据去重: 对 embedding 聚类，同簇视为重复
  - 数据配比: 不同簇代表不同领域，确保训练数据多样
  - 评估: 对模型输出 embedding 可视化（t-SNE + 颜色标注）
```

## 5-6. 例题/习题

**练习 1：** 用 KMeans 对 Embedding 向量聚类，用 t-SNE 可视化。

**练习 2：** 对比 KMeans 和 DBSCAN 在不同形状数据上的效果。
