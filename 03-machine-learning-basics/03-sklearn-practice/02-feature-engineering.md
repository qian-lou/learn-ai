# 特征工程
# Feature Engineering

## 1. 背景（Background）

> **为什么要学这个？**
>
> 特征工程是传统 ML 最重要的技能——"数据和特征决定了 ML 的上限，算法只是逼近这个上限"。虽然深度学习（尤其是 LLM）减少了手动特征工程的需求，但理解特征工程的思想对数据预处理和理解模型仍然重要。

## 2. 知识点（Key Concepts）

| 技术 | 方法 | 适用场景 |
|------|------|---------|
| 标准化 | StandardScaler | 连续特征 |
| 编码 | OneHotEncoder | 类别特征 |
| 降维 | PCA | 高维数据 |
| 特征选择 | SelectKBest | 去除冗余 |

## 3. 内容（Content）

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# ============================================================
# 1. 数值特征处理 / Numeric features
# ============================================================
scaler = StandardScaler()  # z = (x - μ) / σ
X_scaled = scaler.fit_transform(X_train)
# ⚠️ 用 train 的统计量 transform test
X_test_scaled = scaler.transform(X_test)

# MinMax 归一化 [0, 1]
normalizer = MinMaxScaler()
X_norm = normalizer.fit_transform(X_train)

# ============================================================
# 2. 类别特征编码 / Categorical encoding
# ============================================================
# One-Hot
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = ohe.fit_transform(df[['category']])

# Label Encoding (树模型可用)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['category'])

# ============================================================
# 3. 降维 / Dimensionality reduction
# ============================================================
pca = PCA(n_components=0.95)  # 保留 95% 方差
X_reduced = pca.fit_transform(X_scaled)
print(f"降维: {X_scaled.shape[1]} → {X_reduced.shape[1]} 维")

# ============================================================
# 4. 特征选择 / Feature selection
# ============================================================
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
```

## 4. 详细推理（Deep Dive）

```
深度学习时代的特征工程:
  传统 ML: 手动特征工程是核心（80% 时间）
  深度学习: 模型自动学习特征（End-to-End）
  LLM: 文本直接输入，几乎不需要特征工程
  
  但数据清洗和预处理仍然重要！
```

## 5-6. 例题/习题

**练习 1：** 用 ColumnTransformer 对混合类型数据做预处理。

**练习 2：** 用 PCA 将 MNIST 从 784 维降到 50 维，对比分类准确率。
