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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# ============================================================
# 0. 构造演示数据 / Build demo data（使本片段可独立运行）
# ============================================================
X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df = pd.DataFrame({'category': ['a', 'b', 'a', 'c']})  # 类别特征示例

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

## 5. 例题（Worked Examples）

### 例题 1：文本及类别特征的离散化和特征选择 / Encoding and Feature Selection

本例题在含有空值、类别字段和长尾数值的数据集上，组合使用独热编码 (One-Hot) 与方差阈值过滤特征。

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# 1. 制造类别与数值混合数据 / Construct dataframe
data = pd.DataFrame({
    'city': ['Beijing', 'Shanghai', 'Beijing', 'Guangzhou'],
    'level': ['High', 'Low', 'High', 'High'],
    'const_feat': [1.0, 1.0, 1.0, 1.0]  # 无信息量常量特征 / Low variance feature
})

# 2. 执行 OneHot 编码 / One-Hot Encoding
# Time: O(N * C), Space: O(N * C_encoded)
encoder = OneHotEncoder(sparse_output=False)
encoded_feat = encoder.fit_transform(data[['city', 'level']])
encoded_df = pd.DataFrame(encoded_feat, columns=encoder.get_feature_names_out())

# 3. 合并特征并利用 VarianceThreshold 剔除零方差特征 / Feature Selection
all_feats = pd.concat([encoded_df, data[['const_feat']]], axis=1)
# Time: O(N * D), Space: O(N * D)
selector = VarianceThreshold(threshold=0.0)
selected_feats = selector.fit_transform(all_feats)

print(f"编码合并后特征数 / Raw feature count: {all_feats.shape[1]}")
print(f"方差过滤后特征数 / Selected feature count: {selected_feats.shape[1]}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在预处理类别特征时，`OneHotEncoder`（独热编码）与 `LabelEncoder`（标签编码）最适合什么场景？
*参考答案*：
- **OneHotEncoder**：适用于无大小顺序关系的离散类别特征（例如城市、职业）。
- **LabelEncoder**：主要用于一维的目标标签 `y` 的数值映射；如果是输入特征，若有大小等级顺序（例如学历：高中 1/大学 2），建议使用 `OrdinalEncoder`。

### 进阶题
**练习 2**：在回归问题中，目标变量呈长尾幂律分布（非正态分布），这会导致最小二乘法效果变差。请问该怎么对目标变量进行特征处理？写出 Python 处理代码。
*参考答案*：
一般使用对数变换 $y_{new} = \ln(y + 1)$ 将其转化为偏度较小的分布，在预测阶段再进行逆指数变换还原 $y = e^{y_{new}} - 1$。
```python
import numpy as np
# 特征转换 / Forward transform
y_transformed = np.log1p(y)
# 推理还原 / Inverse transform
y_original = np.expm1(y_transformed)
```