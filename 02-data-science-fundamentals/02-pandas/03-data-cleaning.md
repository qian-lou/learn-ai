# 数据清洗
# Data Cleaning

## 1. 背景（Background）

> **为什么要学这个？**
>
> 数据科学 80% 的时间花在数据清洗上。缺失值、重复值、异常值、格式不一致——这些问题不处理，模型训练就是 "Garbage In, Garbage Out"。大模型的训练数据质量直接决定模型质量。

## 2. 知识点（Key Concepts）

| 问题 | 方法 | API |
|------|------|-----|
| 缺失值 | 填充/删除 | `fillna()` / `dropna()` |
| 重复值 | 去重 | `drop_duplicates()` |
| 类型转换 | 转换 | `astype()` |
| 异常值 | 裁剪 | `clip()` |
| 字符串清洗 | 正则 | `.str.replace()` |

## 3. 内容（Content）

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "Alice", "Eve"],
    "age": [25, None, 35, 25, 28],
    "score": [90.5, 85.0, 150.0, 90.5, -10.0],  # 150 和 -10 是异常值
})

# ============================================================
# 1. 缺失值处理 / Missing value handling
# ============================================================
print(df.isnull().sum())  # 每列缺失数

# 删除含缺失值的行
df_clean = df.dropna()

# 填充缺失值
df["age"] = df["age"].fillna(df["age"].median())  # 中位数填充
df["name"] = df["name"].fillna("Unknown")

# ============================================================
# 2. 重复值处理 / Duplicate handling
# ============================================================
print(df.duplicated().sum())  # 重复行数
df = df.drop_duplicates(subset=["name", "age"], keep="first")

# ============================================================
# 3. 异常值处理 / Outlier handling
# ============================================================
# Clip: 裁剪到合理范围
df["score"] = df["score"].clip(lower=0, upper=100)

# IQR 方法
Q1, Q3 = df["score"].quantile([0.25, 0.75])
IQR = Q3 - Q1
mask = (df["score"] >= Q1 - 1.5 * IQR) & (df["score"] <= Q3 + 1.5 * IQR)
df_no_outliers = df[mask]

# ============================================================
# 4. 类型转换 / Type conversion
# ============================================================
df["age"] = df["age"].astype(int)

# ============================================================
# 5. 字符串清洗 / String cleaning
# ============================================================
df["name"] = df["name"].str.strip()         # 去空格
df["name"] = df["name"].str.lower()         # 转小写
# df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)

# 字符串规整完成后再转 category（.str 访问器会把 category 退回 object，故须放在最后）
df["name"] = df["name"].astype("category")  # 节省内存
```

## 4. 详细推理（Deep Dive）

```
LLM 训练数据清洗要点:
  1. 去重（精确去重 + 模糊去重 MinHash）
  2. 过滤低质量（过短/过长/乱码/广告）
  3. PII 脱敏（个人信息移除）
  4. 格式统一（统一换行符/编码）
```

## 5. 例题（Worked Examples）

### 例题 1：数据清洗与异常值百分位裁剪 / Data cleaning and outlier trimming using percentiles

在将特征送入机器学习模型前，清洗空值、重复值并剪裁异常偏离值是非常关键的工程步骤。

```python
import pandas as pd
import numpy as np

# 构造脏数据 / Construct raw dirty data
data = {
    'user': ['Alice', 'Bob', 'Alice', 'Charlie', 'David'],
    'income': [5000.0, np.nan, 5000.0, 100000.0, 2000.0]  # Charlie 包含异常高收入
}
df = pd.DataFrame(data)

# 1. 过滤重复行 / Remove duplicate rows
# Time: O(N), Space: O(N)
df = df.drop_duplicates()

# 2. 填充缺失的收入（以中位数填充） / Fill missing income with median
# Time: O(N), Space: O(1)
df['income'] = df['income'].fillna(df['income'].median())

# 3. 使用盖帽法限制过大值（如取 95 分位数裁剪） / Clip extreme values at 95th percentile
# Time: O(N), Space: O(1)
p95 = df['income'].quantile(0.95)
df['income'] = df['income'].clip(upper=p95)

print("清洗并裁剪后的数据集 / Cleaned dataset:")
print(df)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：删除含有缺失值的整行数据。
*参考答案*：
```python
# Time: O(N), Space: O(N)
# df_cleaned = df.dropna()
```

### 进阶题
**练习 2**：假设有一个商品价格 DataFrame，有些商品价格是空值，你想根据它们所属的“品类”的平均价格分别填充这些商品的缺失价格。
*参考答案*：
```python
import pandas as pd
import numpy as np
df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'price': [10.0, np.nan, np.nan, 20.0]})
# Time: O(N), Space: O(N)
df['price'] = df.groupby('category')['price'].transform(lambda x: x.fillna(x.mean()))
print(df)
```