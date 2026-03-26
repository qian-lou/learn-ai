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
df["name"] = df["name"].astype("category")  # 节省内存

# ============================================================
# 5. 字符串清洗 / String cleaning
# ============================================================
df["name"] = df["name"].str.strip()         # 去空格
df["name"] = df["name"].str.lower()         # 转小写
# df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)
```

## 4. 详细推理（Deep Dive）

```
LLM 训练数据清洗要点:
  1. 去重（精确去重 + 模糊去重 MinHash）
  2. 过滤低质量（过短/过长/乱码/广告）
  3. PII 脱敏（个人信息移除）
  4. 格式统一（统一换行符/编码）
```

## 5-6. 例题/习题

**练习 1：** 清洗一个含缺失值、重复值、异常值的数据集。

**练习 2：** 实现一个数据质量报告函数，输出每列的缺失率、唯一值数、类型。
