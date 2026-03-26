# Series 与 DataFrame
# Series and DataFrame

## 1. 背景（Background）

> **为什么要学这个？**
>
> Pandas 是 Python 数据分析的**核心库**，`DataFrame` 类似数据库表或 Excel 表格。在大模型开发中，数据清洗、特征工程、评估结果分析都离不开 Pandas。
>
> 对于 Java 工程师来说，DataFrame 相当于 **JDBC ResultSet + Stream API**——结构化数据的查询和变换工具。

## 2. 知识点（Key Concepts）

| Pandas | Java 类比 | 说明 |
|--------|----------|------|
| Series | List/Column | 一维带标签数组 |
| DataFrame | ResultSet/Table | 二维表格 |
| Index | Primary Key | 行标签 |
| dtype | Column Type | 数据类型 |

## 3. 内容（Content）

```python
import pandas as pd
import numpy as np

# ============================================================
# Series（一维带标签数组）
# ============================================================
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="scores")
print(s["a"])     # 10（按标签访问）
print(s[0])       # 10（按位置访问）
print(s.mean())   # 20.0

# ============================================================
# DataFrame（二维表格）
# ============================================================
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [90.5, 85.0, 92.3],
})
print(df)
#       name  age  score
# 0    Alice   25   90.5
# 1      Bob   30   85.0
# 2  Charlie   35   92.3

# 列选择
print(df["name"])           # Series
print(df[["name", "age"]])  # DataFrame

# 行选择
print(df.loc[0])            # 按标签
print(df.iloc[0:2])         # 按位置切片

# 条件过滤（类似 SQL WHERE）
adults = df[df["age"] >= 30]
top = df[df["score"] > 90]

# 添加列
df["grade"] = df["score"].apply(lambda x: "A" if x >= 90 else "B")

# ============================================================
# 常用属性 / Common attributes
# ============================================================
print(df.shape)     # (3, 4)
print(df.dtypes)    # 每列类型
print(df.describe()) # 统计摘要
print(df.info())     # 内存和类型信息
```

## 4. 详细推理（Deep Dive）

```
Pandas vs SQL:
  df[df["age"] > 30]        →  WHERE age > 30
  df[["name", "age"]]       →  SELECT name, age
  df.sort_values("score")   →  ORDER BY score
  df.head(10)               →  LIMIT 10
  df.groupby("grade").mean() →  GROUP BY grade

内存优化:
  df["category"] = df["category"].astype("category")
  → 字符串列转 category 类型可节省 90% 内存
```

## 5-6. 例题/习题

**练习 1：** 创建一个包含 1000 个样本的 DataFrame，计算各列的统计信息。

**练习 2：** 用条件过滤和排序模拟 SQL 查询：`SELECT * FROM df WHERE score > 80 ORDER BY age DESC LIMIT 10`。
