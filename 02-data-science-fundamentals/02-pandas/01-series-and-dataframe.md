# Series 与 DataFrame
# Series and DataFrame

## 1. 背景（Background）

> Pandas 是 Python 的"SQL 引擎"——提供类似数据库表的 DataFrame 结构。作为 Java 工程师，可以把 DataFrame 想象为一个 `List<Map<String, Object>>`，但带有强大的查询和变换能力。NLP 数据集处理（加载、清洗、统计）几乎都用 Pandas。

## 2-3. 知识点与内容

```python
import pandas as pd
import numpy as np

# Series：一维带标签数组（类似 Map<Index, Value>）
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s["a"])  # 10

# DataFrame：二维表（类似 SQL 表 / List<Map>）
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [95.5, 87.3, 92.1],
})

# 基础操作 / Basic operations
df.head()              # 前 5 行
df.describe()          # 统计摘要
df.info()              # 列类型和空值信息
df["name"]             # 选择列（返回 Series）
df[df["age"] > 28]     # 条件过滤（类似 SQL WHERE）
df.sort_values("score", ascending=False)  # 排序
df["grade"] = df["score"].apply(lambda x: "A" if x >= 90 else "B")  # 新增列
```

## 4-6. 推理/例题/习题

**核心类比：** `df[df["age"] > 28]` ≈ `SELECT * FROM df WHERE age > 28`

**练习：** 加载一个 CSV 文件，筛选年龄 > 20 的记录，按分数降序排列，输出前 10 条。
