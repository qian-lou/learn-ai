# 分组与聚合（SQL 对比）
# GroupBy and Aggregation (vs SQL)

## 1. 背景（Background）

> Pandas 的 `groupby` 等价于 SQL 的 `GROUP BY`。Java 工程师可以类比为 Stream 的 `Collectors.groupingBy`。

## 2-3. 知识点与内容

```python
import pandas as pd

df = pd.DataFrame({
    "dept": ["AI", "AI", "Backend", "Backend", "AI"],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "salary": [100, 120, 80, 90, 110],
})

# SQL: SELECT dept, AVG(salary) FROM df GROUP BY dept
result = df.groupby("dept")["salary"].mean()

# 多个聚合函数 / Multiple aggregations
# SQL: SELECT dept, COUNT(*), AVG(salary), MAX(salary) FROM df GROUP BY dept
result = df.groupby("dept")["salary"].agg(["count", "mean", "max"])

# 自定义聚合 / Custom aggregation
result = df.groupby("dept").agg(
    headcount=("name", "count"),
    avg_salary=("salary", "mean"),
    top_earner=("salary", "max"),
)

# 转换（对每个组内应用变换）/ Transform
df["salary_rank"] = df.groupby("dept")["salary"].rank(ascending=False)
```

## 4-6. 推理/例题/习题

**练习：** 给定一个 NLP 数据集（含 text 和 label 列），统计每个 label 的样本数和平均文本长度。
