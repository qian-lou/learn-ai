# 分组与聚合
# GroupBy and Aggregation

## 1. 背景（Background）

> **为什么要学这个？**
>
> GroupBy 是 Pandas 最强大的功能之一，等价于 SQL 的 `GROUP BY`。在模型评估中，按类别/数据集/模型分组统计指标是常规操作。

## 2. 知识点（Key Concepts）

| Pandas | SQL | 说明 |
|--------|-----|------|
| `groupby("col")` | `GROUP BY col` | 分组 |
| `.agg()` | 聚合函数 | 多种聚合 |
| `.transform()` | 窗口函数 | 组内变换 |
| `.apply()` | - | 自定义操作 |

## 3. 内容（Content）

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "model": ["BERT", "GPT", "BERT", "GPT", "T5", "T5"],
    "dataset": ["IMDB", "IMDB", "SST", "SST", "IMDB", "SST"],
    "accuracy": [0.92, 0.89, 0.91, 0.93, 0.90, 0.88],
    "latency_ms": [15, 45, 14, 42, 20, 18],
})

# ============================================================
# 基础分组 / Basic groupby
# ============================================================
# 按模型分组，计算平均准确率
print(df.groupby("model")["accuracy"].mean())
# BERT    0.915
# GPT     0.910
# T5      0.890

# ============================================================
# 多重聚合 / Multiple aggregations
# ============================================================
result = df.groupby("model").agg(
    avg_accuracy=("accuracy", "mean"),
    max_accuracy=("accuracy", "max"),
    avg_latency=("latency_ms", "mean"),
    count=("accuracy", "count"),
)
print(result)

# ============================================================
# 多列分组 / Multi-column groupby
# ============================================================
pivot = df.groupby(["model", "dataset"])["accuracy"].mean().unstack()
print(pivot)  # 交叉表格

# ============================================================
# Transform（组内变换，保持原始形状）
# ============================================================
df["accuracy_zscore"] = df.groupby("model")["accuracy"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# ============================================================
# 自定义聚合 / Custom aggregation
# ============================================================
def accuracy_range(series):
    return series.max() - series.min()

print(df.groupby("model")["accuracy"].agg(accuracy_range))
```

## 4. 详细推理（Deep Dive）

```
GroupBy 三步流程（Split-Apply-Combine）:
  1. Split: 按列值分组
  2. Apply: 对每组应用函数
  3. Combine: 合并结果

等价 SQL:
  SELECT model, AVG(accuracy), MAX(accuracy)
  FROM results
  GROUP BY model
  HAVING AVG(accuracy) > 0.9
  ORDER BY AVG(accuracy) DESC

→ Pandas:
  df.groupby("model").agg(...)
    .query("avg_accuracy > 0.9")
    .sort_values("avg_accuracy", ascending=False)
```

## 5. 例题（Worked Examples）

### 例题 1：按组统计员工多维度指标 / Groupby and Multi-aggregation

本例模拟 SQL 的 GROUP BY 聚合查询，统计各部门员工的人数、平均薪资以及最高月薪。

```python
import pandas as pd

# 创建部门薪资表 / Create employee data
# Time: O(R * C), Space: O(R * C)
data = {
    'name': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
    'dept': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT'],
    'salary': [8000, 6000, 9000, 7500, 6500, 12000]
}
df = pd.DataFrame(data)

# 进行分组多指标汇总 / Perform groupby multi-aggregation
# Time: O(R), Space: O(G * C) - G 为分组个数 / G is the number of groups.
summary = df.groupby('dept')['salary'].agg(
    count='count',
    avg_salary='mean',
    max_salary='max'
).reset_index()

print("部门汇总分析报表 / Department Summary Report:")
print(summary)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：求每个班级的平均成绩，并输出为 Series 格式。
*参考答案*：
```python
# Time: O(N), Space: O(G)
# df.groupby('class')['score'].mean()
```

### 进阶题
**练习 2**：给定一个用户流水账单，包含用户ID、交易日期和交易金额。找出每个用户累计交易总额，并计算每个用户单笔交易最大值，同时筛选出累计交易总额超过 1000 元的用户。
*参考答案*：
```python
import pandas as pd
df = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3],
    'amount': [600, 500, 1500, 100, 300]
})
# Time: O(N), Space: O(G)
grouped = df.groupby('user_id')['amount'].agg(total='sum', max_val='max')
filtered_users = grouped[grouped['total'] > 1000]
print(filtered_users)
```