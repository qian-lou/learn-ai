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

## 5-6. 例题/习题

**练习 1：** 按模型和数据集分组，统计准确率均值和标准差。

**练习 2：** 实现一个模型评估报告：按模型分组展示所有指标的均值和排名。
