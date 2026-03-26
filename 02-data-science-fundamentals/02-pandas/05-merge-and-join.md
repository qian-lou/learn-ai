# 合并与连接
# Merge and Join

## 1. 背景（Background）

> **为什么要学这个？**
>
> 合并（Merge/Join）是关联多个数据表的核心操作，等价于 SQL 的 JOIN。在多数据源分析中（如模型指标表 + 模型配置表）经常使用。

## 2. 知识点（Key Concepts）

| Pandas | SQL | 说明 |
|--------|-----|------|
| `merge` | `JOIN` | 按键合并 |
| `concat` | `UNION ALL` | 纵向拼接 |
| `join` | `JOIN ON index` | 按索引合并 |

## 3. 内容（Content）

```python
import pandas as pd

models = pd.DataFrame({
    "model_id": [1, 2, 3],
    "name": ["BERT", "GPT-2", "T5"],
    "params_B": [0.11, 1.5, 0.22],
})

scores = pd.DataFrame({
    "model_id": [1, 2, 1, 3, 4],
    "dataset": ["IMDB", "IMDB", "SST", "SST", "IMDB"],
    "accuracy": [0.92, 0.89, 0.91, 0.88, 0.85],
})

# ============================================================
# Merge（SQL JOIN）
# ============================================================
# Inner Join（默认）
inner = pd.merge(models, scores, on="model_id")

# Left Join（保留左表所有行）
left = pd.merge(models, scores, on="model_id", how="left")

# Outer Join（保留所有行）
outer = pd.merge(models, scores, on="model_id", how="outer")

# 多键合并
# pd.merge(df1, df2, on=["model_id", "dataset"])

# ============================================================
# Concat（纵向/横向拼接）
# ============================================================
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# 纵向拼接（相当于 SQL UNION ALL）
vertical = pd.concat([df1, df2], ignore_index=True)

# 横向拼接
horizontal = pd.concat([df1, df2], axis=1)
```

## 4. 详细推理（Deep Dive）

```
Merge 类型速查:
  inner: 只保留两表都有的键（交集）
  left:  保留左表所有行，右表无匹配则 NaN
  right: 保留右表所有行
  outer: 保留所有行（并集）

性能注意:
  大数据合并时先 sort → merge（更快）
  或使用 join 按索引合并（避免 hash）
```

## 5-6. 例题/习题

**练习 1：** 合并模型信息表和评估结果表，生成完整的评估报告。

**练习 2：** 将多个 CSV 文件纵向拼接为一个 DataFrame。
