# 合并与连接（JOIN 对比）
# Merge and Join (vs SQL JOIN)

## 1. 背景（Background）

> Pandas 的 `merge` 等价于 SQL 的 JOIN。Java 中需要手动遍历两个列表进行关联，Pandas 一行搞定。

## 2-3. 知识点与内容

```python
import pandas as pd

users = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
scores = pd.DataFrame({"user_id": [1, 2, 4], "score": [95, 87, 92]})

# INNER JOIN
result = pd.merge(users, scores, left_on="id", right_on="user_id", how="inner")

# LEFT JOIN
result = pd.merge(users, scores, left_on="id", right_on="user_id", how="left")

# 纵向拼接 / Vertical concatenation (UNION ALL)
df1 = pd.DataFrame({"text": ["hello"], "label": [1]})
df2 = pd.DataFrame({"text": ["world"], "label": [0]})
combined = pd.concat([df1, df2], ignore_index=True)
```

## 4-6. 推理/例题/习题

**练习：** 合并训练集和测试集的元数据，计算各分类在训练/测试中的分布差异。
