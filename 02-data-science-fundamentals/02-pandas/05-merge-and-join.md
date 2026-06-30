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

## 5. 例题（Worked Examples）

### 例题 1：表合并连接（找出没有下单的用户） / Join Operations to find inactive users

在电商场景中，我们需要将用户表与订单表进行外连接，找出哪些注册用户尚未生成任何订单。

```python
import pandas as pd

# 1. 创建用户表和订单表 / Create users and orders tables
# Time: O(N), Space: O(N)
users = pd.DataFrame({
    'user_id': [1001, 1002, 1003],
    'username': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': ['O_01', 'O_02'],
    'user_id': [1001, 1002],
    'amount': [150.0, 320.0]
})

# 2. 进行左连接 / Perform left join
# Time: O(N + M), Space: O(N + M)
merged = pd.merge(users, orders, on='user_id', how='left')

# 3. 筛选订单字段为空的用户 / Filter users without orders
inactive = merged[merged['order_id'].isna()]

print("未下单用户 / Users with no orders:")
print(inactive[['user_id', 'username']])
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：将两个结构相同的 DataFrame 纵向拼接（垂直堆叠）在一起。
*参考答案*：
```python
# Time: O(N + M), Space: O(N + M)
# combined_df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
```

### 进阶题
**练习 2**：在特征工程中，你有多个文件分别存储了用户的“年龄段特征”、“消费等级特征”和“标签特征”，主键都是 `user_id`。编写代码将这三个表高效合并为一个宽表，并处理部分主键缺失的情况。
*参考答案*：
```python
import pandas as pd
t1 = pd.DataFrame({'user_id': [1, 2], 'age': [20, 25]})
t2 = pd.DataFrame({'user_id': [2, 3], 'spending': ['H', 'M']})
t3 = pd.DataFrame({'user_id': [1, 3], 'tag': ['A', 'B']})

# 使用 reduce 或 链式 merge 进行 Outer 合并 / Chain merge
res = pd.merge(t1, t2, on='user_id', how='outer')
wide_table = pd.merge(res, t3, on='user_id', how='outer')
print(wide_table)
```\n