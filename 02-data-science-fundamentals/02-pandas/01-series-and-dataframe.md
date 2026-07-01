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

## 5. 例题（Worked Examples）

### 例题 1：从字典创建 DataFrame 并执行多条件过滤与列添加 / Creating DataFrame and multi-condition filtering

在特征工程中，我们常用 Pandas 整理多维特征。本例展示创建 DataFrame，添加新特征并筛选特定子集。

```python
import pandas as pd
import numpy as np

# 1. 创建学生信息 DataFrame / Create student DataFrame
# Time: O(R * C), Space: O(R * C)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'score': [85, 92, 58, 76],
    'class': ['A', 'B', 'A', 'B']
}
df = pd.DataFrame(data)

# 2. 新增是否合格列 / Add pass/fail column
# Time: O(R), Space: O(R)
df['passed'] = df['score'] >= 60

# 3. 过滤出班级为 A 且成绩合格的学生 / Filter class A and passed students
# Time: O(R), Space: O(R_filtered * C)
filtered_df = df[(df['class'] == 'A') & (df['passed'] == True)]

print("过滤后的学生列表 / Filtered Students:")
print(filtered_df)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：创建一个包含五行数据的 DataFrame，每一行代表一个员工的姓名、部门和月薪，并计算全体员工的月薪平均值。
*参考答案*：
```python
import pandas as pd
df = pd.DataFrame({
    'name': ['E1', 'E2', 'E3', 'E4', 'E5'],
    'dept': ['IT', 'HR', 'IT', 'Sales', 'HR'],
    'salary': [8000, 6000, 9000, 7500, 6500]
})
# Time: O(N), Space: O(1)
print(f"平均月薪: {df['salary'].mean()}")
```

### 进阶题
**练习 2**：给定一个学生成绩 DataFrame，新增一列表示“成绩评级”：分数 >= 90 为 'A'，>= 80 为 'B'，>= 60 为 'C'，否则为 'D'。要求使用 pandas 的内置向量化方法（如 `pd.cut` 或 `apply`），避免使用 for 循环。
*参考答案*：
```python
import pandas as pd
# Time: O(N), Space: O(N)
df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David'], 'score': [85, 92, 58, 76]})
bins = [0, 60, 80, 90, 100]
labels = ['D', 'C', 'B', 'A']
df['grade'] = pd.cut(df['score'], bins=bins, labels=labels, right=False)
print(df)
```