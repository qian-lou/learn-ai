# 索引与切片
# Indexing and Slicing

## 1. 背景（Background）

> NumPy 的索引和切片比 Python 列表强大得多，支持花式索引、布尔索引和多维切片。这些操作在数据预处理和特征提取中无处不在。

## 2-3. 知识点与内容

```python
import numpy as np

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])  # Shape: [3, 4]

# 基础索引 / Basic indexing
a[0, 0]      # 1
a[1, 2]      # 7
a[-1, -1]    # 12

# 切片（返回视图，不复制数据！）/ Slicing (returns view, no copy!)
a[0:2, 1:3]  # [[2,3],[6,7]]  Shape: [2, 2]
a[:, 0]      # [1, 5, 9]  第 0 列
a[1, :]      # [5, 6, 7, 8]  第 1 行

# 布尔索引（极常用！）/ Boolean indexing
mask = a > 5
print(a[mask])  # [6, 7, 8, 9, 10, 11, 12]

# 花式索引 / Fancy indexing
indices = [0, 2]
print(a[indices])  # 选取第 0 行和第 2 行

# ⚠️ 视图 vs 副本
# Slicing returns VIEW (shared memory), copy() returns COPY
view = a[0:2]
view[0, 0] = 999  # 原数组 a 也被修改！
```

## 4-6. 推理/例题/习题

**练习 1：** 用布尔索引将矩阵中所有负数替换为 0。
**练习 2：** 不使用循环，提取矩阵对角线元素。
