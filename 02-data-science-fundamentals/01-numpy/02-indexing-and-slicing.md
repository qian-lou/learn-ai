# 索引与切片
# Indexing and Slicing

## 1. 背景（Background）

> **为什么要学这个？**
>
> NumPy 的索引和切片是数据预处理和特征提取的基础操作。布尔索引在数据清洗、掩码操作（Attention Mask）中无处不在。理解视图 vs 副本可以避免隐蔽的 bug。

## 2. 知识点（Key Concepts）

| 操作 | 语法 | 返回 |
|------|------|------|
| 基础索引 | `a[i, j]` | 标量 |
| 切片 | `a[start:end]` | 视图 |
| 布尔索引 | `a[a > 0]` | 副本 |
| 花式索引 | `a[[0, 2]]` | 副本 |

## 3. 内容（Content）

```python
import numpy as np

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])  # Shape: [3, 4]

# ============================================================
# 基础索引 / Basic indexing
# ============================================================
a[0, 0]      # 1
a[1, 2]      # 7
a[-1, -1]    # 12

# ============================================================
# 切片（返回视图！）/ Slicing (returns VIEW!)
# ============================================================
a[0:2, 1:3]  # [[2,3],[6,7]]  Shape: [2, 2]
a[:, 0]      # [1, 5, 9]  第 0 列
a[1, :]      # [5, 6, 7, 8]  第 1 行
a[::2, :]    # 每隔一行取 → [[1,2,3,4],[9,10,11,12]]

# ============================================================
# 布尔索引（极常用！）/ Boolean indexing
# ============================================================
mask = a > 5
print(a[mask])  # [6, 7, 8, 9, 10, 11, 12]

# 条件替换（类似 ReLU）
b = a.copy()
b[b < 5] = 0   # 小于 5 的全部置 0

# np.where（三元运算符的向量化版本）
result = np.where(a > 5, a, 0)  # 大于 5 保留，否则 0

# ============================================================
# 花式索引 / Fancy indexing
# ============================================================
indices = [0, 2]
print(a[indices])  # 选取第 0 行和第 2 行

# 同时选择特定行和列
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 3])
print(a[rows, cols])  # [2, 7, 12]（对角线方向）

# ============================================================
# ⚠️ 视图 vs 副本（重要陷阱！）
# ============================================================
# 切片 = 视图（共享内存）
view = a[0:2]
view[0, 0] = 999  # 原数组 a 也被修改！

# 布尔/花式索引 = 副本（独立内存）
copy = a[a > 5]
copy[0] = 999  # 原数组 a 不受影响

# 显式复制
safe_copy = a[0:2].copy()  # 强制创建副本
```

## 4. 详细推理（Deep Dive）

```
视图 vs 副本的判断规则:

返回视图（共享内存）:
  - 基础切片: a[0:2], a[:, 1:3]
  - reshape: a.reshape(2, 6)
  - 转置: a.T

返回副本（独立内存）:
  - 布尔索引: a[a > 0]
  - 花式索引: a[[0, 2]]
  - .copy(): 显式复制

为什么重要？
  视图修改会影响原数组 → 可能产生隐蔽 bug
  副本消耗额外内存 → 大数据时注意内存
```

## 5. 例题（Worked Examples）

### 例题 1：提取特征图的局部窗口与掩码过滤 / Extract feature map local window and mask filtering

本例模拟提取卷积特征图中特定感兴趣区域 (ROI)，并将其中所有负数截断为 0 (类似 ReLU 激活)。

```python
import numpy as np

# 模拟 8x8 特征图 / Simulate an 8x8 feature map
feature_map = np.random.randn(8, 8)  # Shape: [8, 8]

# 1. 提取中心 4x4 局部窗口 / Extract central 4x4 window
# Time: O(1) - 视图切片不进行数据拷贝 / View slicing is zero-copy.
# Space: O(1)
roi = feature_map[2:6, 2:6]  # Shape: [4, 4]

# 2. 对 ROI 进行 ReLU 处理 (布尔索引过滤并修改) / Apply ReLU to ROI
# Time: O(W_roi * H_roi), Space: O(1)
roi[roi < 0] = 0.0

print("处理后的特征图局部窗口 / Processed ROI:")
print(roi)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：给定一维数组，提取所有奇数索引处的元素。
*参考答案*：
```python
import numpy as np
# Time: O(1) view slice
arr = np.array([10, 20, 30, 40, 50])
odd_idx_elements = arr[1::2]  # [20, 40]
```

### 进阶题
**练习 2**：从一个二维矩阵中，筛选出所有行均值大于 0 的完整行，并组成新的矩阵。
*参考答案*：
```python
import numpy as np
# Time: O(R * C), Space: O(R_selected * C)
matrix = np.random.randn(5, 4)
row_means = matrix.mean(axis=1)  # Shape: [5]
filtered_matrix = matrix[row_means > 0]  # 布尔索引筛选行
print(f"筛选后的矩阵形状: {filtered_matrix.shape}")
```