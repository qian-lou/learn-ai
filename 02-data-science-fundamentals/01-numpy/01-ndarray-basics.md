# ndarray 基础与创建
# ndarray Basics and Creation

## 1. 背景（Background）

> NumPy 是 Python 科学计算的基石，`ndarray`（N-dimensional array）是其核心数据结构。PyTorch 的 `Tensor` 设计灵感来自 NumPy。Java 没有原生的多维数组库，最接近的是 Apache Commons Math。掌握 NumPy 是理解 PyTorch 张量操作的前提。

## 2. 知识点（Key Concepts）

- `ndarray`：多维同构数组，所有元素类型相同
- `dtype`：数据类型（float32/float64/int64 等），类似 Java 泛型约束
- `shape`：数组形状，如 `(3, 4)` 表示 3 行 4 列
- 向量化运算：用数组运算替代 for 循环，性能提升 100x+

## 3. 内容（Content）

```python
import numpy as np

# 创建 ndarray / Create ndarray
a = np.array([1, 2, 3])              # 1D: Shape [3]
b = np.array([[1, 2], [3, 4]])        # 2D: Shape [2, 2]

# 常用创建函数 / Common creation functions
zeros = np.zeros((3, 4))             # Shape: [3, 4] 全 0
ones = np.ones((2, 3))               # Shape: [2, 3] 全 1
eye = np.eye(3)                       # Shape: [3, 3] 单位矩阵
rand = np.random.randn(3, 4)         # Shape: [3, 4] 标准正态分布
arange = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0]

# 属性 / Attributes
print(b.shape)     # (2, 2)
print(b.dtype)     # int64
print(b.ndim)      # 2
print(b.size)      # 4

# 向量化运算 vs 循环 / Vectorized ops vs loops
# ❌ 慢：for 循环
result_slow = [x**2 for x in range(1000000)]
# ✅ 快：向量化（100x faster）
arr = np.arange(1000000)
result_fast = arr ** 2
```

## 4. 详细推理（Deep Dive）

- NumPy 底层用 C 实现，内存连续存储，CPU 缓存友好
- `float32` 是大模型训练的标准数据类型（平衡精度与内存）
- 向量化的本质：将循环下推到 C 层面，同时利用 SIMD 指令

## 5-6. 例题/习题

**练习：** 不使用 for 循环，用 NumPy 计算两个 1000 维向量的余弦相似度。
