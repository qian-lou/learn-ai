# ndarray 基础与创建
# ndarray Basics and Creation

## 1. 背景（Background）

> **为什么要学这个？**
>
> NumPy 是 Python 科学计算的**基石**，`ndarray` 是其核心数据结构。PyTorch 的 `Tensor` 完全借鉴了 NumPy 的设计。掌握 NumPy 就是掌握 PyTorch 张量操作的前提——90% 的 API 用法一致。
>
> Java 没有原生多维数组库，NumPy 的能力相当于 **Apache Commons Math + Stream API + SIMD 优化**。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | PyTorch 对应 |
|------|------|-------------|
| `ndarray` | 多维同构数组 | `torch.Tensor` |
| `dtype` | 数据类型 | `torch.float32` |
| `shape` | 数组形状 | `.shape` |
| 向量化 | 替代 for 循环 | 同 |

## 3. 内容（Content）

### 3.1 创建 ndarray

```python
import numpy as np

# ============================================================
# 从 Python 对象创建 / Create from Python objects
# ============================================================
a = np.array([1, 2, 3])              # 1D: Shape [3]
b = np.array([[1, 2], [3, 4]])        # 2D: Shape [2, 2]
c = np.array([1, 2, 3], dtype=np.float32)  # 指定类型

# ============================================================
# 常用创建函数 / Common creation functions
# ============================================================
zeros = np.zeros((3, 4))             # Shape: [3, 4] 全 0
ones = np.ones((2, 3))               # Shape: [2, 3] 全 1
full = np.full((2, 3), fill_value=7) # Shape: [2, 3] 全 7
eye = np.eye(3)                       # Shape: [3, 3] 单位矩阵

# 序列
arange = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0]

# 随机数
rand_uniform = np.random.rand(3, 4)      # Shape: [3, 4] 均匀分布 [0,1)
rand_normal = np.random.randn(3, 4)      # Shape: [3, 4] 标准正态分布
rand_int = np.random.randint(0, 10, (3, 4)) # Shape: [3, 4] 随机整数

# ============================================================
# 属性 / Attributes
# ============================================================
print(b.shape)     # (2, 2)
print(b.dtype)     # int64
print(b.ndim)      # 2（维度数）
print(b.size)      # 4（元素总数）
print(b.nbytes)    # 32（字节数）
```

### 3.2 dtype 数据类型

```
大模型常用 dtype:

np.float32 / torch.float32  → 标准训练精度
np.float16 / torch.float16  → 半精度（推理加速）
np.float64 / torch.float64  → 双精度（科学计算）
np.int64   / torch.long     → 索引/标签
np.bool_   / torch.bool     → Mask

内存对比 (1000 万元素):
  float64: 80 MB
  float32: 40 MB
  float16: 20 MB
  int8:    10 MB
```

### 3.3 形状操作

```python
# ============================================================
# Reshape / 形状变换
# ============================================================
a = np.arange(12)           # Shape: [12]
b = a.reshape(3, 4)         # Shape: [3, 4]
c = a.reshape(2, 2, 3)      # Shape: [2, 2, 3]
d = a.reshape(-1, 4)        # Shape: [3, 4]  -1 自动推断

# ============================================================
# 向量化运算 vs 循环 / Vectorized ops vs loops
# ============================================================
n = 1_000_000
arr = np.arange(n)

# ❌ 慢：Python 循环（~500ms）
result_slow = [x**2 for x in range(n)]

# ✅ 快：向量化（~2ms，快 250x）
result_fast = arr ** 2
```

## 4. 详细推理（Deep Dive）

```
为什么 NumPy 比 Python 循环快 100-1000x？

1. C 实现: NumPy 底层用 C/Fortran 写循环
2. 内存连续: ndarray 连续存储，CPU 缓存友好
3. SIMD: 利用 CPU 的向量化指令（一次处理 4/8 个数）
4. 无 Python 开销: 不需要类型检查、引用计数等

Python list vs ndarray:
  list:  [指针→对象, 指针→对象, ...] → 内存分散
  ndarray: [数据, 数据, 数据, ...]    → 内存连续
```

## 5. 例题（Worked Examples）

### 例题 1：从零构建模型权重矩阵并计算 L2 正则项 / Build a model weight matrix from scratch and calculate L2 regularization

本例演示如何使用 NumPy 创建一个全零的偏置向量、一个随机初始化的权重矩阵，并计算其 L2 范数（正则化项）。

```python
import numpy as np

# 1. 初始化模型参数 / Initialize model parameters
# Time: O(D_in * D_out), Space: O(D_in * D_out)
d_in, d_out = 128, 64
weights = np.random.randn(d_in, d_out) * 0.01   # Shape: [128, 64]
biases = np.zeros((d_out,))                      # Shape: [64]

# 2. 计算权重的 L2 正则项 (平方和除以 2) / Calculate L2 regularization term
# Time: O(D_in * D_out), Space: O(1)
l2_reg = 0.5 * np.sum(np.square(weights))

print(f"权重形状 / Weights shape: {weights.shape}")
print(f"偏置形状 / Biases shape: {biases.shape}")
print(f"L2 正则值 / L2 Reg value: {l2_reg:.6f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：创建一个 5x5 的单位矩阵，并将其对角线元素修改为 10。
*参考答案*：
```python
import numpy as np
# Time: O(N^2), Space: O(N^2)
matrix = np.eye(5)
matrix[np.diag_indices(5)] = 10
print(matrix)
```

### 进阶题
**练习 2**：假设有一个一维数组表示的输入层特征 `x`（长度 128）以及权重矩阵 `W`（128x64），编写代码手动计算前向传播输出 $z = xW + b$，要求参数符合 NumPy 高效的底层加速特性（禁止使用 for 循环）。
*参考答案*：
```python
import numpy as np
# Time: O(D_in * D_out), Space: O(D_out)
x = np.random.randn(128)  # Shape: [128]
W = np.random.randn(128, 64)  # Shape: [128, 64]
b = np.zeros(64)  # Shape: [64]
z = np.dot(x, W) + b  # Shape: [64]
print(f"输出形状: {z.shape}")
```\n