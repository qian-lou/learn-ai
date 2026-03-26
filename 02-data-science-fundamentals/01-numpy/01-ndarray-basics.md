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

## 5-6. 例题/习题

**练习 1：** 用 NumPy 计算两个 1000 维向量的余弦相似度（不用循环）。

**练习 2：** 创建一个 1000×768 的随机矩阵（模拟 BERT embeddings），统计其均值、标准差。

**练习 3：** 对比 `np.float32` 和 `np.float64` 在 1 亿元素上的内存占用和计算速度。
