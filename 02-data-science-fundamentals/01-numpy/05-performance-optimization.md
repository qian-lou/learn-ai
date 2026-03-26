# 性能优化（向量化 vs 循环）
# Performance Optimization (Vectorization vs Loops)

## 1. 背景（Background）

> **为什么要学这个？**
>
> **永远不要用 for 循环遍历数组元素**——这是从 Java 转 Python 最重要的思维转变。NumPy 向量化操作比 Python 循环快 **100-1000 倍**。在大模型数据预处理中，一个错误的循环可能让 pipeline 从 5 秒变成 50 分钟。

## 2. 知识点（Key Concepts）

| 模式 | Python 循环 | NumPy 向量化 |
|------|-----------|-------------|
| 加法 | `for` + `+=` | `a + b` |
| 条件 | `for` + `if` | `np.where()` |
| 聚合 | `sum()` | `np.sum()` |
| 查找 | `for` + `max` | `np.argmax()` |

## 3. 内容（Content）

```python
import numpy as np
import time

n = 10_000_000

# ============================================================
# 1. 基础运算 / Basic operations
# ============================================================
a = np.random.randn(n)
b = np.random.randn(n)

# ❌ Python 循环（~5s）
def slow_add(a, b):
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result

# ✅ 向量化（~5ms，快 1000x）
def fast_add(a, b):
    return a + b

# ============================================================
# 2. 条件赋值 / Conditional assignment
# ============================================================
# ❌ 循环 + if
def slow_relu(x):
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = x[i] if x[i] > 0 else 0
    return result

# ✅ np.where（ReLU 的向量化实现）
def fast_relu(x):
    return np.where(x > 0, x, 0)

# ✅ 更快：np.maximum
def fastest_relu(x):
    return np.maximum(x, 0)

# ============================================================
# 3. 聚合运算 / Aggregation
# ============================================================
# ❌ Python 内建（慢）
total = sum(a.tolist())

# ✅ NumPy（快 100x）
total = np.sum(a)
mean = np.mean(a)
std = np.std(a)
idx = np.argmax(a)  # 最大值索引

# 沿指定轴聚合
matrix = np.random.randn(1000, 768)
row_means = matrix.mean(axis=1)      # Shape: [1000]
col_means = matrix.mean(axis=0)      # Shape: [768]

# ============================================================
# 4. 避免临时数组 / Avoid temporary arrays
# ============================================================
# ❌ 创建多个临时数组（内存 3x）
result = np.sqrt(a**2 + b**2)

# ✅ 用 out 参数原地计算（内存 1x）
temp = np.empty_like(a)
np.multiply(a, a, out=temp)        # temp = a²
np.multiply(b, b, out=result)      # result = b²  (复用)
np.add(temp, result, out=result)   # result = a² + b²
np.sqrt(result, out=result)        # result = √(a² + b²)

# ============================================================
# 5. 用 NumPy 替代循环的常见模式
# ============================================================
# 累积和
cumsum = np.cumsum(a)

# 差分
diff = np.diff(a)

# 排序 + 索引
sorted_idx = np.argsort(a)[::-1]  # 降序索引

# 唯一值统计
values, counts = np.unique(np.random.randint(0, 10, 100), return_counts=True)
```

## 4. 详细推理（Deep Dive）

```
核心原则：思考维度和形状，不要思考循环和索引

NumPy 思维模式（正确）:
  "对这个 [1000, 768] 的矩阵，沿 axis=1 求均值，得到 [1000]"

Python 思维模式（错误）:
  "遍历 1000 行，对每行的 768 个元素求平均..."

性能层次:
  1. NumPy/PyTorch 向量化    → 最快
  2. numba.jit 编译         → 次快
  3. 纯 Python 循环          → 最慢（100-1000x）
```

## 5-6. 例题/习题

**练习 1：** 不使用循环，计算 1000 个样本与 100 个聚类中心的距离矩阵 `[1000, 100]`。

**练习 2：** 用向量化实现 Softmax 函数。

**练习 3：** 对比 `np.sum(a)` vs `sum(a.tolist())` 在 1 亿元素上的速度差异。
