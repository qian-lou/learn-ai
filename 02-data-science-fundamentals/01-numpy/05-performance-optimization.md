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

## 5. 例题（Worked Examples）

### 例题 1：使用矢量化计算高维空间中的两两距离 / Pairwise distance calculation in high-dimensional space

本例对比传统的 for 循环方式与 NumPy 的矢量化（矩阵操作）方式，计算一批样本之间的欧氏距离，显示矢量化所带来的巨大性能优势。

```python
import numpy as np
import time

# 模拟 1000 个 128 维的特征向量 / Simulate 1000 128-d vectors
X = np.random.randn(1000, 128)  # Shape: [1000, 128]

# 1. 传统 for 循环计算 / Loop-based computation
# Time: O(N^2 * D), Space: O(N^2)
start = time.perf_counter()
dist_loop = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        dist_loop[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
loop_time = time.perf_counter() - start

# 2. 矢量化计算 (利用公式 (A-B)^2 = A^2 + B^2 - 2AB) / Vectorized computation
# Time: O(N^2 * D) - 用 C 语言级 BLAS 加速 / Accelerated via BLAS.
# Space: O(N^2)
start = time.perf_counter()
X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # Shape: [1000, 1]
dist_vec = np.sqrt(np.maximum(X_sq + X_sq.T - 2 * np.dot(X, X.T), 0.0))
vec_time = time.perf_counter() - start

print(f"循环耗时 / Loop time: {loop_time:.4f}s")
print(f"矢量化耗时 / Vectorized time: {vec_time:.4f}s")
print(f"加速比 / Speedup: {loop_time / vec_time:.1f}x")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：使用 np.where 函数实现：对于数组中的每个元素，如果大于 0 保持不变，否则乘以 0.1（类似 LeakyReLU 激活函数）。
*参考答案*：
```python
import numpy as np
# Time: O(N), Space: O(N)
x = np.array([-2.0, -0.5, 1.0, 3.0])
y = np.where(x > 0, x, x * 0.1)
print(y)  # [-0.2, -0.05, 1.0, 3.0]
```

### 进阶题
**练习 2**：编写一个矢量化的梯度裁剪（Gradient Clipping）函数。输入一个包含梯度的 ndarray 列表（每个 ndarray 代表一个权重参数的梯度），计算它们的全局二范数，并在二范数超过给定的最大值 `max_norm` 时进行按比例缩放。
*参考答案*：
```python
import numpy as np
from typing import List

def clip_grad_norm(grads: List[np.ndarray], max_norm: float) -> None:
    """矢量化梯度裁剪 / Vectorized gradient clipping.
    
    Time: O(Total_Params), Space: O(1)
    """
    # 计算全局平方和
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # 原地修改梯度
    if clip_coef < 1.0:
        for g in grads:
            g *= clip_coef

# 测试
g = [np.array([10.0, 5.0]), np.array([2.0, 1.0])]
clip_grad_norm(g, max_norm=5.0)
print(f"裁剪后梯度: {g}")
```\n