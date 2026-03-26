# 性能优化（向量化 vs 循环）
# Performance Optimization (Vectorization vs Loops)

## 1. 背景（Background）

> NumPy 向量化是 Python 科学计算的性能关键。**永远不要用 for 循环遍历数组元素**——这是从 Java 转 Python 最重要的思维转变。

## 2-3. 知识点与内容

```python
import numpy as np
import time

# ❌ 错误：Python 循环（~5秒）
def slow_add(a, b):
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result

# ✅ 正确：向量化（~5毫秒）
def fast_add(a, b):
    return a + b  # 向量化，底层 C 循环

# 性能对比 / Performance comparison
n = 10_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# 向量化快 1000x！
# Vectorization is 1000x faster!

# 常见向量化模式 / Common vectorization patterns
# 1. 条件赋值
# ❌ for + if
# ✅ np.where(condition, value_if_true, value_if_false)
result = np.where(a > 0, a, 0)  # ReLU!

# 2. 聚合运算
np.sum(a)       # 比 sum(a) 快 100x
np.mean(a)
np.max(a)
np.argmax(a)    # 最大值的索引

# 3. 避免临时数组
# ❌ 创建多个临时数组
result = np.sqrt(a**2 + b**2)
# ✅ 用 out 参数原地计算
temp = np.empty_like(a)
np.multiply(a, a, out=temp)
```

## 4-6. 推理/例题/习题

**核心原则：** 在 NumPy/PyTorch 中，**思考维度和形状，而不是循环和索引**。

**练习：** 不使用循环，计算 1000 个样本与 100 个聚类中心的欧氏距离矩阵（Shape: [1000, 100]）。
