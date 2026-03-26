# 广播机制
# Broadcasting

## 1. 背景（Background）

> 广播（Broadcasting）是 NumPy/PyTorch 中最重要的概念之一——它允许不同形状的数组进行运算，无需手动扩展维度。大模型中的 Attention Score 计算、Mask 操作都依赖广播。

## 2-3. 知识点与内容

```python
import numpy as np

# 标量广播 / Scalar broadcasting
a = np.array([1, 2, 3])  # Shape: [3]
print(a * 2)              # [2, 4, 6]  —— 2 被广播为 [2, 2, 2]

# 不同形状的广播 / Different shape broadcasting
# Shape: [3, 3] + Shape: [3] → 行向量被广播到每一行
matrix = np.ones((3, 3))   # Shape: [3, 3]
row = np.array([1, 2, 3])  # Shape: [3]
print(matrix + row)
# [[2, 3, 4],
#  [2, 3, 4],
#  [2, 3, 4]]

# 广播规则：
# Broadcasting rules:
# 1. 从右向左对齐维度
# 2. 维度大小相同，或其中一个为 1，则兼容
# 3. 大小为 1 的维度被"拉伸"

# Shape: [4, 1] + Shape: [1, 3] → Shape: [4, 3]
a = np.arange(4).reshape(4, 1)  # Shape: [4, 1]
b = np.arange(3).reshape(1, 3)  # Shape: [1, 3]
print((a + b).shape)            # (4, 3) — 外积效果
```

## 4. 详细推理（Deep Dive）

```
广播规则图示：
Shape [4, 1] + Shape [1, 3]      
         ↓              ↓
Shape [4, 3] + Shape [4, 3]  ← 两个维度都被拉伸
         ↓
    Shape [4, 3]  ← 结果

不兼容的例子：
Shape [3] + Shape [4] → ❌ Error!（维度不匹配且都不是1）
```

## 5-6. 例题/习题

**练习：** 用广播实现矩阵每行减去行均值（标准化），不使用 for 循环。

```python
# 参考答案
# Time: O(M*N)  Space: O(M)
matrix = np.random.randn(100, 50)
normalized = matrix - matrix.mean(axis=1, keepdims=True)
```
