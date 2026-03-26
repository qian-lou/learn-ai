# 线性代数运算
# Linear Algebra Operations

## 1. 背景（Background）

> 线性代数是深度学习的数学基础。矩阵乘法、特征分解、SVD 等运算在 Transformer 的注意力计算中无处不在。NumPy 提供了完整的线性代数工具。

## 2-3. 知识点与内容

```python
import numpy as np

# 矩阵乘法 / Matrix multiplication
A = np.array([[1, 2], [3, 4]])  # Shape: [2, 2]
B = np.array([[5, 6], [7, 8]])  # Shape: [2, 2]

C = A @ B          # 推荐写法（Python 3.5+ 的 @ 运算符）
C = np.dot(A, B)   # 等价写法
C = np.matmul(A, B) # 等价写法

# 转置 / Transpose
print(A.T)  # [[1, 3], [2, 4]]

# 逆矩阵 / Inverse
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # 近似单位矩阵

# 行列式 / Determinant
det = np.linalg.det(A)  # -2.0

# 特征值分解 / Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD（大模型中 LoRA 的理论基础！）
# SVD (theoretical basis for LoRA in LLMs!)
U, S, Vt = np.linalg.svd(A)

# einsum（爱因斯坦求和，PyTorch 中超常用）
# Einstein summation (very common in PyTorch)
# 矩阵乘法的 einsum 写法
C = np.einsum('ij,jk->ik', A, B)
# 批量矩阵乘法
# batch_A: Shape [B, M, K], batch_B: Shape [B, K, N]
# result = np.einsum('bmk,bkn->bmn', batch_A, batch_B)
```

## 4. 详细推理

- `@` 运算符调用 `__matmul__`，PyTorch 也支持
- `einsum` 是描述张量运算的通用语言，Attention 的 QK^T 可以用 `einsum('bhsd,bhtd->bhst', Q, K)` 表示
- SVD 将矩阵分解为 U·Σ·V^T，LoRA 利用低秩近似来高效微调大模型

## 5-6. 例题/习题

**练习 1：** 用 NumPy 实现最小二乘法拟合一元线性回归。
**练习 2：** 用 `einsum` 实现批量点积运算。
