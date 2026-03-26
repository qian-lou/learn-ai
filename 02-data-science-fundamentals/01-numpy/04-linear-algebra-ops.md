# 线性代数运算
# Linear Algebra Operations

## 1. 背景（Background）

> **为什么要学这个？**
>
> 线性代数是深度学习的数学基础。矩阵乘法、SVD、特征分解在 Transformer 的注意力计算中无处不在。`einsum` 是描述张量运算的通用语言——掌握它就能读懂任何大模型代码。

## 2. 知识点（Key Concepts）

| 运算 | API | 大模型应用 |
|------|-----|-----------|
| 矩阵乘法 | `A @ B` | Attention: QK^T |
| SVD | `np.linalg.svd` | LoRA 理论基础 |
| einsum | `np.einsum` | 任意张量运算 |
| 范数 | `np.linalg.norm` | 梯度裁剪 |

## 3. 内容（Content）

```python
import numpy as np

# ============================================================
# 矩阵乘法 / Matrix multiplication
# ============================================================
A = np.array([[1, 2], [3, 4]])  # Shape: [2, 2]
B = np.array([[5, 6], [7, 8]])  # Shape: [2, 2]

C = A @ B          # 推荐写法（Python 3.5+）
C = np.dot(A, B)   # 等价写法
C = np.matmul(A, B) # 等价写法

# 转置 / Transpose
print(A.T)  # [[1, 3], [2, 4]]

# ============================================================
# 逆矩阵 + 行列式 / Inverse + Determinant
# ============================================================
A_inv = np.linalg.inv(A)
print(A @ A_inv)     # ≈ 单位矩阵
det = np.linalg.det(A)  # -2.0

# ============================================================
# 特征值分解 / Eigendecomposition
# ============================================================
eigenvalues, eigenvectors = np.linalg.eig(A)

# ============================================================
# SVD（LoRA 的理论基础！）
# ============================================================
# A = U · Σ · V^T
U, S, Vt = np.linalg.svd(A)
# 低秩近似：取前 r 个奇异值重建
r = 1
A_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
print(f"原矩阵:\n{A}")
print(f"秩-{r} 近似:\n{A_approx}")

# ============================================================
# einsum（爱因斯坦求和，超重要！）
# ============================================================
# 矩阵乘法
C = np.einsum('ij,jk->ik', A, B)

# 向量点积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.einsum('i,i->', a, b)  # 32

# 批量矩阵乘法 (Attention 中的 QK^T)
# Q: [B, H, S, D], K: [B, H, S, D]
# scores = einsum('bhsd,bhtd->bhst', Q, K)

# 迹（对角线元素之和）
trace = np.einsum('ii->', A)

# 外积
outer = np.einsum('i,j->ij', a, b)

# ============================================================
# 范数 / Norms
# ============================================================
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))       # L2 范数: 5.0
print(np.linalg.norm(v, ord=1)) # L1 范数: 7.0
# 梯度裁剪: if norm > max_norm: grad *= max_norm / norm
```

## 4. 详细推理（Deep Dive）

```
einsum 速记表:

'ij,jk->ik'      矩阵乘法 (A @ B)
'bhsd,bhtd->bhst' Attention score (QK^T)
'ij->ji'          转置
'ii->'            迹
'i,j->ij'         外积
'bij,bjk->bik'    批量矩阵乘法

SVD 与 LoRA:
  权重矩阵 W ∈ R^{d×d} 的有效秩远小于 d
  SVD: W = U·Σ·V^T
  LoRA: ΔW = B·A，其中 B ∈ R^{d×r}, A ∈ R^{r×d}
  → LoRA 本质是在低秩空间中更新权重
```

## 5-6. 例题/习题

**练习 1：** 用 NumPy 实现最小二乘法线性回归：`w = (X^T X)^{-1} X^T y`。

**练习 2：** 用 `einsum` 实现 Attention Score 计算。

**练习 3：** 对一个大矩阵做 SVD，观察奇异值的衰减速度，验证"低秩假设"。
