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

## 5. 例题（Worked Examples）

### 例题 1：线性方程组求解与矩阵求逆 / Solving Linear Systems and Matrix Inversion

本例题演示如何使用 NumPy 的线性代数模块求解方程组 $Ax = b$ 并求矩阵 $A$ 的逆。

```python
import numpy as np

# 定义系数矩阵 A 和常数向量 b / Define coefficient matrix A and constant vector b
A = np.array([[2, 1], [1, 3]])  # Shape: [2, 2]
b = np.array([5, 8])            # Shape: [2]

# 1. 求解 Ax = b / Solve Ax = b
# Time: O(D^3) - D 为矩阵维度 / D is matrix dimension.
# Space: O(D^2)
x = np.linalg.solve(A, b)  # [1.4, 2.2]

# 2. 求 A 的逆矩阵并验证 / Compute inverse of A and verify
# Time: O(D^3), Space: O(D^2)
A_inv = np.linalg.inv(A)
I_test = np.dot(A, A_inv)  # 应接近单位矩阵 / Should be identity matrix.

print(f"方程组的解 / Solution: {x}")
print(f"A 的逆矩阵 / Inverse matrix A_inv:\n{A_inv}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：计算两个二维矩阵 `A = [[1, 2], [3, 4]]` 和 `B = [[5, 6], [7, 8]]` 的矩阵乘积。
*参考答案*：
```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Time: O(N^3), Space: O(N^2)
C = np.dot(A, B)  # 或 A @ B
print(C)
```

### 进阶题
**练习 2**：在主成分分析（PCA）中，需要计算特征协方差矩阵的特征值与特征向量。生成一个 10x10 的对称正定协方差矩阵，使用 NumPy 计算其特征值和特征向量，并按特征值从大到小对特征向量进行排序。
*参考答案*：
```python
import numpy as np
# Time: O(D^3), Space: O(D^2)
# 随机生成对称正定矩阵
X = np.random.randn(10, 10)
cov = np.dot(X, X.T)  # Shape: [10, 10]

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 降序排列索引
idx = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[idx]
sorted_eigenvectors = eigenvectors[:, idx]

print(f"最大特征值: {sorted_eigenvalues[0]:.4f}")
```