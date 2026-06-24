# 线性代数基础
# Linear Algebra Foundations

## 1. 背景（Background）

> **为什么要学这个？**
>
> 线性代数是深度学习的**数学语言**。矩阵乘法驱动 Transformer 的每一次前向传播，特征分解是 PCA 的基础，SVD 是 LoRA 的理论依据。不懂线性代数就无法真正理解深度学习。

## 2. 知识点（Key Concepts）

| 概念 | 深度学习应用 |
|------|-------------|
| 矩阵乘法 | 全连接层, Attention |
| 特征值/特征向量 | PCA, 谱聚类 |
| SVD | LoRA, 降维 |
| 范数 | 正则化, 梯度裁剪 |
| 正交矩阵 | 权重初始化 |

## 3. 内容（Content）

### 3.1 向量与矩阵

```python
import numpy as np

# ============================================================
# 向量运算 / Vector operations
# ============================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 点积（内积）— 衡量相似度
dot = np.dot(a, b)  # 1×4 + 2×5 + 3×6 = 32

# 余弦相似度 — Embedding 相似度的基础
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# 矩阵运算 / Matrix operations
# ============================================================
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法 — 神经网络的核心运算
C = A @ B  # Shape: [2,2] @ [2,2] → [2,2]
# 含义: 线性变换（旋转 + 缩放）

# 转置
print(A.T)

# 逆矩阵
A_inv = np.linalg.inv(A)

# 行列式（衡量矩阵"缩放程度"）
det = np.linalg.det(A)
```

### 3.2 特征值与 SVD

```python
# ============================================================
# 特征值分解 / Eigendecomposition
# A·v = λ·v (v 是特征向量, λ 是特征值)
# ============================================================
eigenvalues, eigenvectors = np.linalg.eig(A)
# PCA 的核心: 协方差矩阵的特征向量 = 主成分方向

# ============================================================
# SVD 奇异值分解 / Singular Value Decomposition
# A = U·Σ·V^T
# ============================================================
U, S, Vt = np.linalg.svd(A)
# LoRA: 权重矩阵是低秩的 → 只需用 r 个最大奇异值近似
```

### 3.3 范数

```python
v = np.array([3.0, 4.0])

# L1 范数（曼哈顿距离）— L1 正则化
l1 = np.linalg.norm(v, ord=1)  # 7.0

# L2 范数（欧氏距离）— L2 正则化, 梯度裁剪
l2 = np.linalg.norm(v, ord=2)  # 5.0

# Frobenius 范数（矩阵的 L2）
fro = np.linalg.norm(A, 'fro')
```

## 4. 详细推理（Deep Dive）

```
深度学习中的线性代数:

全连接层: y = Wx + b  (矩阵乘法)
Attention: softmax(QK^T/√d)V  (三次矩阵乘法)
BatchNorm: (x - μ) / σ  (向量运算)
梯度裁剪: if ||g|| > max_norm: g *= max_norm / ||g||  (范数)
LoRA: ΔW = BA, rank(BA) = r << d  (低秩分解)
```

## 5. 例题（Worked Examples）

### 例题 1：奇异值分解 (SVD) 降低图片特征维度 / SVD for feature dimension reduction

奇异值分解 (SVD) 是很多降维算法（如 PCA）的核心。本例题演示如何用 NumPy 对特征矩阵进行 SVD 分解，并保留前 K 个奇异值对原矩阵进行近似重构。

```python
import numpy as np

# 模拟 100x64 的数据特征矩阵 / Simulate a 100x64 feature matrix
X = np.random.randn(100, 64)  # Shape: [100, 64]

# 1. 执行 SVD 分解 / Perform SVD
# Time: O(M * N * min(M,N)), Space: O(M * N)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# U Shape: [100, 64], S Shape: [64], Vt Shape: [64, 64]

# 2. 近似重构 (仅保留前 K 个最大的奇异值) / Low-rank approximation (keep top K singular values)
K = 10
# S 是对角矩阵的向量，我们需要切片 / Slice top K elements
S_k = np.diag(S[:K])  # Shape: [10, 10]
U_k = U[:, :K]        # Shape: [100, 10]
Vt_k = Vt[:K, :]      # Shape: [10, 64]

# 重构特征 / Reconstruct matrix
# Time: O(M * K * N), Space: O(M * N)
X_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))

print(f"原始矩阵二范数 / Original Norm: {np.linalg.norm(X):.4f}")
print(f"保留 10 个奇异值后的重构二范数 / Reconstructed Norm (K=10): {np.linalg.norm(X_reconstructed):.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：使用 NumPy 计算矩阵的迹（对角线元素之和）和行列式。
*参考答案*：
```python
import numpy as np
# Time: O(N^3) for det, Space: O(N^2)
A = np.array([[4, 2], [3, 1]])
print(f"迹 / Trace: {np.trace(A)}")
print(f"行列式 / Det: {np.linalg.det(A):.4f}")
```

### 进阶题
**练习 2**：推导在线性回归中，设计矩阵为 $X \in \mathbb{R}^{N \times D}$ 时，解析解（最小二乘解）的公式：$\theta = (X^T X)^{-1} X^T y$。在 $X^T X$ 不满秩时，如何使用伪逆（Moore-Penrose Pseudoinverse）解决求解问题？请编写 NumPy 代码验证。
*参考答案*：
当 $X^T X$ 不可逆（不满秩）时，无法直接求逆，我们需要使用 `np.linalg.pinv` 计算伪逆。
```python
import numpy as np
# Time: O(D^3), Space: O(N * D)
X = np.random.randn(5, 5)
X[:, 4] = X[:, 3] * 2  # 创造线性相关列（X[:,4]=2·X[:,3]）使之不满秩 / dependent column
y = np.random.randn(5)
# 计算伪逆解析解
theta = np.dot(np.linalg.pinv(X), y)
print(f"系数权重 theta: {theta}")
```
