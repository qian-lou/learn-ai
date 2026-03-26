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

## 5-6. 例题/习题

**练习 1：** 用 NumPy 实现 PCA 降维（基于协方差矩阵特征分解）。

**练习 2：** 验证 SVD 低秩近似：对图像矩阵做 SVD，只保留前 k 个奇异值重建，观察图像质量。
