# 线性代数（矩阵/向量/特征值）
# Linear Algebra (Matrix/Vector/Eigenvalue)

## 1. 背景（Background）

> 线性代数是机器学习和深度学习的数学基础。Transformer 中的注意力机制本质上就是矩阵乘法。理解线性代数才能理解模型为什么这样计算。

## 2. 知识点（Key Concepts）

- **向量**：特征的数学表示。一个词嵌入就是一个 768 维向量
- **矩阵**：线性变换的表示。全连接层就是矩阵乘法 `y = Wx + b`
- **特征值/特征向量**：PCA 降维的理论基础
- **矩阵分解**：SVD 是 LoRA 微调的理论基础

## 3. 内容（Content）

```python
import numpy as np

# 向量运算 / Vector operations
v1 = np.array([1, 2, 3])  # Shape: [3]
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)       # 点积: 32
cross = np.cross(v1, v2)            # 叉积
norm = np.linalg.norm(v1)           # L2 范数: sqrt(14)

# 余弦相似度（NLP 中衡量文本相似度）
# Cosine similarity (measuring text similarity in NLP)
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 矩阵运算 / Matrix operations
W = np.random.randn(768, 3072)  # 全连接层权重 Shape: [768, 3072]
x = np.random.randn(768)         # 输入向量 Shape: [768]
y = W.T @ x                      # 线性变换 Shape: [3072]

# 特征值分解 / Eigendecomposition
A = np.array([[2, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD 分解（LoRA 基础）/ SVD decomposition (basis for LoRA)
U, S, Vt = np.linalg.svd(W, full_matrices=False)
# 低秩近似：只保留前 r 个奇异值
r = 16  # LoRA 的典型 rank
W_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
```

## 4. 详细推理（Deep Dive）

大模型中的线性代数无处不在：
- **Embedding 层**：查表 = 矩阵的行选择
- **Attention**：QK^T 是矩阵乘法，Softmax(QK^T/√d)V 也是
- **FFN 层**：两个矩阵乘法
- **LoRA**：W' = W + BA，其中 B∈R^{d×r}, A∈R^{r×k}, r << d

## 5-6. 例题/习题

**练习 1：** 手算 2×2 矩阵的特征值和特征向量。
**练习 2：** 用 SVD 实现图像压缩（保留前 k 个奇异值）。
