# 广播机制
# Broadcasting

## 1. 背景（Background）

> **为什么要学这个？**
>
> 广播（Broadcasting）是 NumPy/PyTorch 最重要的概念之一——它允许不同形状的数组进行运算，无需手动扩展维度。Attention Score 计算、Mask 操作、Batch Normalization 都依赖广播。不理解广播 = 无法理解大模型代码。

## 2. 知识点（Key Concepts）

```
广播三条规则（从右向左对齐）:

1. 维度数不同 → 小数组在左侧补 1
2. 维度大小相同 → 兼容
3. 维度大小为 1 → 被"拉伸"到另一个的大小
4. 维度大小不同且都不为 1 → ❌ Error!
```

## 3. 内容（Content）

```python
import numpy as np

# ============================================================
# 1. 标量广播 / Scalar broadcasting
# ============================================================
a = np.array([1, 2, 3])  # Shape: [3]
print(a * 2)              # [2, 4, 6]  → 2 广播为 [2, 2, 2]

# ============================================================
# 2. 行向量广播 / Row vector broadcasting
# ============================================================
matrix = np.ones((3, 3))   # Shape: [3, 3]
row = np.array([1, 2, 3])  # Shape: [3] → 视为 [1, 3] → 广播为 [3, 3]
print(matrix + row)
# [[2, 3, 4],
#  [2, 3, 4],
#  [2, 3, 4]]

# ============================================================
# 3. 外积效果 / Outer product effect
# ============================================================
a = np.arange(4).reshape(4, 1)  # Shape: [4, 1]
b = np.arange(3).reshape(1, 3)  # Shape: [1, 3]
print((a + b).shape)            # (4, 3)
# [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]

# ============================================================
# 4. 实际应用：行均值标准化
# ============================================================
data = np.random.randn(100, 50)  # Shape: [100, 50]
mean = data.mean(axis=1, keepdims=True)  # Shape: [100, 1]
std = data.std(axis=1, keepdims=True)    # Shape: [100, 1]
normalized = (data - mean) / std  # 广播: [100, 50] - [100, 1]

# ============================================================
# 5. Attention Mask 广播
# ============================================================
# mask: Shape [B, 1, 1, S]  广播到 [B, H, S, S]
# scores: Shape [B, H, S, S]
# scores = scores + mask  ← 广播！
```

## 4. 详细推理（Deep Dive）

```
广播规则图示：

Shape [4, 1] + Shape [1, 3]
  Step 1: 维度对齐 → [4, 1] vs [1, 3]
  Step 2: 1 被拉伸 → [4, 3] vs [4, 3]
  Step 3: 逐元素运算 → Shape [4, 3]

Shape [3, 4, 1] + Shape [4, 5]
  Step 1: 补维度 → [3, 4, 1] vs [1, 4, 5]
  Step 2: 拉伸   → [3, 4, 5] vs [3, 4, 5]
  Step 3: 结果   → Shape [3, 4, 5]

❌ 不兼容: Shape [3] + Shape [4] → Error!
❌ 不兼容: Shape [2, 3] + Shape [3, 2] → Error!

keepdims=True 的作用:
  mean(axis=1):              Shape [100] → 无法与 [100, 50] 广播
  mean(axis=1, keepdims=True): Shape [100, 1] → 可以广播！
```

## 5. 例题（Worked Examples）

### 例题 1：三维图像的通道归一化 / Channel-wise normalization of 3D images

在计算机视觉中，我们常需要将彩色图像（Height x Width x Channels）的三个通道分别减去均值。以下例题演示如何利用广播机制一次性完成三个通道的减均值运算。

```python
import numpy as np

# 模拟 224x224 分辨率的 RGB 图像 / Simulate a 224x224 RGB image
# Shape: [224, 224, 3]
image = np.random.rand(224, 224, 3) * 255

# 定义通道均值向量 / Define channel-wise mean
# Shape: [3]
channel_means = np.array([123.68, 116.779, 103.939])

# 利用广播机制对图像进行归一化 / Apply channel-wise subtraction using broadcasting
# channel_means 形状自动扩展为 [1, 1, 3] 以匹配 image [224, 224, 3]
# Time: O(H * W * C), Space: O(H * W * C)
normalized_image = image - channel_means

print(f"原始图像形状 / Original image shape: {image.shape}")
print(f"归一化图像形状 / Normalized image shape: {normalized_image.shape}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：将一维数组 `[1, 2, 3]` 广播加到形状为 `(3, 3)` 的全一矩阵上，使其分别作用在矩阵的每一行上。
*参考答案*：
```python
import numpy as np
# Time: O(N^2), Space: O(N^2)
matrix = np.ones((3, 3))
bias = np.array([1, 2, 3])  # Shape: [3] 一维行向量，沿行广播，每一行都加 [1, 2, 3]
result = matrix + bias
print(result)
```

### 进阶题
**练习 2**：在模型评估中，我们有一个形状为 `(N, D)` 的样本矩阵 `X`，以及形状为 `(K, D)` 的聚类中心矩阵 `C`。编写代码利用广播机制计算所有样本到所有聚类中心的欧氏距离平方矩阵 `D`（形状为 `(N, K)`），禁止使用任何 Python 循环。
*参考答案*：
```python
import numpy as np
# Time: O(N * K * D), Space: O(N * K * D)
N, K, D = 100, 5, 128
X = np.random.randn(N, D)  # Shape: [N, D]
C = np.random.randn(K, D)  # Shape: [K, D]

# 扩展维度进行广播 / Expand dimensions for broadcasting
# X -> Shape: [N, 1, D]
# C -> Shape: [1, K, D]
diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  # Shape: [N, K, D]
dist_squared = np.sum(diff ** 2, axis=-1)  # Shape: [N, K]
print(f"距离矩阵形状: {dist_squared.shape}")
```