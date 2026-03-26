# 感知机与多层感知机 / Perceptron and MLP

## 1. 背景（Background）

> **为什么要学这个？**
>
> 感知机（Perceptron）是所有神经网络的"原子单元"。理解了它，才能理解后续的 CNN、RNN 和 Transformer。对于 Java 工程师来说，可以把感知机类比为一个**简单的分类器函数**——输入若干特征，通过加权求和后输出判断结果。
>
> 多层感知机（MLP）则是感知机的"堆叠升级版"，通过引入隐藏层和非线性激活函数，获得了拟合任意复杂函数的能力。值得注意的是，**Transformer 中的 Feed-Forward Network (FFN) 本质上就是一个两层 MLP**，所以理解 MLP 是理解大模型的必经之路。
>
> **在整个体系中的位置：** 这是深度学习的第一课。感知机 → MLP → CNN/RNN → Transformer → LLM，每一步都建立在前一步的基础之上。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | Java 类比 |
|------|------|-----------|
| 感知机 (Perceptron) | 单层线性分类器 | 一个 `if-else` 条件判断 |
| 权重 (Weights) | 每个输入特征的重要程度 | 方法参数的"优先级系数" |
| 偏置 (Bias) | 决策边界的偏移量 | `if (score > threshold)` 中的 `threshold` |
| 激活函数 (Activation) | 引入非线性的函数 | 对输出做一次变换处理 |
| 多层感知机 (MLP) | 多个感知机层堆叠 | 多层 Service 处理链 |
| 前向传播 (Forward Pass) | 输入→隐藏层→输出的计算过程 | 一次请求经过各层处理 |
| 通用近似定理 | 一个隐藏层即可近似任意连续函数 | 理论上一层就够，但实际多层更高效 |

**核心要点：**
- 单层感知机**只能解决线性可分**问题（无法解决 XOR）
- MLP 通过隐藏层 + 非线性激活函数突破线性限制
- **通用近似定理**保证了 MLP 的理论表达能力
- 大模型中的 FFN 层就是一个 MLP：`FFN(x) = GELU(xW₁ + b₁)W₂ + b₂`

## 3. 内容（Content）

### 3.1 感知机的数学模型

```
感知机结构：
                      ┌─── w₁ ─── x₁
                      │
输出 y = f(Σ) ← Σ ──┼─── w₂ ─── x₂
                      │
                      ├─── w₃ ─── x₃
                      │
                      └─── b（偏置）

计算过程：
  z = w₁·x₁ + w₂·x₂ + w₃·x₃ + b    （加权求和）
  y = f(z)                             （激活函数）
```

```python
import numpy as np

# ============================================================
# 从零实现感知机（不使用框架）
# Implement perceptron from scratch (no framework)
# Time: O(N*D) per epoch, Space: O(D)
# N = 样本数, D = 特征维度
# ============================================================
class Perceptron:
    """感知机分类器 / Perceptron classifier.
    
    Args:
        n_features: 输入特征数 / Number of input features.
        learning_rate: 学习率 / Learning rate.
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        # 初始化权重为零（Java 对比：成员变量初始化）
        # Initialize weights to zero
        self.weights = np.zeros(n_features)  # Shape: [D]
        self.bias = 0.0
        self.lr = learning_rate
    
    def predict(self, x: np.ndarray) -> int:
        """前向传播 / Forward pass.
        
        Args:
            x: 输入特征向量 / Input feature vector. Shape: [D]
        
        Returns:
            预测类别 (0 或 1) / Predicted class (0 or 1).
        """
        # z = w·x + b（加权求和）
        z = np.dot(self.weights, x) + self.bias
        # 阶跃函数：z >= 0 → 1, 否则 → 0
        # Step function: z >= 0 → 1, else → 0
        return 1 if z >= 0 else 0
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """训练感知机 / Train the perceptron.
        
        Args:
            X: 训练数据 / Training data. Shape: [N, D]
            y: 标签 / Labels. Shape: [N]
            epochs: 训练轮数 / Number of training epochs.
        """
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                # 感知机学习规则：w = w + lr * (y - ŷ) * x
                # Perceptron learning rule
                error = yi - prediction
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"第 {epoch+1} 轮收敛 / Converged at epoch {epoch+1}")
                break
```

### 3.2 感知机的局限：XOR 问题

```python
# ============================================================
# XOR 问题：感知机无法解决
# XOR problem: Perceptron cannot solve this
# ============================================================
import numpy as np

# XOR 数据集
# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# 单层感知机尝试学习 XOR（会失败）
# Single perceptron trying to learn XOR (will fail)
p = Perceptron(n_features=2, learning_rate=0.1)
p.train(X_xor, y_xor, epochs=1000)

for xi, yi in zip(X_xor, y_xor):
    pred = p.predict(xi)
    print(f"输入: {xi}, 期望: {yi}, 预测: {pred}, {'✅' if pred == yi else '❌'}")
# 至少有一个会错！因为 XOR 不是线性可分的
# At least one will be wrong! XOR is not linearly separable
```

```
XOR 为何不可分？
              x₂
              │
  (0,1)=1  ● │ ● (1,1)=0
              │
  ────────────┼────────── x₁
              │
  (0,0)=0  ○ │ ○ (1,0)=1
              │

无法用一条直线将 ● 和 ○ 分开！
A single line cannot separate ● from ○!
```

### 3.3 多层感知机（MLP）突破线性限制

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch 实现 MLP
# MLP implementation with PyTorch
# ============================================================
class MLP(nn.Module):
    """多层感知机 / Multi-Layer Perceptron.
    
    Args:
        input_dim: 输入维度 / Input dimension.
        hidden_dim: 隐藏层维度 / Hidden layer dimension.
        output_dim: 输出维度 / Output dimension.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # 两层全连接网络（类比 Java 的两层 Service 调用链）
        # Two-layer fully connected network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # Shape: [B, input] -> [B, hidden]
            nn.ReLU(),                          # 非线性激活 / Non-linear activation
            nn.Linear(hidden_dim, output_dim),  # Shape: [B, hidden] -> [B, output]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward pass.
        
        Args:
            x: 输入张量 / Input tensor. Shape: [B, input_dim]
        
        Returns:
            输出张量 / Output tensor. Shape: [B, output_dim]
        """
        return self.layers(x)


# ============================================================
# 用 MLP 解决 XOR 问题
# Solving XOR with MLP
# ============================================================
# 准备数据 / Prepare data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 创建模型: 2 -> 4 -> 1
# Create model: 2 -> 4 -> 1
model = MLP(input_dim=2, hidden_dim=4, output_dim=1)
criterion = nn.BCEWithLogitsLoss()  # 二分类损失 / Binary cross-entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 训练循环 / Training loop
for epoch in range(1000):
    logits = model(X)               # Shape: [4, 1]
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc.item():.2f}")

# 最终预测 / Final prediction
with torch.no_grad():
    preds = (torch.sigmoid(model(X)) > 0.5).int()
    print(f"XOR 预测结果 / XOR predictions: {preds.squeeze().tolist()}")
    # 输出: [0, 1, 1, 0] ✅
```

### 3.4 MLP 的结构变体

```
常见 MLP 结构（以分类任务为例）：

1) 浅层宽网络：          2) 深层窄网络：          3) 大模型 FFN：
   Input [784]              Input [784]              Input [768]
      │                        │                        │
   Hidden [1024]            Hidden [256]             Hidden [3072]  ← 4x 扩展
      │                        │                        │
   Output [10]              Hidden [128]             GELU 激活
                               │                        │
                            Hidden [64]              Output [768]
                               │
                            Output [10]

Transformer FFN 公式：
  FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
  
  其中 W₁: [768, 3072], W₂: [3072, 768]
  隐藏层维度通常是嵌入维度的 4 倍
```

## 4. 详细推理（Deep Dive）

### 4.1 通用近似定理（Universal Approximation Theorem）

```
定理（Cybenko, 1989）：
  对于任何连续函数 f: [0,1]^n → R 和 ε > 0，
  存在一个单隐藏层 MLP：
    F(x) = Σᵢ αᵢ · σ(wᵢ · x + bᵢ)
  使得 |F(x) - f(x)| < ε 对所有 x ∈ [0,1]^n 成立。

直觉理解：
  ┌─────────────────────────────────────┐
  │ 每个隐藏神经元 = 一个"阶梯函数"     │
  │ 多个阶梯 = 可以拼出任意形状的曲线   │
  │ 隐藏神经元越多 → 阶梯越细 → 逼近越精确│
  └─────────────────────────────────────┘
  
  类比：傅里叶级数用正弦波拼出任意周期函数
        MLP 用激活函数拼出任意连续函数
```

**为什么还需要深层网络？**
- 通用近似定理说"一层就够"，但没说需要多少个神经元
- 浅层宽网络可能需要**指数级**的神经元
- 深层网络可以用**多项式级**的参数学到等价的表示
- 示例：表达 `x₁ XOR x₂ XOR ... XOR xₙ` ，浅层需要 O(2ⁿ) 个神经元，深层只需 O(n)

### 4.2 MLP 的前向传播数学推导

```
给定 L 层 MLP，第 l 层的计算：

  z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾    （线性变换）
  a⁽ˡ⁾ = f(z⁽ˡ⁾)                    （激活函数）

其中：
  W⁽ˡ⁾: 权重矩阵，Shape [dₗ, dₗ₋₁]
  b⁽ˡ⁾: 偏置向量，Shape [dₗ]
  a⁽⁰⁾ = x（输入）
  a⁽ᴸ⁾ = ŷ（输出）

参数总量: Σₗ (dₗ × dₗ₋₁ + dₗ)
例如 [784, 256, 128, 10] 的 MLP:
  784×256 + 256 + 256×128 + 128 + 128×10 + 10 = 235,146 参数
```

### 4.3 从 MLP 到 Transformer FFN

```python
# ============================================================
# Transformer 中的 FFN 就是 MLP
# The FFN in Transformer IS an MLP
# ============================================================
class TransformerFFN(nn.Module):
    """Transformer 前馈网络 / Transformer Feed-Forward Network.
    
    标准结构：Linear → GELU → Linear
    Standard: Linear → GELU → Linear
    """
    
    def __init__(self, d_model: int = 768, d_ff: int = 3072):
        super().__init__()
        # d_ff 通常 = 4 × d_model
        self.w1 = nn.Linear(d_model, d_ff)    # Shape: [B, S, 768] -> [B, S, 3072]
        self.w2 = nn.Linear(d_ff, d_model)    # Shape: [B, S, 3072] -> [B, S, 768]
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        
        Args:
            x: 输入 / Input. Shape: [B, S, d_model]
        
        Returns:
            输出 / Output. Shape: [B, S, d_model]
        """
        return self.w2(self.gelu(self.w1(x)))
```

## 5. 例题（Worked Examples）

### 例题 1：手算两层感知机的前向传播

**问题：** 给定以下两层网络，计算输入 x = [1, 2] 的输出。

```
权重和偏置：
  W1 = [[0.5, -0.3],     b1 = [0.1, -0.2]
        [0.8,  0.2]]
  W2 = [0.6, 0.9]        b2 = 0.1
  
  激活函数：ReLU
```

**解答：**

```python
import numpy as np

# 输入 / Input
x = np.array([1.0, 2.0])

# 第一层：线性变换 + 激活
# Layer 1: Linear + Activation
W1 = np.array([[0.5, -0.3], [0.8, 0.2]])
b1 = np.array([0.1, -0.2])
z1 = W1 @ x + b1           # [0.5*1 + (-0.3)*2 + 0.1, 0.8*1 + 0.2*2 + (-0.2)]
                             # = [0.5 - 0.6 + 0.1, 0.8 + 0.4 - 0.2]
                             # = [0.0, 1.0]
a1 = np.maximum(0, z1)      # ReLU: [0.0, 1.0]

# 第二层：线性变换 / Layer 2: Linear
W2 = np.array([0.6, 0.9])
b2 = 0.1
output = W2 @ a1 + b2       # 0.6*0.0 + 0.9*1.0 + 0.1 = 1.0

print(f"z1 = {z1}")         # [0.0, 1.0]
print(f"a1 = {a1}")         # [0.0, 1.0]
print(f"output = {output}") # 1.0
```

### 例题 2：PyTorch 构建不同深度的 MLP 并对比

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 对比浅层宽网络 vs 深层窄网络
# Comparing shallow-wide vs deep-narrow networks
# ============================================================

# 生成非线性数据（同心圆）
# Generate nonlinear data (concentric circles)
from sklearn.datasets import make_circles
X_np, y_np = make_circles(n_samples=500, noise=0.1, factor=0.5)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# 浅层宽网络: 2 → 64 → 1 (参数量: 2*64+64 + 64*1+1 = 257)
# Shallow-wide: 2 → 64 → 1
shallow = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1)
)

# 深层窄网络: 2 → 8 → 8 → 8 → 1 (参数量: 2*8+8 + 8*8+8 + 8*8+8 + 8*1+1 = 177)
# Deep-narrow: 2 → 8 → 8 → 8 → 1
deep = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1)
)

print(f"浅层参数量 / Shallow params: {sum(p.numel() for p in shallow.parameters())}")
print(f"深层参数量 / Deep params: {sum(p.numel() for p in deep.parameters())}")
# 深层用更少的参数也能学到好的表示！
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用纯 NumPy（不使用 PyTorch）实现一个 `Perceptron` 类，完成 AND 门和 OR 门的学习。验证感知机能学会这两个逻辑门。

**练习 2：** 解释为什么 MLP 需要非线性激活函数。如果去掉激活函数（或者使用线性激活 `f(x) = x`），多层 MLP 等价于什么？

> **提示：** 两个线性变换的组合 `W₂(W₁x + b₁) + b₂ = W₂W₁x + W₂b₁ + b₂ = W'x + b'` 仍然是线性变换。

### 进阶题

**练习 3：** 用 PyTorch 构建一个 3 层 MLP（2 → 16 → 8 → 1），训练它解决 XOR 问题。记录训练过程中 loss 的变化，观察收敛行为。

**练习 4：** 修改 `TransformerFFN` 类，实现 **GLU (Gated Linear Unit)** 变体：
```
FFN_GLU(x) = (xW₁ ⊙ GELU(xW_gate)) W₂
```
这是 LLaMA 等现代大模型使用的 FFN 结构。

> **参考答案：**
> ```python
> class GLU_FFN(nn.Module):
>     """Gated Linear Unit FFN (LLaMA-style)."""
>     
>     def __init__(self, d_model: int = 768, d_ff: int = 3072):
>         super().__init__()
>         self.w1 = nn.Linear(d_model, d_ff, bias=False)
>         self.w_gate = nn.Linear(d_model, d_ff, bias=False)
>         self.w2 = nn.Linear(d_ff, d_model, bias=False)
>         self.gelu = nn.GELU()
>     
>     def forward(self, x: torch.Tensor) -> torch.Tensor:
>         # Shape: [B, S, d_model] -> [B, S, d_ff] -> [B, S, d_model]
>         return self.w2(self.w1(x) * self.gelu(self.w_gate(x)))
> ```

**练习 5：** 一个 MLP 结构为 [784, 512, 256, 128, 10]，请计算：
1. 总参数量（含偏置）
2. 单次前向传播的浮点运算数（FLOPs，只算乘法）

> **参考答案：**
> 1. 参数量: 784×512+512 + 512×256+256 + 256×128+128 + 128×10+10 = **534,794**
> 2. FLOPs: 784×512 + 512×256 + 256×128 + 128×10 = **565,248**
