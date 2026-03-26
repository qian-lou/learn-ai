# 激活函数 / Activation Functions

## 1. 背景（Background）

> **为什么要学这个？**
>
> 激活函数是神经网络能够拟合非线性关系的**关键要素**。没有激活函数，无论多少层的神经网络都只能做线性变换。对于 Java 工程师来说，可以把激活函数理解为数据处理管道中的一个**变换器（Transformer pattern）**——它不改变数据结构，但会对值进行非线性映射。
>
> 激活函数的选择直接影响模型训练效果。从早期的 Sigmoid/Tanh，到 ReLU 革命性地解决了梯度消失问题，再到大模型时代的 GELU/SiLU（Swish），激活函数的演进推动了深度学习的每一步发展。
>
> **在整个体系中的位置：** 激活函数是 MLP 的核心组件。理解不同激活函数的特性，才能理解为什么 GPT 用 GELU、LLaMA 用 SiLU，以及 Softmax 在 Attention 中的作用。

## 2. 知识点（Key Concepts）

| 激活函数 | 公式 | 使用场景 | 时代 |
|----------|------|----------|------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | 早期网络、二分类输出 | 1990s |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 早期 RNN | 1990s |
| ReLU | max(0, x) | CNN、通用默认 | 2012+ |
| Leaky ReLU | max(αx, x), α=0.01 | 解决 ReLU 死亡问题 | 2013+ |
| GELU | x·Φ(x) | GPT, BERT, ViT | 2016+ |
| SiLU/Swish | x·σ(x) | LLaMA, Mistral | 2017+ |
| Softmax | eˣⁱ/Σeˣʲ | 多分类输出、Attention | 通用 |

**核心要点：**
- **无激活函数 = 只有线性变换**，多层等于一层
- ReLU 解决了梯度消失，但存在"神经元死亡"问题
- GELU/SiLU 更平滑，允许少量负值通过，大模型效果更好
- Softmax 将任意实数向量转为概率分布，温度参数 T 控制分布锐利度

## 3. 内容（Content）

### 3.1 经典激活函数详解

```python
import torch
import torch.nn.functional as F
import numpy as np

# ============================================================
# 1. Sigmoid：最早的激活函数
# Sigmoid: The earliest activation function
# ============================================================
def sigmoid(x):
    """σ(x) = 1 / (1 + e^(-x))
    
    输出范围 / Output range: (0, 1)
    导数 / Derivative: σ'(x) = σ(x) · (1 - σ(x))
    """
    return 1 / (1 + np.exp(-x))

# 问题：梯度消失！当 |x| 很大时，导数接近 0
# Problem: Vanishing gradient! When |x| is large, derivative → 0
# σ'(0) = 0.25（最大值才 0.25），10层网络：0.25^10 ≈ 0.000001
# 10 layers: 0.25^10 ≈ 0.000001 (gradient nearly vanishes)


# ============================================================
# 2. Tanh：Sigmoid 的改进版
# Tanh: Improved version of Sigmoid
# ============================================================
def tanh(x):
    """tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    输出范围 / Output range: (-1, 1) — 零中心化
    导数 / Derivative: tanh'(x) = 1 - tanh²(x)
    """
    return np.tanh(x)

# 优点：零中心化，梯度更大（最大值 1.0 vs Sigmoid 的 0.25）
# Advantage: Zero-centered, larger gradient (max 1.0 vs Sigmoid's 0.25)
# 缺点：仍然存在梯度消失问题
# Disadvantage: Still has vanishing gradient problem


# ============================================================
# 3. ReLU：深度学习革命的关键
# ReLU: Key to the deep learning revolution
# ============================================================
def relu(x):
    """ReLU(x) = max(0, x)
    
    输出范围 / Output range: [0, +∞)
    导数 / Derivative: 1 if x > 0, else 0
    """
    return np.maximum(0, x)

# 优势：
# Advantages:
# 1. 计算简单：只需比较和取最大值（比 exp() 快 6x）
# 2. 缓解梯度消失：正半轴梯度恒为 1
# 3. 稀疏激活：约 50% 神经元输出为 0 → 节省计算

# 缺陷：神经元死亡（Dead ReLU）
# Flaw: Dead ReLU problem
# 如果 x < 0，梯度为 0，权重永远不更新 → 神经元"死了"
# If x < 0, gradient is 0, weights never update → neuron "dies"


# ============================================================
# 4. Leaky ReLU：解决死亡问题
# Leaky ReLU: Solving the dead neuron problem
# ============================================================
def leaky_relu(x, alpha=0.01):
    """LeakyReLU(x) = max(αx, x)
    
    α 通常 = 0.01，让负半轴也有微小梯度
    """
    return np.where(x > 0, x, alpha * x)
```

### 3.2 现代激活函数（大模型核心）

```python
import torch
import torch.nn.functional as F

# ============================================================
# 5. GELU：GPT / BERT / ViT 的选择
# GELU: Used by GPT / BERT / ViT
# ============================================================
# GELU(x) = x · Φ(x)
# 其中 Φ(x) 是标准正态分布的 CDF
# Where Φ(x) is the CDF of standard normal distribution

x = torch.linspace(-4, 4, 200)
gelu_out = F.gelu(x)

# 近似公式（用于快速计算）:
# Approximate formula (for fast computation):
# GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

# 直觉理解：
# Intuitive understanding:
# - x 很大（正）→ GELU(x) ≈ x（类似 ReLU）
# - x 很小（负）→ GELU(x) ≈ 0（类似 ReLU）
# - x 在 0 附近 → 平滑过渡，允许少量负值通过
# - 不像 ReLU 那样在 x=0 有"硬拐点"


# ============================================================
# 6. SiLU / Swish：LLaMA / Mistral 的选择
# SiLU / Swish: Used by LLaMA / Mistral
# ============================================================
# SiLU(x) = x · σ(x) = x · sigmoid(x)

silu_out = F.silu(x)

# SiLU 与 GELU 非常相似，但：
# SiLU is very similar to GELU, but:
# - 没有异常的"驼峰"形状
# - 实验中 LLaMA 发现 SiLU 效果略好
# - 计算更简单（只需 sigmoid，不需要 CDF 近似）


# ============================================================
# 对比所有激活函数
# Compare all activation functions
# ============================================================
print("主要激活函数在 x=[-2, -1, 0, 1, 2] 处的值：")
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

activations = {
    'ReLU':  F.relu(test_x),
    'GELU':  F.gelu(test_x),
    'SiLU':  F.silu(test_x),
    'Tanh':  torch.tanh(test_x),
}

for name, values in activations.items():
    print(f"  {name:10s}: {[f'{v:.3f}' for v in values.tolist()]}")
```

### 3.3 Softmax：从 logits 到概率

```python
import torch
import torch.nn.functional as F

# ============================================================
# Softmax：多分类 + Attention 的核心
# Softmax: Core of multi-class + Attention
# ============================================================

# Softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
# 将任意实数向量转为概率分布（和为 1）
# Converts any real vector to probability distribution (sum = 1)

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"Logits: {logits.tolist()}")
print(f"Probs:  {[f'{p:.3f}' for p in probs.tolist()]}")
# [0.659, 0.243, 0.099] — 最大值对应最高概率

# ============================================================
# 温度参数 T：控制分布锐利度
# Temperature T: Controls distribution sharpness
# ============================================================
# softmax(x/T)
# T → 0: 退化为 argmax（只保留最大值）
# T = 1: 标准 softmax
# T → ∞: 均匀分布

for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    scaled_probs = F.softmax(logits / T, dim=0)
    print(f"T={T:>4}: {[f'{p:.3f}' for p in scaled_probs.tolist()]}")

# T=0.1: [1.000, 0.000, 0.000]  ← 几乎 one-hot
# T=1.0: [0.659, 0.243, 0.099]  ← 标准
# T=10:  [0.354, 0.336, 0.309]  ← 接近均匀

# 在大模型推理中：
# In LLM inference:
# - T < 1：更确定的回答（代码生成）
# - T > 1：更多样的回答（创意写作）
```

### 3.4 PyTorch 中激活函数的使用方式

```python
import torch.nn as nn

# ============================================================
# 方式 1：作为 nn.Module（推荐用于 Sequential）
# Method 1: As nn.Module (recommended for Sequential)
# ============================================================
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.GELU(),              # 作为层使用 / Used as a layer
    nn.Linear(256, 10),
)

# ============================================================
# 方式 2：作为函数调用（推荐用于 forward()）
# Method 2: As function call (recommended in forward())
# ============================================================
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))  # 函数式调用 / Functional call
        return self.fc2(x)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 GELU 比 ReLU 更适合大模型？

```
ReLU vs GELU 的本质差异：

ReLU: 硬阈值门控
  ┌──────────────────────
  │  对于每个输入 x:
  │  - x > 0: 完全通过（门 = 1）
  │  - x < 0: 完全阻断（门 = 0）
  │  - x = 0: 不可导（硬拐点）
  └──────────────────────

GELU: 软概率门控
  ┌──────────────────────
  │  对于每个输入 x:
  │  - 通过的概率 = Φ(x)（标准正态 CDF）
  │  - x 很大 → 几乎确定通过
  │  - x 很小 → 几乎确定阻断
  │  - x 在 0 附近 → 有一定概率通过
  └──────────────────────

为什么大模型更喜欢 GELU？
1. 平滑性：梯度处处连续，优化更稳定
2. 概率性：允许"微弱信号"部分通过，而非完全丢弃
3. 自正则化：类似 Dropout 的随机效果
4. 实验验证：BERT / GPT-2 / GPT-3 / ViT 均使用 GELU 取得 SOTA
```

### 4.2 梯度消失与梯度爆炸

```
梯度消失问题（以 Sigmoid 为例）：

反向传播中，梯度 = 各层导数的连乘：
  ∂L/∂w₁ = ∂L/∂aₙ · ∂aₙ/∂zₙ · ∂zₙ/∂aₙ₋₁ · ... · ∂z₁/∂w₁

Sigmoid 导数: σ'(x) = σ(x)(1-σ(x)) ∈ (0, 0.25]

假设 10 层网络，每层导数最大 0.25：
  梯度衰减 ≤ 0.25^10 = 9.5 × 10⁻⁷

结果：前面的层几乎学不到东西！底层参数"冻结"了。

ReLU 如何缓解：
  ReLU 导数: 1 (x>0) 或 0 (x<0)
  正半轴梯度恒为 1 → 不衰减！
  10层：1^10 = 1（完美传递梯度）
  
  代价：负半轴梯度为 0 → 神经元可能"死亡"
```

### 4.3 数值稳定性：Log-Softmax

```python
# ============================================================
# Softmax 的数值稳定性问题和解决方案
# Numerical stability issues with Softmax
# ============================================================

# 直接计算 softmax 可能溢出
# Direct softmax can overflow
logits = torch.tensor([1000.0, 1001.0, 1002.0])
# e^1000 = Inf！直接计算会 NaN
# e^1000 = Inf! Direct computation gives NaN

# 解决方案：减去最大值（数学等价）
# Solution: Subtract max (mathematically equivalent)
# softmax(x) = softmax(x - max(x))
safe_logits = logits - logits.max()  # [0, 1, 2] — 安全范围
probs = F.softmax(safe_logits, dim=0)

# PyTorch 内部已处理，直接用即可
# PyTorch handles this internally
probs = F.softmax(logits, dim=0)  # 内部自动减去 max

# 在训练中，使用 log_softmax + NLLLoss 更稳定
# In training, log_softmax + NLLLoss is more stable
# 等价于 CrossEntropyLoss（推荐）
# Equivalent to CrossEntropyLoss (recommended)
loss_fn = nn.CrossEntropyLoss()  # 内部 = log_softmax + NLLLoss
```

## 5. 例题（Worked Examples）

### 例题 1：手算 Softmax

**问题：** 给定 logits = [3.0, 1.0, 0.2]，手算 softmax 输出。

**解答：**

```python
import numpy as np

logits = np.array([3.0, 1.0, 0.2])

# 步骤 1: 计算 e^x
exp_vals = np.exp(logits)  # [20.086, 2.718, 1.221]

# 步骤 2: 求和
total = np.sum(exp_vals)   # 24.025

# 步骤 3: 归一化
probs = exp_vals / total   # [0.836, 0.113, 0.051]

print(f"e^x = {exp_vals}")
print(f"sum = {total:.3f}")
print(f"softmax = {probs}")
print(f"验证和 = {probs.sum():.6f}")  # 1.000000
```

### 例题 2：可视化激活函数及其导数

```python
import torch
import matplotlib.pyplot as plt

# ============================================================
# 绘制 6 种激活函数及其导数
# Plot 6 activation functions and their derivatives
# ============================================================
x = torch.linspace(-5, 5, 500, requires_grad=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('激活函数及其导数 / Activation Functions and Derivatives', fontsize=14)

activations = [
    ('Sigmoid', torch.sigmoid),
    ('Tanh', torch.tanh),
    ('ReLU', F.relu),
    ('LeakyReLU', lambda x: F.leaky_relu(x, 0.1)),
    ('GELU', F.gelu),
    ('SiLU', F.silu),
]

for idx, (name, fn) in enumerate(activations):
    ax = axes[idx // 3, idx % 3]
    
    # 计算函数值 / Compute function values
    x_grad = x.clone().requires_grad_(True)
    y = fn(x_grad)
    
    # 计算导数 / Compute derivatives
    y.sum().backward()
    grad = x_grad.grad.detach()
    
    ax.plot(x.numpy(), y.detach().numpy(), 'b-', label=f'{name}(x)', linewidth=2)
    ax.plot(x.numpy(), grad.numpy(), 'r--', label=f"{name}'(x)", linewidth=1.5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 手算以下 Sigmoid 值：σ(0), σ(1), σ(-1), σ(10), σ(-10)。观察输出范围和饱和现象。

> **参考答案：**
> σ(0)=0.5, σ(1)≈0.731, σ(-1)≈0.269, σ(10)≈0.99995, σ(-10)≈0.00005

**练习 2：** 在一个 5 层全连接网络中分别使用 Sigmoid 和 ReLU, 训练相同的数据集。记录每一层的梯度大小，验证 Sigmoid 的梯度消失问题。

### 进阶题

**练习 3：** 实现一个自定义的 Swish（SiLU）激活函数，支持可学习的 β 参数：
```
Swish_β(x) = x · σ(βx)
```
当 β=1 时退化为标准 SiLU，当 β→∞ 时退化为 ReLU。

> **参考答案：**
> ```python
> class ParametricSwish(nn.Module):
>     """可学习 β 的 Swish 激活 / Swish with learnable β."""
> 
>     def __init__(self, init_beta: float = 1.0):
>         super().__init__()
>         self.beta = nn.Parameter(torch.tensor(init_beta))
> 
>     def forward(self, x: torch.Tensor) -> torch.Tensor:
>         return x * torch.sigmoid(self.beta * x)
> ```

**练习 4：** 解释为什么 Softmax 的温度参数 T 可以控制大模型输出的"创造性"。当 T=0.1 和 T=2.0 时，模型生成文本的行为有什么区别？

> **提示：** T 越低分布越"尖锐"（top-1 概率接近 1），模型倾向于选择最可能的 token；T 越高分布越"平坦"，低概率 token 也有机会被选中。
