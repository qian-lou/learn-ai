# 优化算法 / Optimization Algorithms

## 1. 背景（Background）

> **为什么要学这个？**
>
> 如果说反向传播告诉我们"每个参数往哪个方向调整"，那么优化算法决定的是"调整多大步"。优化器是模型训练的"驾驶员"——同样的模型和数据，不同的优化器可能带来天壤之别的训练效果。对于 Java 工程师来说，可以把优化器理解为**自动调参的策略模式（Strategy Pattern）**，不同策略（SGD、Adam、AdamW）对应不同的参数更新策略。
>
> 在大模型时代，**AdamW + Warmup + Cosine Decay** 已成为标准配方。理解优化器的演进历史和原理，能帮你诊断训练不收敛的问题、选择合适的超参数、以及理解为什么大模型训练需要特殊的学习率调度策略。
>
> **在整个体系中的位置：** 优化算法是训练循环的核心组件（前向传播 → 计算损失 → 反向传播 → **优化器更新参数**）。理解优化器演进，才能理解大模型训练中的各种技巧。

## 2. 知识点（Key Concepts）

| 优化器 | 核心思想 | 发布年份 | 主要用途 |
|--------|----------|----------|----------|
| SGD | 最基础，沿梯度方向更新 | - | 教学、简单任务 |
| SGD + Momentum | 引入"惯性"加速收敛 | 1964 | CV 领域仍常用 |
| Adagrad | 自适应学习率（频繁特征学步更小） | 2011 | NLP、稀疏特征 |
| RMSprop | 修复 Adagrad 学习率单调递减问题 | 2012 | 通用 |
| Adam | Momentum + RMSprop 的结合 | 2014 | 通用默认选择 |
| AdamW | 修正 Adam 的权重衰减实现 | 2017 | **大模型训练标配** ✅ |

**核心演进路线：**
```
SGD → Momentum → Adagrad → RMSprop → Adam → AdamW
│      加速收敛    自适应lr    修复衰减    合体      修正衰减
```

**学习率调度（Learning Rate Schedule）：**
- **Warmup**：训练初期逐步增大学习率，避免初始不稳定
- **Cosine Decay**：训练后期按余弦曲线衰减学习率
- **大模型标配：Warmup + Cosine Decay**

## 3. 内容（Content）

### 3.1 SGD 及其变体

```python
import torch
import torch.optim as optim

# ============================================================
# 1. 原始 SGD (Stochastic Gradient Descent)
# Vanilla SGD
# ============================================================
# 更新规则: θ = θ - lr × ∇L
# Update rule: θ = θ - lr × ∇L

# 问题：
# Problems:
# 1. 容易在"峡谷"中震荡（梯度方向不断变化）
# 2. 所有参数使用相同学习率
# 3. 容易卡在鞍点

model = torch.nn.Linear(10, 1)
sgd = optim.SGD(model.parameters(), lr=0.01)


# ============================================================
# 2. SGD + Momentum（动量加速）
# SGD with Momentum
# ============================================================
# 引入"速度"变量: 
#   v = β × v + ∇L         （累积过去的梯度方向）
#   θ = θ - lr × v

# 直觉：像一个小球在损失曲面上滚动，积累动量
# Intuition: Like a ball rolling down the loss surface, accumulating momentum

sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# β = 0.9 是最常用的值
# β 越大，历史梯度的影响越大，"惯性"越强
```

```
Momentum 的效果：

无 Momentum（震荡严重）：     有 Momentum（平滑收敛）：
  ╱╲╱╲╱╲╱╲                    ╲
  ╲╱╲╱╲╱╲╱──→ 最优点           ╲__
                                  ╲___──→ 最优点
```

### 3.2 Adam 与 AdamW

```python
import torch.optim as optim

# ============================================================
# 3. Adam (Adaptive Moment Estimation)
# ============================================================
# 结合 Momentum（一阶矩）和 RMSprop（二阶矩）
# Combines Momentum (1st moment) and RMSprop (2nd moment)

# 更新规则：
# m = β₁·m + (1-β₁)·g          # 一阶矩 / 1st moment (mean of gradients)
# v = β₂·v + (1-β₂)·g²         # 二阶矩 / 2nd moment (variance of gradients)
# m̂ = m / (1-β₁ᵗ)              # 偏差修正 / Bias correction
# v̂ = v / (1-β₂ᵗ)
# θ = θ - lr · m̂ / (√v̂ + ε)    # 自适应更新

adam = optim.Adam(
    model.parameters(),
    lr=1e-3,           # 学习率（默认 1e-3）
    betas=(0.9, 0.999), # β₁, β₂
    eps=1e-8,           # 数值稳定项
    weight_decay=0.0,   # L2 正则化（Adam 实现有问题！）
)

# ============================================================
# 4. AdamW（修正的权重衰减 — 大模型标配！）
# AdamW (Decoupled Weight Decay — LLM standard!)
# ============================================================

# Adam 的 weight_decay 实际是 L2 正则化，混入了自适应学习率
# Adam's weight_decay is actually L2 regularization, mixed with adaptive lr

# AdamW 将权重衰减从梯度更新中解耦：
# AdamW decouples weight decay from gradient update:
# θ = θ - lr · (m̂ / (√v̂ + ε) + λ·θ)
#                                   └── 独立的权重衰减

adamw = optim.AdamW(
    model.parameters(),
    lr=1e-4,             # 大模型通常 1e-4 到 5e-5
    betas=(0.9, 0.999),  # GPT-3: (0.9, 0.95)
    eps=1e-8,
    weight_decay=0.01,   # 真正的权重衰减 ✅
)
```

### 3.3 学习率调度策略

```python
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
)

# ============================================================
# 大模型标配：Warmup + Cosine Decay
# LLM standard: Warmup + Cosine Decay
# ============================================================

model = torch.nn.Linear(768, 768)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

total_steps = 10000     # 总训练步数
warmup_steps = 1000     # Warmup 步数（通常占总步数 5-10%）

# 方法 1：手动组合 LinearLR + CosineAnnealingLR
# Method 1: Combine LinearLR + CosineAnnealingLR
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1e-8 / 1e-4,  # 从接近 0 开始
    end_factor=1.0,             # 升到 lr=1e-4
    total_iters=warmup_steps
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,  # 余下步数做余弦衰减
    eta_min=1e-6,                       # 最终学习率
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# 训练循环中使用 / Usage in training loop
# for step in range(total_steps):
#     loss = train_step()
#     optimizer.step()
#     scheduler.step()


# ============================================================
# 方法 2：OneCycleLR（一种更简洁的替代方案）
# Method 2: OneCycleLR (a simpler alternative)
# ============================================================
onecycle = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,
    pct_start=0.1,  # Warmup 占比 10%
    anneal_strategy='cos',
)
```

```
Warmup + Cosine Decay 学习率曲线：

lr
↑  1e-4 ┌──────╮
│       │       ╲
│       │        ╲
│      ╱│         ╲
│    ╱  │          ╲
│  ╱    │           ╲___
│╱      │                ╲____
0───────┼─────────────────────→ step
    Warmup    Cosine Decay
   (5-10%)     (余下 90-95%)

为什么需要 Warmup？
  训练初期，Adam 的一阶和二阶矩估计不准确（偏差大）
  如果一上来就用大学习率，参数更新量会很大 → 发散
  Warmup 让模型先"热身"，等矩估计稳定后再加大学习率
```

### 3.4 完整优化器对比实验

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 对比不同优化器在同一任务上的收敛速度
# Compare convergence speed of different optimizers
# ============================================================

def train_with_optimizer(optimizer_name: str, lr: float = 0.01, epochs: int = 200):
    """用指定优化器训练模型 / Train model with specified optimizer.
    
    Args:
        optimizer_name: 优化器名称 / Optimizer name.
        lr: 学习率 / Learning rate.
        epochs: 训练轮数 / Number of training epochs.
    
    Returns:
        训练损失列表 / List of training losses.
    """
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    criterion = nn.MSELoss()
    
    # 创建优化器 / Create optimizer
    optimizers = {
        'SGD':      optim.SGD(model.parameters(), lr=lr),
        'Momentum': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'Adam':     optim.Adam(model.parameters(), lr=lr),
        'AdamW':    optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01),
    }
    opt = optimizers[optimizer_name]
    
    # 生成数据 / Generate data
    X = torch.randn(100, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).unsqueeze(1)  # 非线性函数
    
    losses = []
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    return losses

# 运行对比 / Run comparison
for name in ['SGD', 'Momentum', 'Adam', 'AdamW']:
    losses = train_with_optimizer(name, lr=0.01)
    print(f"{name:10s}: 最终损失 = {losses[-1]:.4f}")
```

## 4. 详细推理（Deep Dive）

### 4.1 Adam 的数学推导

```
Adam 算法伪代码（Kingma & Ba, 2014）：

给定：学习率 α, 矩估计衰减率 β₁=0.9, β₂=0.999, ε=1e-8
初始化：m₀ = 0（一阶矩）, v₀ = 0（二阶矩）, t = 0

对每个训练步:
  t = t + 1
  gₜ = ∇L(θₜ₋₁)                    # 计算梯度
  mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ   # 更新一阶矩（梯度均值）
  vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²  # 更新二阶矩（梯度方差）
  m̂ₜ = mₜ / (1 - β₁ᵗ)              # 偏差修正（初始 m₀=0 导致偏差）
  v̂ₜ = vₜ / (1 - β₂ᵗ)
  θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)  # 参数更新

关键洞察：
  - m̂ₜ / √v̂ₜ ≈ 梯度的信噪比（SNR）
  - 梯度方差大 → v̂ₜ 大 → 步长小（谨慎更新）
  - 梯度方差小 → v̂ₜ 小 → 步长大（自信更新）
  - 这就是"自适应学习率"的含义
```

### 4.2 AdamW vs Adam（权重衰减的正确实现）

```
Adam + L2 正则化（错误的方式）：
  gₜ = ∇L(θₜ₋₁) + λ · θₜ₋₁        # L2 梯度混入了原始梯度
  mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ   # 自适应矩估计"吸收"了衰减项
  θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)  # 权重衰减被自适应缩放扭曲

  问题：L2 正则化信号被 Adam 的自适应机制"稀释"了
        高梯度方差的参数，权重衰减效果减弱
        
AdamW（正确的方式）：
  gₜ = ∇L(θₜ₋₁)                     # 只有原始梯度
  mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ   # 矩估计不含衰减项
  θₜ = θₜ₋₁ - α · (m̂ₜ/(√v̂ₜ + ε) + λ·θₜ₋₁)  # 衰减独立应用
                                       └── 解耦的权重衰减

  优势：所有参数受到均匀的权重衰减，不被自适应缩放影响
```

### 4.3 大模型训练的超参数选择

```
GPT-3 训练配置参考（175B 参数）：
  ┌─────────────────────────────────┐
  │ 优化器:     AdamW                │
  │ 学习率:     6e-5                 │
  │ β₁, β₂:    0.9, 0.95            │
  │ ε:         1e-8                  │
  │ 权重衰减:   0.1                  │
  │ Warmup:    375M tokens           │
  │ 衰减策略:   Cosine → 0.1 × lr   │
  │ 梯度裁剪:   max_norm = 1.0      │
  │ Batch size: 3.2M tokens         │
  └─────────────────────────────────┘

LLaMA 训练配置参考（7B-65B 参数）：
  ┌─────────────────────────────────┐
  │ 优化器:     AdamW                │
  │ 学习率:     3e-4（7B）, 1.5e-4（65B）│
  │ β₁, β₂:    0.9, 0.95            │
  │ 权重衰减:   0.1                  │
  │ Warmup:    2000 steps            │
  │ 衰减策略:   Cosine → lr/10      │
  │ 梯度裁剪:   max_norm = 1.0      │
  └─────────────────────────────────┘
```

## 5. 例题（Worked Examples）

### 例题 1：手算一步 SGD 更新

**问题：** 参数 θ = [2.0, -1.0]，梯度 g = [0.5, -0.3]，学习率 lr = 0.1，计算一步 SGD 更新。

**解答：**
```python
import numpy as np

theta = np.array([2.0, -1.0])
grad = np.array([0.5, -0.3])
lr = 0.1

# SGD: θ_new = θ - lr × g
theta_new = theta - lr * grad
print(f"θ_old = {theta}")
print(f"θ_new = {theta_new}")  # [1.95, -0.97]
# θ₁: 2.0 - 0.1 × 0.5 = 1.95
# θ₂: -1.0 - 0.1 × (-0.3) = -0.97
```

### 例题 2：对比不同优化器的收敛行为

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 在 Rosenbrock 函数上对比优化器
# Compare optimizers on Rosenbrock function
# f(x,y) = (1-x)² + 100(y-x²)²
# 最优解: (1, 1), 最优值: 0
# ============================================================

def rosenbrock(params):
    """Rosenbrock 函数（优化器的经典测试函数）."""
    x, y = params[0], params[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

results = {}
for opt_name, opt_class, lr in [
    ('SGD',      torch.optim.SGD,   1e-4),
    ('Momentum', torch.optim.SGD,   1e-4),
    ('Adam',     torch.optim.Adam,  1e-2),
    ('AdamW',    torch.optim.AdamW, 1e-2),
]:
    params = torch.tensor([-1.0, 1.0], requires_grad=True)
    kwargs = {'lr': lr}
    if opt_name == 'Momentum':
        kwargs['momentum'] = 0.9
    
    opt = opt_class([params], **kwargs)
    trajectory = [params.detach().clone().numpy()]
    
    for step in range(2000):
        loss = rosenbrock(params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trajectory.append(params.detach().clone().numpy())
    
    results[opt_name] = {
        'final': params.detach().numpy(),
        'loss': rosenbrock(params).item(),
    }
    print(f"{opt_name:10s}: θ = {results[opt_name]['final']}, loss = {results[opt_name]['loss']:.6f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 解释 SGD 的 `momentum=0.9` 参数的含义。为什么一般设为 0.9 而不是 0.5 或 0.99？

> **提示：** 0.9 意味着约保留过去 10 步梯度的指数加权平均。太小（0.5）惯性不够，太大（0.99）可能越过最优点。

**练习 2：** Adam 的默认超参数是 `lr=1e-3, betas=(0.9, 0.999), eps=1e-8`。解释每个参数的作用以及为什么选择这些默认值。

### 进阶题

**练习 3：** 实现一个简单的学习率 Warmup 调度器：在前 `warmup_steps` 步内，学习率从 0 线性增长到 `target_lr`。

> **参考答案：**
> ```python
> class WarmupScheduler:
>     """线性 Warmup 调度器 / Linear warmup scheduler."""
> 
>     def __init__(self, optimizer, warmup_steps: int, target_lr: float):
>         self.optimizer = optimizer
>         self.warmup_steps = warmup_steps
>         self.target_lr = target_lr
>         self.step_count = 0
> 
>     def step(self):
>         self.step_count += 1
>         if self.step_count <= self.warmup_steps:
>             lr = self.target_lr * (self.step_count / self.warmup_steps)
>             for pg in self.optimizer.param_groups:
>                 pg['lr'] = lr
> ```

**练习 4：** 为什么大模型训练使用 AdamW 而非 SGD + Momentum？从以下角度分析：
1. 参数规模（百亿级参数的梯度特征）
2. 训练稳定性
3. 超参数调节难度

**练习 5：** 一个 7B 参数的模型使用 AdamW 训练，需要存储 m（一阶矩）和 v（二阶矩）。如果模型参数占用 14GB（FP16），那么优化器状态额外占用多少 GPU 显存？

> **参考答案：** 
> - m 和 v 通常使用 FP32 存储：7B × 4 bytes × 2 = **56 GB**
> - 加上模型参数本身的 FP32 副本：7B × 4 bytes = 28 GB
> - 总优化器显存开销：**约 84 GB**（这就是为什么需要 ZeRO 等显存优化技术）
