# 微积分与优化
# Calculus and Optimization

## 1. 背景（Background）

> **为什么要学这个？**
>
> 梯度下降是训练所有深度学习模型的核心算法。理解偏导数、链式法则、学习率调度——这些是调优模型的必备数学基础。

## 2. 知识点（Key Concepts）

| 概念 | ML 应用 |
|------|---------|
| 偏导数 / 梯度 | 反向传播 |
| 链式法则 | 自动微分 |
| 梯度下降 | 模型训练 |
| Adam 优化器 | 自适应学习率 |
| 学习率调度 | Warmup + Cosine |

## 3. 内容（Content）

### 3.1 梯度与梯度下降

```python
import numpy as np

# ============================================================
# 梯度下降（一维）/ Gradient descent (1D)
# ============================================================
# 目标: 最小化 f(x) = x² + 2x + 1 = (x+1)²
# 导数: f'(x) = 2x + 2
# 最小值: x = -1

def f(x):
    return x**2 + 2*x + 1

def f_grad(x):
    return 2*x + 2

x = 5.0  # 初始值
lr = 0.1
for step in range(20):
    grad = f_grad(x)
    x = x - lr * grad  # 梯度下降更新
    if step % 5 == 0:
        print(f"Step {step}: x={x:.4f}, f(x)={f(x):.6f}")
# x 逐渐收敛到 -1

# ============================================================
# 梯度下降（多维）— 线性回归
# ============================================================
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

w, b = 0.0, 0.0
lr = 0.01

for epoch in range(100):
    pred = w * X + b
    loss = np.mean((pred - y) ** 2)  # MSE
    dw = np.mean(2 * X * (pred - y))  # ∂L/∂w
    db = np.mean(2 * (pred - y))       # ∂L/∂b
    w -= lr * dw
    b -= lr * db

print(f"w={w:.2f}, b={b:.2f}")  # ≈ w=3.0, b=2.0
```

### 3.2 优化器

```
SGD:     w = w - lr × g
Momentum: v = β×v + g; w = w - lr×v
Adam:    m = β₁m + (1-β₁)g       (一阶矩)
         v = β₂v + (1-β₂)g²      (二阶矩)
         w = w - lr × m̂ / (√v̂ + ε)

Adam 为什么好:
  - 自动调整每个参数的学习率
  - 稀疏梯度也能快速收敛
  - 大模型标配优化器
```

### 3.3 学习率调度

```
大模型常用调度策略:

Warmup + Cosine Decay:
  前 N 步: lr 从 0 线性增长到 max_lr（预热）
  之后:    lr 按余弦曲线衰减到 min_lr

为什么需要 Warmup?
  训练初期梯度不稳定 → 大学习率容易发散
  Warmup 让模型先"适应"数据
```

## 4. 详细推理（Deep Dive）

```
链式法则 = 反向传播的数学基础:

y = f(g(h(x)))

∂y/∂x = ∂f/∂g × ∂g/∂h × ∂h/∂x

PyTorch autograd 自动计算这个链式:
  loss.backward() → 自动对所有参数计算梯度
```

## 5-6. 例题/习题

**练习 1：** 手动实现梯度下降求解线性回归。

**练习 2：** 对比 SGD 和 Adam 的收敛速度。

**练习 3：** 实现 Warmup + Cosine Decay 学习率调度器。
