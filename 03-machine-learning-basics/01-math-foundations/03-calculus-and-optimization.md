# 微积分与优化（梯度下降）
# Calculus and Optimization (Gradient Descent)

## 1. 背景（Background）

> 神经网络训练的核心就是"梯度下降"——计算损失函数对参数的梯度，然后沿梯度方向更新参数。理解梯度下降是理解模型训练过程的关键。

## 2-3. 知识点与内容

```python
import numpy as np

# 梯度下降直觉 / Gradient descent intuition
# 想象站在山顶，闭眼找最低点：
# 1. 感受脚下坡度（计算梯度）
# 2. 沿着最陡方向走一小步（参数更新）
# 3. 重复直到到达最低点

# 一维梯度下降 / 1D gradient descent
def f(x): return x**2 + 2*x + 1  # 最小值在 x=-1
def df(x): return 2*x + 2         # 导数

x = 10.0  # 初始点
lr = 0.1   # 学习率 (learning rate)
for _ in range(100):
    gradient = df(x)
    x = x - lr * gradient  # 参数更新公式
print(f"最小值点: x = {x:.4f}")  # ≈ -1.0

# 多维梯度下降（线性回归）/ Multi-dim gradient descent
# 损失函数: L = (1/N) * Σ(y_pred - y_true)²
# 梯度: dL/dw = (2/N) * X^T @ (X @ w - y)

# 学习率调度 / Learning rate scheduling
# - 固定学习率
# - Warmup + Cosine Decay（大模型训练标配）
# - 学习率预热：从 0 线性增大到目标 lr
# - 余弦衰减：从目标 lr 按余弦函数衰减到 0
```

## 4. 详细推理

**关键公式：** θ_new = θ_old - lr × ∇L(θ)

**优化器演进：** SGD → Momentum → RMSprop → Adam → AdamW

**对 Java 工程师：** 梯度下降类似于调优 JVM 参数——你有一个"损失"（延迟），通过调整参数找到最优配置。区别是梯度下降是自动化的。

## 5-6. 例题/习题

**练习：** 纯 NumPy 实现线性回归的梯度下降训练，不使用任何 ML 库。
