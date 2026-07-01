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

## 5. 例题（Worked Examples）

### 例题 1：从零用梯度下降法寻找多元 Rosenbrock 函数的极小值 / Gradient descent optimization of Rosenbrock function

本例演示梯度下降的核心原理，求解著名的香蕉函数（Rosenbrock function）的极小值，公式为 $f(x, y) = (a-x)^2 + b(y-x^2)^2$。

```python
import numpy as np

# Rosenbrock 函数及其梯度 / Rosenbrock function and its gradient
# Time: O(1), Space: O(1)
def f(x: float, y: float) -> float:
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def grad_f(x: float, y: float) -> np.ndarray:
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    df_dy = 200.0 * (y - x**2)
    return np.array([df_dx, df_dy])

# 梯度下降循环 / Gradient descent loop
# Time: O(Steps), Space: O(1)
lr = 0.001
epochs = 5000
p = np.array([-1.2, 1.0])  # 起始点 / Starting point

for step in range(epochs):
    g = grad_f(p[0], p[1])
    p = p - lr * g  # 权重更新公式 / Parameter update formula
    if step % 1000 == 0:
        print(f"Step {step}: f({p[0]:.4f}, {p[1]:.4f}) = {f(p[0], p[1]):.6f}")

print(f"最终收敛点 / Convergence point: {p}")  # 理论极小值在 (1, 1)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：一元函数 $f(x) = x^2 - 4x + 4$。写出它的导数，并说明导数为 0 处的极值点。
*参考答案*：
导数 $f'(x) = 2x - 4$。令 $2x - 4 = 0$，解得 $x = 2$。因为二阶导数 $f''(x) = 2 > 0$，所以 $x=2$ 是极小值点，极小值为 0。

### 进阶题
**练习 2**：在机器学习反向传播中，已知激活函数为 Sigmoid 且输出为 $a = \sigma(z)$，损失函数为二元交叉熵损失 $L = -[y\ln a + (1-y)\ln(1-a)]$。利用链式法则，推导损失函数对网络输入 $z$ 的偏导数 $rac{\partial L}{\partial z}$，并给出最终化简表达式。
*参考答案*：
根据链式法则：
$rac{\partial L}{\partial z} = rac{\partial L}{\partial a} \cdot rac{\partial a}{\partial z}$
1. 计算第一项：$rac{\partial L}{\partial a} = -rac{y}{a} + rac{1-y}{1-a} = rac{a-y}{a(1-a)}$
2. 计算第二项：Sigmoid 的导数为 $rac{\partial a}{\partial z} = a(1-a)$
3. 相乘化简得：$rac{\partial L}{\partial z} = rac{a-y}{a(1-a)} \cdot a(1-a) = a - y$
结论为极其简洁的 $a - y$（即网络输出值与真实标签的差值，也就是误差项）。
```python
# 该性质使得逻辑回归/神经网络反向传播计算极其高效。
```