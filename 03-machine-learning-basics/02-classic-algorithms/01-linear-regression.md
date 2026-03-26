# 线性回归
# Linear Regression

## 1. 背景（Background）

> **为什么要学这个？**
>
> 线性回归是最简单的机器学习算法，也是神经网络的**基本构建块**——全连接层本质就是线性回归 + 激活函数。理解线性回归的数学原理（最小二乘法、梯度下降）是理解深度学习的起点。

## 2. 知识点（Key Concepts）

| 概念 | 公式 | 说明 |
|------|------|------|
| 模型 | y = Wx + b | 线性变换 |
| 损失函数 | MSE = mean((y - ŷ)²) | 均方误差 |
| 解析解 | w = (X^T X)^{-1} X^T y | 正规方程 |
| 数值解 | w -= lr × ∂L/∂w | 梯度下降 |

## 3. 内容（Content）

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. 手动实现 / Manual implementation
# ============================================================
np.random.seed(42)
X = np.random.randn(200, 3)
true_w = np.array([3.0, -1.5, 2.0])
y = X @ true_w + 0.5 + np.random.randn(200) * 0.3

# 正规方程（解析解）
# Time: O(n*d² + d³)  Space: O(d²)
X_bias = np.c_[X, np.ones(200)]
w_analytical = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
print(f"解析解: {w_analytical}")

# ============================================================
# 2. Sklearn 实现 / Sklearn implementation
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"系数: {model.coef_}")
```

## 4. 详细推理（Deep Dive）

```
线性回归 → 神经网络 的演进:

线性回归:     y = Wx + b
Logistic:     y = σ(Wx + b)
单层神经网络:  y = σ(W₂ · σ(W₁x + b₁) + b₂)
深度网络:      y = fₙ(...f₂(f₁(x)))

每一层都是 "线性变换 + 非线性激活"
```

## 5-6. 例题/习题

**练习 1：** 用梯度下降实现线性回归（不用 sklearn）。

**练习 2：** 添加 L2 正则化（Ridge Regression），观察过拟合变化。
