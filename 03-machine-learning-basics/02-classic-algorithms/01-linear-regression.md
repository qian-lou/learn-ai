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

## 5. 例题（Worked Examples）

### 例题 1：使用 Scikit-Learn 训练一元线性回归模型并评估 / Simple Linear Regression with Sklearn

通过该例题熟悉使用 Sklearn 模型库进行训练、计算决定系数 $R^2$ 及特征权重。

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 构造特征和目标 / Construct features and target
# Time: O(N), Space: O(N)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 个样本 / 100 samples
y = 2.5 * X + 1.2 + np.random.randn(100, 1) * 2  # 真值权重 2.5，偏置 1.2

# 2. 拟合模型 / Fit model
# Time: O(N * D^2 + D^3) -> 极快 / Extremely fast.
# Space: O(N * D)
model = LinearRegression()
model.fit(X, y)

# 3. 评估模型 / Evaluate model
preds = model.predict(X)
r2 = r2_score(y, preds)
mse = mean_squared_error(y, preds)

print(f"估计权重 / Coeff W: {model.coef_[0][0]:.4f}")
print(f"估计偏置 / Intercept b: {model.intercept_[0]:.4f}")
print(f"决定系数 / R^2 Score: {r2:.4f}")
print(f"均方误差 / MSE: {mse:.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释线性回归模型中的决定系数 $R^2$ 的物理意义。当 $R^2 = 1.0$ 和 $R^2 = 0.0$ 分别代表什么？
*参考答案*：
$R^2$ 代表响应变量的变异中，能被自变量的回归方程所解释的比例。
- $R^2 = 1.0$：模型完美拟合数据，无任何残差。
- $R^2 = 0.0$：模型的预测效果等同于直接取目标变量的均值。

### 进阶题
**练习 2**：当特征数量 $D$ 远大于样本量 $N$ 时，线性回归会出现严重的过拟合问题。此时应使用何种正则化（惩罚项）方法解决？请使用 Sklearn 编写带有 L1 正则项（Lasso）和 L2 正则项（Ridge）的回归代码。
*参考答案*：
应该使用 Lasso (L1) 获得稀疏解，或 Ridge (L2) 压制权重大小，或 ElasticNet (双结合)。
```python
from sklearn.linear_model import Lasso, Ridge
# Time: O(N * D), Space: O(N * D)
# Lasso 实例
lasso = Lasso(alpha=0.1)
# Ridge 实例
ridge = Ridge(alpha=1.0)
```\n