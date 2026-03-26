# 逻辑回归 / Logistic Regression

## 1. 背景（Background）
> 逻辑回归是二分类的基础，Sigmoid + 交叉熵就是神经网络分类层的原理。本质是单层神经网络。

## 2-3. 知识点与内容
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 交叉熵损失: L = -[y*log(p) + (1-y)*log(1-p)]
model = LogisticRegression()
model.fit(X_train, y_train)
```

## 4. 详细推理
- 逻辑回归 = 线性回归 + Sigmoid 激活 + 交叉熵损失
- Softmax 是 Sigmoid 的多分类推广：`softmax(x_i) = exp(x_i) / Σexp(x_j)`

## 5-6. 例题/习题
**练习：** 纯 NumPy 实现逻辑回归并画出决策边界。
