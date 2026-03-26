# 逻辑回归
# Logistic Regression

## 1. 背景（Background）

> **为什么要学这个？**
>
> 逻辑回归是最基础的**分类算法**，也是神经网络输出层的理论基础。Softmax + 交叉熵损失的组合直接来源于逻辑回归的推广。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| Sigmoid | σ(z) = 1/(1+e^{-z})，输出概率 |
| 交叉熵损失 | L = -[y·log(p) + (1-y)·log(1-p)] |
| 决策边界 | Wx + b = 0 的超平面 |
| Softmax | 多分类的推广 |

## 3. 内容（Content）

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# Sigmoid 函数 / Sigmoid function
# ============================================================
def sigmoid(z):
    """Sigmoid: 将任意实数映射到 (0, 1).
    Time: O(N)  Space: O(N)
    """
    return 1 / (1 + np.exp(-z))

# ============================================================
# Sklearn 逻辑回归 / Sklearn logistic regression
# ============================================================
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ============================================================
# 多分类（Softmax 回归）
# ============================================================
from sklearn.datasets import load_iris
iris = load_iris()
model_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
model_multi.fit(iris.data, iris.target)
print(f"Iris Accuracy: {model_multi.score(iris.data, iris.target):.4f}")
```

## 4. 详细推理（Deep Dive）

```
逻辑回归 → Softmax → 神经网络输出层:

二分类: p = sigmoid(Wx + b)  →  L = BCE(y, p)
多分类: p = softmax(Wx + b)  →  L = CE(y, p)

BERT/GPT 分类头:
  hidden_state → Linear(d, num_classes) → Softmax → CE Loss
  本质就是一个逻辑回归层！
```

## 5-6. 例题/习题

**练习 1：** 手动实现逻辑回归（sigmoid + 交叉熵 + 梯度下降）。

**练习 2：** 用逻辑回归做文本分类（TF-IDF 特征 + 情感分类）。
