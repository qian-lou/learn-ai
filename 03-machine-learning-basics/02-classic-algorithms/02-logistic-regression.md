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
model_multi = LogisticRegression(max_iter=1000)  # sklearn 1.5+ 默认即 multinomial，multi_class 参数已废弃
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

## 5. 例题（Worked Examples）

### 例题 1：利用逻辑回归实现二分类任务并输出性能指标 / Logistic Regression Binary Classification

逻辑回归广泛用于点击率预估。本例训练模型并计算混淆矩阵、AUC 曲线下面积。

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. 制造分类数据集 / Generate dataset
# Time: O(N * D), Space: O(N * D)
X, y = make_classification(n_samples=500, n_features=10, n_informative=8, random_state=42)

# 2. 训练逻辑回归模型 / Train Logistic Regression
# Time: O(Iterations * N * D), Space: O(D)
clf = LogisticRegression()
clf.fit(X, y)

# 3. 概率预测与指标计算 / Prediction and validation
preds = clf.predict(X)
probs = clf.predict_proba(X)[:, 1]  # 获取正例概率 / Probability of positive class.

print("分类指标分析报告 / Classification Report:")
print(classification_report(y, preds))
print(f"ROC-AUC 指标 / ROC-AUC Score: {roc_auc_score(y, probs):.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：写出逻辑回归模型使用的 Sigmoid 函数数学公式，并说明为什么它能将数值映射到 `(0, 1)` 区间。
*参考答案*：
$\sigma(z) = \frac{1}{1 + e^{-z}}$
当 $z \to \infty$ 时，$e^{-z} \to 0$，$\sigma(z) \to 1$；当 $z \to -\infty$ 时，$e^{-z} \to \infty$，$\sigma(z) \to 0$。因此输出天然限制在 `(0, 1)` 区间，适合表示概率。

### 进阶题
**练习 2**：从零编写 Python 函数，实现逻辑回归中的交叉熵损失函数（Binary Cross Entropy Loss）计算，并且要考虑边界情况（概率接近 0 或 1 时 `log` 的数值溢出问题）。
*参考答案*：
```python
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Time: O(N), Space: O(1)"""
    # 使用 np.clip 截断，防止 log(0) 产生 NaN / Avoid overflow
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
    return float(loss)

y_true = np.array([1, 0, 1, 0])
y_prob = np.array([0.95, 0.05, 0.99, 0.01])
print(f"交叉熵损失: {binary_cross_entropy(y_true, y_prob):.6f}")
```