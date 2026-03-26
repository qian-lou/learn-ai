# 支持向量机
# Support Vector Machine (SVM)

## 1. 背景（Background）

> **为什么要学这个？**
>
> SVM 通过最大化间隔找到最优决策边界，核技巧可以处理非线性分类。虽然深度学习时代 SVM 用得少了，但其"间隔最大化"思想影响了 contrastive learning。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| 最大间隔 | 找到离两类最远的超平面 |
| 支持向量 | 距超平面最近的样本 |
| 核技巧 | 映射到高维空间处理非线性 |
| C 参数 | 正则化（软间隔） |

## 3. 内容（Content）

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ⚠️ SVM 对特征缩放敏感！必须标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 线性 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
print(f"Linear SVM: {svm_linear.score(X_test, y_test):.4f}")

# RBF 核（非线性）
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale')
svm_rbf.fit(X_train, y_train)
print(f"RBF SVM: {svm_rbf.score(X_test, y_test):.4f}")
```

## 4. 详细推理（Deep Dive）

```
SVM 核函数:
  linear: K(x,y) = x·y               → 线性可分
  rbf:    K(x,y) = exp(-γ||x-y||²)   → 最常用
  poly:   K(x,y) = (x·y + c)^d       → 多项式

SVM vs 深度学习:
  小数据 + 高维特征: SVM 可能更好
  大数据 + 复杂模式: 深度学习完胜
```

## 5-6. 例题/习题

**练习 1：** 用 SVM 做手写数字分类（MNIST 子集）。

**练习 2：** 对比不同核函数（linear, rbf, poly）的效果。
