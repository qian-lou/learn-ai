# 支持向量机 / Support Vector Machine

## 1. 背景（Background）
> SVM 寻找最大间隔超平面来分类数据。核技巧允许在高维空间中实现非线性分类。

## 2-3. 知识点与内容
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
# kernel: linear(线性), rbf(高斯), poly(多项式)
# C: 正则化参数，越大越容易过拟合
```

## 4. 详细推理
- 核心思想：最大化分类间隔 margin = 2/||w||
- 核技巧：将低维不可分数据映射到高维，在高维中线性可分
- SVM 在小样本高维数据上表现优异（文本分类经典方法）

## 5-6. 例题/习题
**练习：** 对比 linear/rbf/poly 核在不同数据集上的效果，可视化决策边界。
