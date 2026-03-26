# 决策树与随机森林
# Decision Tree and Random Forest

## 1. 背景（Background）

> **为什么要学这个？**
>
> 决策树是最直观的 ML 算法——模型可解释性强，无需特征缩放。随机森林通过集成多棵树，显著提升准确率。XGBoost/LightGBM 仍是结构化数据竞赛的王者。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| 信息增益 / Gini | 分裂标准 |
| 剪枝 | 防止过拟合 |
| Bagging | 随机森林的核心（Bootstrap + 聚合） |
| 特征重要性 | 可解释性 |

## 3. 内容（Content）

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ============================================================
# 决策树 / Decision Tree
# ============================================================
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
dt.fit(X_train, y_train)
print(f"DT Accuracy: {dt.score(X_test, y_test):.4f}")

# ============================================================
# 随机森林 / Random Forest
# ============================================================
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"RF Accuracy: {rf.score(X_test, y_test):.4f}")

# 特征重要性
import numpy as np
for name, imp in zip(load_iris().feature_names, rf.feature_importances_):
    print(f"  {name}: {imp:.4f}")

# 交叉验证
scores = cross_val_score(rf, X, y, cv=5)
print(f"CV Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 4. 详细推理（Deep Dive）

```
决策树 vs 随机森林 vs 深度学习:

结构化数据（表格）: RF / XGBoost > 深度学习
图像/文本/序列:      深度学习 >> RF

随机森林为什么好？
  - 多棵树投票 → 方差降低
  - 随机特征子集 → 树之间独立
  - 不容易过拟合（相比单棵树）
```

## 5-6. 例题/习题

**练习 1：** 对比决策树和随机森林在 Iris 数据集上的准确率。

**练习 2：** 用 XGBoost 做表格数据分类，与随机森林对比。
