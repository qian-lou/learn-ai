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

## 5. 例题（Worked Examples）

### 例题 1：手动计算基尼系数（Gini Impurity）与节点切分 / Manual calculation of Gini Impurity

决策树在寻找分割点时使用基尼系数衡量节点纯度。本例计算样本分裂前后的基尼系数增益。

```python
import numpy as np

# 样本类别标签 / Label list
# 10 个样本，5 个正类(1)，5 个负类(0)
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

def calc_gini(y: np.ndarray) -> float:
    """计算基尼系数 / Compute Gini Impurity.
    
    Time: O(N), Space: O(U) - U 为类别个数 / U is the number of classes.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))

# 分裂前的基尼系数 / Gini before split
gini_init = calc_gini(labels)

# 假设某个特征分裂：左子树 4 个正 1 个负；右子树 1 个正 4 个负
# Split left and right nodes
left_node = np.array([1, 1, 1, 1, 0])
right_node = np.array([1, 0, 0, 0, 0])

# 加权计算分裂后的基尼系数 / Weighted Gini after split
gini_split = (len(left_node) / len(labels)) * calc_gini(left_node) + \
             (len(right_node) / len(labels)) * calc_gini(right_node)

print(f"初始基尼系数 / Initial Gini: {gini_init:.4f}")
print(f"分裂后基尼系数 / Split Gini: {gini_split:.4f}")
print(f"基尼系数增益 / Gini Gain: {gini_init - gini_split:.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释随机森林中“Bagging”与“特征随机选择”这两个核心机制对降低过拟合的作用。
*参考答案*：
- **Bagging（自助采样）**：每次随机有放回地抽取部分样本进行训练，增加了树之间的差异性，降低方差。
- **特征随机选择**：每次切分节点时，只随机选择特征子集，避免某些强特征霸占所有的分裂，使各决策树更为互补和独立。

### 进阶题
**练习 2**：使用 Sklearn 的决策树分类器对手写数字数据集进行分类，利用网格搜索（GridSearchCV）优化最大树深度 `max_depth` 与最小叶子节点样本数 `min_samples_leaf`，并输出最佳参数组合。
*参考答案*：
```python
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# Time: O(Grid_Size * cv * Train_Time), Space: O(N * D)
digits = load_digits()
clf = DecisionTreeClassifier()
param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 5]}
grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(digits.data, digits.target)
print(f"最佳超参数组合: {grid.best_params_}")
```\n