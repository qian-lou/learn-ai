# Sklearn Pipeline
# Sklearn Pipeline

## 1. 背景（Background）

> **为什么要学这个？**
>
> Pipeline 将预处理和模型训练封装为一个统一的工作流，防止数据泄露（data leakage），简化代码。对于 Java 工程师来说，Pipeline 就像 **责任链模式（Chain of Responsibility）**。

## 2. 知识点（Key Concepts）

| 组件 | 功能 |
|------|------|
| `Pipeline` | 串联多个步骤 |
| `ColumnTransformer` | 对不同列做不同预处理 |
| `GridSearchCV` | 超参数搜索 |

## 3. 内容（Content）

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# ============================================================
# 基础 Pipeline
# ============================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),     # 标准化
    ('pca', PCA(n_components=2)),      # 降维
    ('clf', RandomForestClassifier()), # 分类
])

pipe.fit(X, y)
print(f"Pipeline Accuracy: {pipe.score(X, y):.4f}")

# ============================================================
# 超参数搜索（GridSearchCV）
# ============================================================
param_grid = {
    'pca__n_components': [2, 3, 4],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 5, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print(f"Best params: {grid.best_params_}")
print(f"Best score:  {grid.best_score_:.4f}")
```

## 4. 详细推理（Deep Dive）

```
为什么用 Pipeline？
  1. 防止数据泄露: scaler.fit 只在训练集上
  2. 代码简洁: 一行 pipe.fit(X, y) 搞定一切
  3. 可序列化: joblib.dump(pipe, 'model.pkl')
  4. 与 GridSearchCV 无缝集成
```

## 5. 例题（Worked Examples）

### 例题 1：构建包含数据插补、编码、降维和回归的完整 Pipeline / End-to-End Pipeline

为了防止机器学习中的数据泄露（Data Leakage），预处理步骤必须与模型估计器绑定。以下例题演示使用 Sklearn Pipeline 执行一连串任务。

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# 1. 制造带缺失值的数据集 / Generate data with missing values
X, y = make_regression(n_samples=200, n_features=10, random_state=42)
X[np.random.rand(*X.shape) < 0.1] = np.nan  # 10% 缺失率

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 组装 Pipeline / Define Pipeline pipeline
# Time: O(Train_Time), Space: O(N * D)
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),      # 缺失值填充
    ('scaler', StandardScaler()),                    # 数据标准化
    ('pca', PCA(n_components=5)),                    # PCA 降维到 5 维
    ('regressor', Ridge(alpha=1.0))                  # 岭回归
])

# 3. 拟合与评估 / Fit and evaluate
pipe.fit(X_train, y_train)
test_score = pipe.score(X_test, y_test)

print(f"训练集拟合正常。测试集评估决定系数 R^2 Score: {test_score:.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在 Pipeline 中，如果我们想对数值特征和类别特征应用不同的预处理策略，应该使用 Sklearn 中的哪个类？
*参考答案*：
应该使用 `ColumnTransformer`（列转换器），它允许对不同的特征列分别指定各自对应的 Pipeline 步骤，然后纵向合并。

### 进阶题
**练习 2**：构建一个 Pipeline 包含缺失值填充和决策树分类器。利用网格搜索（GridSearchCV）优化 Pipeline 内部决策树的 `max_depth` 属性。
*参考答案*：
```python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()
# 使用带 __ 的命名约定引用步骤内部超参数 / Reference internal nested params
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', DecisionTreeClassifier())
])
param_grid = {'clf__max_depth': [3, 5, None]}
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(iris.data, iris.target)
print(f"最佳 Pipeline 超参数: {grid.best_params_}")
```\n