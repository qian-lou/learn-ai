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

## 5-6. 例题/习题

**练习 1：** 构建 Pipeline: 缺失值填充 → 标准化 → PCA → 分类。

**练习 2：** 用 GridSearchCV 搜索最佳超参数组合。
