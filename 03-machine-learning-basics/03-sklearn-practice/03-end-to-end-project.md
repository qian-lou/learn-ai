# 端到端 ML 项目实战 / End-to-End ML Project

## 1. 背景（Background）
> 完整的 ML 项目流程：数据加载 → EDA → 清洗 → 特征工程 → 训练 → 调优 → 评估 → 保存部署。

## 2-3. 知识点与内容
```python
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import joblib

# 1. 加载数据 / Load data
df = pd.read_csv("data.csv")

# 2. EDA 探索分析 / Exploratory data analysis
df.describe()
df.info()

# 3. 数据预处理 / Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 超参数搜索 / Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")

# 5. 模型保存 / Save model
joblib.dump(grid.best_estimator_, 'model.pkl')
loaded_model = joblib.load('model.pkl')
```

## 4-6. 推理/例题/习题
**练习：** 从 Kaggle 选一个数据集，完成从 EDA 到模型部署的全流程。
