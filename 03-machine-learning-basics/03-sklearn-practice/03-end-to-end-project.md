# 端到端项目实战
# End-to-End ML Project

## 1. 背景（Background）

> **为什么要学这个？**
>
> 本节将一个 ML 项目从头到尾走一遍：问题定义 → 数据探索 → 特征工程 → 模型训练 → 评估 → 部署。这个流程与大模型微调项目的结构一致。

## 2. 知识点（Key Concepts）

```
ML 项目标准流程:
  1. 问题定义 (分类/回归/聚类?)
  2. 数据收集与探索 (EDA)
  3. 数据预处理 (清洗/编码/缩放)
  4. 特征工程
  5. 模型选择与训练
  6. 评估与调优
  7. 部署与监控
```

## 3. 内容（Content）

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ============================================================
# Step 1: 数据加载与探索 (EDA)
# ============================================================
# df = pd.read_csv("data.csv")
# print(df.shape)
# print(df.describe())
# print(df.isnull().sum())
# print(df['target'].value_counts())

# ============================================================
# Step 2: 数据预处理
# ============================================================
# X = df.drop('target', axis=1)
# y = df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# ============================================================
# Step 3: 模型对比
# ============================================================
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    "Random Forest": Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100)),
    ]),
    "Gradient Boosting": Pipeline([
        ('clf', GradientBoostingClassifier(n_estimators=100)),
    ]),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"{name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# Step 4: 最佳模型训练与评估
# ============================================================
best_model = models["Gradient Boosting"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# ============================================================
# Step 5: 模型保存
# ============================================================
joblib.dump(best_model, 'best_model.pkl')
# loaded_model = joblib.load('best_model.pkl')
```

## 4. 详细推理（Deep Dive）

```
ML 项目 vs LLM 项目:

传统 ML:                    LLM:
  数据收集 → EDA              数据收集 → 质量筛选
  特征工程                    Tokenization
  模型选择(RF/XGB)            模型选择(LLaMA/Qwen)
  超参数调优(GridSearch)       超参数调优(lr/rank/epochs)
  交叉验证                    验证集评估
  joblib 保存                 safetensors 保存
  Flask/FastAPI 部署          vLLM/TGI 部署

流程一致，工具不同
```

## 5-6. 例题/习题

**练习 1：** 完成一个完整的分类项目：数据探索 → 预处理 → 3 个模型对比 → 超参数调优 → 评估报告。

**练习 2：** 用 joblib 保存模型，用 FastAPI 构建预测 API。
