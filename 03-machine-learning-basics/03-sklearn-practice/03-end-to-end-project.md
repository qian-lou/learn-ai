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

## 5. 例题（Worked Examples）

### 例题 1：端到端波士顿房价（模拟数据集）预测回归工程项目 / End-to-End Regression Project

本例演示从获取数据、训练集测试集划分、特征插补预处理、交叉验证模型拟合、到计算泛化指标的完整全套流程。

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 生成并划分数据集 / Generate and split data
X, y = make_regression(n_samples=500, n_features=8, noise=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特征标准化 (注意：先 fit_transform 训练集，再 transform 测试集) / Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 使用交叉验证选择模型超参数 alpha / Cross-validation for hyperparameter tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_alpha = 1.0
best_r2 = -float('inf')

for alpha in [0.01, 0.1, 1.0, 10.0]:
    r2_scores = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        # 拆分 K 折数据
        k_X_tr, k_X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        k_y_tr, k_y_val = y_train[train_idx], y_train[val_idx]
        
        # 训练 Ridge 模型
        model = Ridge(alpha=alpha)
        model.fit(k_X_tr, k_y_tr)
        r2_scores.append(r2_score(k_y_val, model.predict(k_X_val)))
    
    mean_r2 = np.mean(r2_scores)
    if mean_r2 > best_r2:
        best_r2 = mean_r2
        best_alpha = alpha

# 4. 最终评估 / Final Evaluation
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)
preds = final_model.predict(X_test_scaled)

print(f"最佳超参数 alpha / Best alpha: {best_alpha}")
print(f"测试集 MAE 误差 / Test MAE: {mean_absolute_error(y_test, preds):.4f}")
print(f"测试集 R^2 决定系数 / Test R^2: {r2_score(y_test, preds):.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在模型评估前，为什么必须严格执行“训练集拟合 StandardScaler，测试集仅进行 transform 运算”的规则？
*参考答案*：
如果使用测试集去拟合 `scaler.fit`，就等于泄露了测试集的整体数据分布（如均值、标准差）到了训练过程中，这种现象称为数据泄露（Data Leakage），会导致评估出来的模型泛化指标过度虚高。

### 进阶题
**练习 2**：在模型性能调优中，我们常常面临“高方差（过拟合）”和“高偏差（欠拟合）”这两种极端情况。请为这两种情况分别列举 3 种有效的工程治理优化手段。
*参考答案*：
- **治理高偏差（欠拟合）**：
  1. 增加特征量或创建组合交叉特征；
  2. 换用更复杂的模型（例如从线性模型换到集成树模型或神经网络）；
  3. 减少模型的正则化惩罚（例如调小正则项权重）。
- **治理高方差（过拟合）**：
  1. 收集更多的数据样本；
  2. 限制模型复杂度（如限制决策树的深度，对神经网络添加 Dropout）；
  3. 增加 L1/L2 正则项惩罚，或者利用特征选择减少冗余特征。