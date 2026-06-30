# 模型评估
# Model Evaluation

## 1. 背景（Background）

> **为什么要学这个？**
>
> 准确率不是万能指标。在不平衡数据集上，99% 准确率可能毫无意义（全预测多数类）。F1-Score、AUC-ROC、混淆矩阵——选择正确的评估指标是模型开发的关键。

## 2. 知识点（Key Concepts）

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| Accuracy | 正确数/总数 | 平衡数据 |
| Precision | TP/(TP+FP) | 关注假阳性 |
| Recall | TP/(TP+FN) | 关注假阴性 |
| F1 | 2PR/(P+R) | 不平衡数据 |
| AUC-ROC | ROC 曲线下面积 | 排序能力 |

## 3. 内容（Content）

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# 模拟预测
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

# ============================================================
# 基础指标 / Basic metrics
# ============================================================
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")

# 完整报告
print(classification_report(y_true, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print(f"混淆矩阵:\n{cm}")

# ============================================================
# 交叉验证 / Cross-validation
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)

# 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 4. 详细推理（Deep Dive）

```
指标选择指南:
  平衡数据:    Accuracy + F1
  不平衡数据:  F1 + AUC-ROC（不要用 Accuracy！）
  医疗诊断:    Recall 优先（不能漏诊）
  垃圾邮件:    Precision 优先（不能误判）
  排序任务:    AUC-ROC / NDCG
  生成任务:    BLEU / ROUGE / BERTScore

过拟合检测:
  训练 ACC 高 + 验证 ACC 低 → 过拟合
  → 增加数据 / 正则化 / 减小模型
```

## 5. 例题（Worked Examples）

### 例题 1：计算不平衡分类下的查准率 (Precision)、查全率 (Recall) 与 F1 值 / Precision-Recall calculation

在信贷风控等样本极不均衡的场景，高准确率（Accuracy）极具欺骗性。本例计算真正能表征大模型性能的关键指标。

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 真实标签与预测标签 / True labels and predictions
# 1 为有风险（正类，极少数），0 为无风险（负类）
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1])  # 漏判了 1 个，误判了 1 个

# 1. 计算混淆矩阵 / Compute confusion matrix
# Time: O(N), Space: O(1)
cm = confusion_matrix(y_true, y_pred)

# 2. 计算各项核心评估指标 / Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"混淆矩阵 / Confusion Matrix:\n{cm}")
print(f"查准率 / Precision: {precision:.4f}")
print(f"查全率 / Recall: {recall:.4f}")
print(f"F1 调和均值 / F1-Score: {f1:.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释为什么在极度倾斜的数据集上，F1-Score 比 Accuracy 更适合作为模型评估的终极指标？
*参考答案*：
如果数据集中 99% 的样本为负类，若模型全部盲猜负类，Accuracy 依然能达到 99%（看似完美，实则对正类毫无识别能力）。F1-Score 同时考虑了精确率与召回率的调和均值，若对少数类的召回率很低或精确率很低，F1-Score 都会迅速降低，能够更客观地反映模型在关键类别上的性能。

### 进阶题
**练习 2**：编写代码，使用 `cross_val_score` 对线性模型在波士顿房价数据集（或生成数据集）上进行 5 折交叉验证，输出每一折的 RMSE 并计算平均均方根误差。
*参考答案*：
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
# Time: O(5 * Train_Time), Space: O(N * D)
X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
reg = LinearRegression()
scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"5折 RMSE 分数: {rmse_scores}")
print(f"平均 RMSE: {np.mean(rmse_scores):.4f}")
```\n