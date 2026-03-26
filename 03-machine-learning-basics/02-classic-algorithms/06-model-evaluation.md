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

## 5-6. 例题/习题

**练习 1：** 在不平衡数据集上，对比 Accuracy 和 F1 的差异。

**练习 2：** 画 ROC 曲线和 Precision-Recall 曲线。
