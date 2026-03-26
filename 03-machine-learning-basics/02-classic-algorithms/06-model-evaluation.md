# 模型评估与调优 / Model Evaluation and Tuning

## 1. 背景（Background）
> Accuracy/Precision/Recall/F1 在大模型评测中依然广泛使用。理解过拟合与欠拟合是调优基础。

## 2-3. 知识点与内容
```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
# Precision（精确率）：预测为正中实际为正的比例
# Recall（召回率）：实际为正中被正确预测的比例
# F1 = 2 * P * R / (P + R)

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.4f} +/- {scores.std():.4f}")
```

## 4. 详细推理
- 过拟合：训练集好，测试集差 → 增加数据/正则化/dropout
- 欠拟合：两个都差 → 增大模型/减少正则化
- 偏差-方差权衡（Bias-Variance Tradeoff）

## 5-6. 例题/习题
**练习：** 训练多个模型，用交叉验证比较，绘制 ROC 曲线和 PR 曲线。
