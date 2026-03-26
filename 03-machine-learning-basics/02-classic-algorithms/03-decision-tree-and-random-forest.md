# 决策树与随机森林 / Decision Tree and Random Forest

## 1. 背景（Background）
> 决策树自动学习 if-else 规则，随机森林是多棵树的投票集成。表格数据领域至今依然是强力 baseline。

## 2-3. 知识点与内容
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
print(export_text(dt, feature_names=feature_names))

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
importances = rf.feature_importances_  # 特征重要性排序
```

## 4. 详细推理
- 决策树通过信息增益/基尼系数选择最优分裂特征
- 随机森林 = Bagging + 特征随机采样 → 降低方差，防止过拟合
- XGBoost/LightGBM 是梯度提升树，表格数据竞赛的常胜之选

## 5-6. 例题/习题
**练习：** 用随机森林做特征重要性分析，可视化 Top 10 重要特征。
