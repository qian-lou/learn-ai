# Seaborn 高级可视化
# Seaborn Advanced Visualization

## 1. 背景（Background）

> Seaborn 基于 Matplotlib，提供更美观的统计图表和更简洁的 API。EDA（探索性数据分析）阶段的标配。

## 2-3. 知识点与内容

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置主题 / Set theme
sns.set_theme(style="whitegrid", palette="husl")

# 查看内置数据集
tips = sns.load_dataset("tips")

# 分布图 / Distribution plot
sns.histplot(data=tips, x="total_bill", hue="time", kde=True)

# 箱线图 / Box plot
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")

# 热力图（相关性矩阵）/ Heatmap (correlation matrix)
corr = tips.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)

# 配对图 / Pair plot
sns.pairplot(tips, hue="time", diag_kind="kde")

# 混淆矩阵可视化（分类模型评估必备）
# Confusion matrix visualization
confusion = [[85, 15], [10, 90]]
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
```

## 4-6. 推理/例题/习题

**练习：** 用 Seaborn 可视化一个分类数据集的特征分布和类别平衡情况。
