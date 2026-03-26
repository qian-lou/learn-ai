# Seaborn 高级可视化
# Seaborn Advanced Visualization

## 1. 背景（Background）

> **为什么要学这个？**
>
> Seaborn 建立在 Matplotlib 基础上，提供更美观的统计图表。混淆矩阵热力图、特征相关性矩阵、箱线图等在模型评估中非常常用。

## 2. 知识点（Key Concepts）

| 图表 | API | ML 应用 |
|------|-----|---------|
| 热力图 | `sns.heatmap()` | 混淆矩阵、相关性 |
| 箱线图 | `sns.boxplot()` | 分布对比 |
| 小提琴图 | `sns.violinplot()` | 密度分布 |
| 成对图 | `sns.pairplot()` | 特征关系 |
| 类别图 | `sns.catplot()` | 分类统计 |

## 3. 内容（Content）

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid")

# ============================================================
# 1. 混淆矩阵热力图 / Confusion matrix heatmap
# ============================================================
confusion = np.array([[85, 10, 5], [8, 82, 10], [3, 12, 85]])
labels = ["Positive", "Negative", "Neutral"]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# 2. 特征相关性热力图 / Feature correlation heatmap
# ============================================================
df = pd.DataFrame(np.random.randn(100, 5), columns=["F1", "F2", "F3", "F4", "F5"])
df["F2"] = df["F1"] * 0.8 + np.random.randn(100) * 0.2  # 人造相关

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# ============================================================
# 3. 分布对比 / Distribution comparison
# ============================================================
results = pd.DataFrame({
    "model": ["BERT"] * 50 + ["GPT"] * 50 + ["T5"] * 50,
    "accuracy": np.concatenate([
        np.random.normal(0.92, 0.02, 50),
        np.random.normal(0.89, 0.03, 50),
        np.random.normal(0.90, 0.025, 50),
    ])
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=results, x="model", y="accuracy", ax=axes[0])
axes[0].set_title("Box Plot")
sns.violinplot(data=results, x="model", y="accuracy", ax=axes[1])
axes[1].set_title("Violin Plot")
plt.tight_layout()
plt.show()

# ============================================================
# 4. 成对关系图 / Pair plot
# ============================================================
# sns.pairplot(df, diag_kind="kde", corner=True)
# plt.show()
```

## 4. 详细推理（Deep Dive）

```
Seaborn vs Matplotlib:
  Matplotlib: 底层灵活，但需要手动配置很多
  Seaborn: 统计图表开箱即用，默认就很美观

ML 常用可视化:
  训练过程: 损失/准确率曲线 (Matplotlib)
  模型评估: 混淆矩阵/ROC 曲线 (Seaborn)
  数据探索: 分布图/相关性矩阵 (Seaborn)
  注意力可视化: 热力图 (Matplotlib/Seaborn)
```

## 5-6. 例题/习题

**练习 1：** 绘制分类模型的混淆矩阵热力图。

**练习 2：** 用小提琴图对比 3 个模型在 5 个数据集上的性能分布。

**练习 3：** 绘制 Attention Weight 热力图（假设 attention 矩阵为 `[seq_len, seq_len]`）。
