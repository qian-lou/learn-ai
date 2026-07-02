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

## 5. 例题（Worked Examples）

### 例题 1：绘制神经网络输入特征相关性热力图 / Visualizing Feature Correlation Heatmap

在训练多变量回归模型前，相关性分析可以过滤多重共线特征。本例使用 Seaborn 可视化这一关联矩阵。

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 构造模拟特征数据集 / Construct feature dataframe
np.random.seed(42)
# Time: O(R * C), Space: O(R * C)
data = pd.DataFrame(np.random.randn(100, 4), columns=['feat_1', 'feat_2', 'feat_3', 'target'])
data['feat_1'] = data['target'] * 0.8 + np.random.randn(100) * 0.5  # 使得 feat_1 与 target 强相关

# 计算相关系数矩阵 / Compute correlation matrix
# Time: O(C^2 * R), Space: O(C^2)
corr = data.corr()

# 绘制热力图 / Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：使用 Seaborn 绘制关于鸢尾花（Iris）数据集中两个维度特征的分类散点图。
*参考答案*：
```python
import seaborn as sns
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
# Time: O(N), Space: O(N)
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')
plt.show()
```

### 进阶题
**练习 2**：在模型泛化性对比分析中，已知多个评估指标（Accuracy, F1-Score, ROC-AUC）在不同基座模型（LLaMA, GPT-3.5, BERT）上的分布。使用 Seaborn 的 Boxplot（箱线图）在同一张图中并排展示它们的对比分布。
*参考答案*：
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 构造宽格式数据：3 个模型 × 3 个指标，各 50 次评估 / Construct wide-format data
n = 50
wide = pd.DataFrame({
    'Model': np.repeat(['BERT', 'GPT-3.5', 'LLaMA'], n),
    'Accuracy': np.concatenate([
        np.random.normal(0.80, 0.02, n),
        np.random.normal(0.88, 0.015, n),
        np.random.normal(0.92, 0.01, n)
    ]),
    'F1-Score': np.concatenate([
        np.random.normal(0.78, 0.02, n),
        np.random.normal(0.86, 0.015, n),
        np.random.normal(0.90, 0.01, n)
    ]),
    'ROC-AUC': np.concatenate([
        np.random.normal(0.85, 0.02, n),
        np.random.normal(0.91, 0.015, n),
        np.random.normal(0.95, 0.01, n)
    ])
})

# melt 成长格式（Model × Metric × Value 三列），hue='Metric' 即可在同一张图并排展示三个指标
# Time: O(N), Space: O(N)
long_df = wide.melt(id_vars='Model', var_name='Metric', value_name='Value')
sns.boxplot(data=long_df, x='Model', y='Value', hue='Metric')
plt.title('Metric Distributions across Models')
plt.show()
```