# 基础绘图
# Basic Plotting

## 1. 背景（Background）

> **为什么要学这个？**
>
> 数据可视化是理解数据和展示结果的核心技能。Matplotlib 是 Python 绑图的基石，Seaborn 在其基础上提供更美观的统计图表。在模型开发中，损失曲线、混淆矩阵、特征分布图都是必备技能。

## 2. 知识点（Key Concepts）

| 图表类型 | API | 用途 |
|----------|-----|------|
| 折线图 | `plt.plot()` | 训练曲线 |
| 散点图 | `plt.scatter()` | 数据分布 |
| 柱状图 | `plt.bar()` | 对比 |
| 直方图 | `plt.hist()` | 分布 |
| 热力图 | `sns.heatmap()` | 混淆矩阵 |

## 3. 内容（Content）

```python
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. 折线图（训练损失曲线）
# ============================================================
epochs = range(1, 21)
train_loss = [2.5 * np.exp(-0.2 * x) + 0.3 + np.random.randn() * 0.05 for x in epochs]
val_loss = [2.5 * np.exp(-0.15 * x) + 0.5 + np.random.randn() * 0.08 for x in epochs]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
plt.plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()

# ============================================================
# 2. 散点图 / Scatter plot
# ============================================================
x = np.random.randn(200)
y = 2 * x + 1 + np.random.randn(200) * 0.5
plt.scatter(x, y, alpha=0.6, c=y, cmap='viridis')
plt.colorbar(label='y value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.show()

# ============================================================
# 3. 直方图（权重分布）
# ============================================================
weights = np.random.randn(10000) * 0.02  # 模拟模型权重
plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.title('Model Weight Distribution')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()

# ============================================================
# 4. 柱状图（模型对比）
# ============================================================
models = ['BERT', 'GPT-2', 'T5', 'LLaMA']
accuracy = [92.1, 89.5, 90.8, 93.2]
plt.bar(models, accuracy, color=['#2196F3', '#FF9800', '#4CAF50', '#F44336'])
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')
plt.ylim(85, 95)
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.3, f'{v}%', ha='center')
plt.show()
```

## 4. 详细推理（Deep Dive）

```
Matplotlib 两种 API 风格:
  1. pyplot (plt.*): 快速绘图，类似 MATLAB
  2. OOP (fig, ax): 更灵活，适合复杂布局

推荐: 简单图用 pyplot，多子图用 OOP
```

## 5-6. 例题/习题

**练习 1：** 绘制模型训练的损失和准确率双 Y 轴曲线。

**练习 2：** 绘制模型权重分布的直方图，标注均值和标准差。
