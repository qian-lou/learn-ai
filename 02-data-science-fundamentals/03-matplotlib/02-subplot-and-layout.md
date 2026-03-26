# 子图与布局
# Subplot and Layout

## 1. 背景（Background）

> **为什么要学这个？**
>
> 实际做模型分析时，常需要在一张图中展示多个子图——训练曲线、混淆矩阵、特征分布并排展示。`subplot` 和 `subplots` 是创建复杂布局的核心。

## 2. 知识点（Key Concepts）

| API | 说明 |
|-----|------|
| `plt.subplot(r, c, idx)` | 创建单个子图 |
| `fig, axes = plt.subplots(r, c)` | 创建子图网格（推荐） |
| `plt.tight_layout()` | 自动调整间距 |
| `GridSpec` | 复杂不规则布局 |

## 3. 内容（Content）

```python
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. 基础子图 / Basic subplots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 子图 1: 折线图
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin(x)')

# 子图 2: 散点图
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100), alpha=0.5)
axes[0, 1].set_title('Scatter')

# 子图 3: 直方图
axes[1, 0].hist(np.random.randn(1000), bins=30, alpha=0.7)
axes[1, 0].set_title('Histogram')

# 子图 4: 柱状图
axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 5])
axes[1, 1].set_title('Bar Chart')

fig.suptitle('Model Analysis Dashboard', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 2. 共享坐标轴 / Shared axes
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
ax1.plot(x, np.sin(x), label="sin")
ax2.plot(x, np.cos(x), label="cos", color='r')
ax1.legend()
ax2.legend()
plt.show()

# ============================================================
# 3. 双 Y 轴 / Dual Y-axis
# ============================================================
fig, ax1 = plt.subplots(figsize=(8, 5))
epochs = range(1, 11)
loss = [2.0, 1.5, 1.2, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38]
acc = [60, 70, 75, 82, 85, 87, 89, 90, 91, 92]

ax1.plot(epochs, loss, 'b-o', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')

ax2 = ax1.twinx()
ax2.plot(epochs, acc, 'r-s', label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color='r')

fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
plt.title('Training Progress')
plt.show()
```

## 4-6. 推理/例题/习题

**练习 1：** 创建一个 2×3 的子图网格，展示 6 种不同的数据分布。

**练习 2：** 用双 Y 轴图展示训练过程中 loss 和 learning rate 的变化。
