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

## 4. 详细推理（Deep Dive）

- `plt.subplots` 产生的 `axes` 变量类型在不同网格数下有别：
  - 1x1：返回单一 `AxesSubplot` 对象。
  - 1xN 或 Nx1：返回一维 `ndarray` 包含各子图。
  - MxN：返回二维 `ndarray`（形状为 `[M, N]`）。
- `GridSpec` 提供了更加精细和灵活的跨行、跨列不规则布局，类似 HTML 中的 colspan 和 rowspan。

## 5. 例题（Worked Examples）

### 例题 1：使用 GridSpec 构建不规则多图的神经网络诊断面板 / GridSpec for Irregular Dashboard Layout

本例展示如何绘制跨列的大图（顶部显示训练损失演进）与底部并排的小图（展示模型参数直方图与学习率变化），构建美观的分析看板。

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 1. 模拟数据 / Mock Data
epochs = np.arange(1, 11)
loss = 1.0 / (epochs ** 0.5)
weights_dist = np.random.randn(1000)

# 2. 定义不规则的 2x2 网格 / Setup gridspec layout
# Time: O(Data_Len), Space: O(Data_Len)
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])

# 子图 1: 跨列大图（占据第 0 行的所有列）
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(epochs, loss, 'b-o', label='Training Loss')
ax0.set_title('Global Loss Progression')
ax0.set_xlabel('Epoch')
ax0.set_ylabel('Loss')
ax0.legend()

# 子图 2: 左下方小图
ax1 = fig.add_subplot(gs[1, 0])
ax1.hist(weights_dist, bins=30, color='g', alpha=0.6)
ax1.set_title('Weight Distribution')

# 子图 3: 右下方小图
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(epochs, 0.001 * (0.9 ** epochs), 'r-s')
ax2.set_title('Learning Rate Decay')

plt.tight_layout()
plt.show()
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：使用 `plt.subplots` 创建一个 `1x3` 的子图网络，将当前样本在三个通道上的色彩直方图并排绘制出来。
*参考答案*：
```python
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes 是一维数组
for i in range(3):
    data = np.random.randn(100)
    axes[i].hist(data, bins=10, color='c')
    axes[i].set_title(f"Channel {i+1}")
plt.tight_layout()
plt.show()
```

### 进阶题
**练习 2**：在进行分类模型诊断时，我们常常需要一边显示混淆矩阵热力图（使用矩阵坐标），一边在侧边显示 ROC 曲线（折线图）。编写代码绘制这一诊断布局，并要求主图与侧图的大小宽高比例为 `2:1`。
*参考答案*：
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

ax_cm = fig.add_subplot(gs[0])
ax_cm.imshow(np.array([[90, 10], [5, 95]]), cmap='Blues')
ax_cm.set_title("Confusion Matrix")

ax_roc = fig.add_subplot(gs[1])
ax_roc.plot([0, 0.2, 1], [0, 0.8, 1], 'r-o')
ax_roc.set_title("ROC Curve")

plt.tight_layout()
plt.show()
```\n