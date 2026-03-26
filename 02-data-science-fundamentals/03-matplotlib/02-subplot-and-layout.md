# 子图与布局
# Subplot and Layout

## 1. 背景（Background）

> 实际工作中常需要在一张图中展示多个子图，用于对比分析。类似 Java Swing 的 GridLayout。

## 2-3. 知识点与内容

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：折线图 / Top-left: line plot
axes[0, 0].plot(range(10), np.random.randn(10).cumsum())
axes[0, 0].set_title('Training Loss')

# 右上：柱状图 / Top-right: bar chart
axes[0, 1].bar(['A', 'B', 'C'], [85, 92, 78])
axes[0, 1].set_title('Model Accuracy')

# 左下：散点图 / Bottom-left: scatter
axes[1, 0].scatter(np.random.randn(50), np.random.randn(50))
axes[1, 0].set_title('Data Distribution')

# 右下：直方图 / Bottom-right: histogram
axes[1, 1].hist(np.random.randn(1000), bins=30, edgecolor='black')
axes[1, 1].set_title('Feature Distribution')

plt.tight_layout()
plt.savefig('subplots.png', dpi=150)
```

## 4-6. 推理/例题/习题

**练习：** 创建 3×2 的子图，展示 6 种不同激活函数的图形（Sigmoid, Tanh, ReLU, LeakyReLU, GELU, SiLU）。
