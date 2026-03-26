# 基础绑图（折线/柱状/散点）
# Basic Plotting (Line/Bar/Scatter)

## 1. 背景（Background）

> Matplotlib 是 Python 最基础的可视化库。训练过程可视化（loss 曲线、accuracy 曲线）、数据分布分析都需要它。Java 中需要 JFreeChart 等第三方库，Python 内置了强大的可视化工具链。

## 2-3. 知识点与内容

```python
import matplotlib.pyplot as plt
import numpy as np

# 折线图（训练 loss 曲线）/ Line plot (training loss curve)
epochs = range(1, 21)
loss = [1/(x**0.5) + np.random.random()*0.1 for x in epochs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b-o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# 柱状图 / Bar chart
models = ['BERT', 'GPT-2', 'T5', 'LLaMA']
accuracies = [88.5, 91.2, 89.8, 93.1]
plt.bar(models, accuracies, color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'])
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')

# 散点图 / Scatter plot
x = np.random.randn(200)
y = 2 * x + np.random.randn(200) * 0.5
plt.scatter(x, y, alpha=0.6, c=y, cmap='viridis')
plt.colorbar(label='y value')
plt.xlabel('x')
plt.ylabel('y')
```

## 4-6. 推理/例题/习题

**练习：** 可视化一个模型在训练集和验证集上的 loss 和 accuracy 变化（双 Y 轴图）。
