# 优化算法 / Optimization Algorithms

## 1. 背景（Background）
> 优化器决定模型"如何学习"。AdamW 是大模型训练的标准优化器。

## 2-3. 知识点与内容
```python
import torch.optim as optim

# SGD → Momentum → Adam → AdamW（演进路线）
adamw = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 学习率调度：Warmup + Cosine Decay（大模型标配）
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(adamw, T_max=100)
```

## 4. 详细推理
- SGD: θ = θ - lr × ∇L（最基础）
- Adam: 自适应学习率 = Momentum + RMSprop
- AdamW: 修正权重衰减实现，大模型训练首选

## 5-6. 例题/习题
**练习：** 用不同优化器训练同一模型，对比收敛曲线。
