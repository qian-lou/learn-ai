# Tensor 基础与自动求导 / Tensor Basics and Autograd

## 1. 背景（Background）
> PyTorch Tensor 类似 NumPy ndarray 但支持 GPU 加速和自动求导，是深度学习计算的基本单元。

## 2-3. 知识点与内容
```python
import torch
import numpy as np

x = torch.randn(2, 3, requires_grad=True)  # Shape: [2, 3]

# NumPy 互转
np_arr = x.detach().numpy()
tensor = torch.from_numpy(np_arr)

# GPU 操作
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# 形状操作 / Shape operations
a = torch.randn(2, 3, 4)       # Shape: [2, 3, 4]
b = a.view(2, 12)               # reshape
c = a.permute(0, 2, 1)          # 转置: [2, 4, 3]
d = a.unsqueeze(0)               # 增加维度: [1, 2, 3, 4]
e = d.squeeze(0)                 # 去掉维度: [2, 3, 4]
```

## 4-6. 推理/例题/习题
**练习：** 对比 CPU 和 GPU 上矩阵乘法的性能差异。
