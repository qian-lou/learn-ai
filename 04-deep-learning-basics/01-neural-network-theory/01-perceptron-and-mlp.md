# 感知机与多层感知机 / Perceptron and MLP

## 1. 背景（Background）
> 感知机是最简单的神经网络。MLP 通过堆叠多层获得非线性拟合能力。Transformer 中的 FFN 层就是一个 MLP。

## 2-3. 知识点与内容
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # Shape: [B, input] -> [B, hidden]
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # Shape: [B, hidden] -> [B, output]
        )
    
    def forward(self, x):
        return self.layers(x)
```

## 4. 详细推理
- 单层感知机只能解决线性可分问题（XOR 不行）
- 通用近似定理：一个隐藏层的 MLP 可以近似任何连续函数
- Transformer FFN: `FFN(x) = GELU(xW1 + b1)W2 + b2`

## 5-6. 例题/习题
**练习：** 用 PyTorch 构建 MLP 解决 XOR 问题，验证通用近似定理。
