# 自注意力机制详解 / Self-Attention in Detail

## 1. 背景（Background）
> 自注意力是 Transformer 的核心——序列中每个位置都能关注所有其他位置。这就是 Transformer 能处理长距离依赖的关键。

## 2-3. 知识点与内容
```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.d_k = d_model ** 0.5
    
    def forward(self, x):
        # x Shape: [B, seq_len, d_model]
        Q = self.W_q(x)  # [B, S, D]
        K = self.W_k(x)  # [B, S, D]
        V = self.W_v(x)  # [B, S, D]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k  # [B, S, S]
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)  # [B, S, D]
```

## 4. 详细推理
- 自注意力复杂度 O(N²·D)，N=序列长度，D=维度
- 这就是为什么超长序列需要 FlashAttention 等优化技术

## 5-6. 例题/习题
**练习：** 从零实现自注意力层，验证输出形状。
