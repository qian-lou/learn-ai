# 多头注意力 / Multi-Head Attention

## 1. 背景（Background）
> 多头注意力让模型从多个视角关注信息。类似于用多个"眼睛"同时看一段文本。

## 2-3. 知识点与内容
```python
import torch.nn as nn

# PyTorch 内置实现
mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
# 12 个头，每个头的维度 = 768/12 = 64
# Input: [B, S, 768] -> Output: [B, S, 768]

# 多头的本质：将 d_model 分成 h 个子空间
# 每个头独立做注意力，最后拼接
# MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
```

## 4-6. 推理/例题/习题
**练习：** 从零实现多头注意力，理解头的并行性。
