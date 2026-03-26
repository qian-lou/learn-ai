# 位置编码 / Positional Encoding

## 1. 背景（Background）
> Transformer 没有循环结构，需要位置编码告诉模型 token 的顺序信息。

## 2-3. 知识点与内容
```python
import torch
import math

def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # Shape: [seq_len, d_model]

# 现代大模型多用 RoPE（旋转位置编码）替代正弦位置编码
# RoPE 支持外推到更长序列（LLaMA/Qwen/ChatGLM 都使用）
```

## 4-6. 推理/例题/习题
**练习：** 可视化位置编码矩阵，理解不同位置的编码模式。
