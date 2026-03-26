# 从零实现 Transformer / Transformer from Scratch

## 1. 背景（Background）
> 动手从零实现 Transformer 是深入理解架构的最佳方式。

## 2-3. 知识点与内容
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # 自注意力 + 残差连接 + LayerNorm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN + 残差连接 + LayerNorm
        x = self.norm2(x + self.ffn(x))
        return x
```

## 4-6. 推理/例题/习题
**练习：** 堆叠 6 个 TransformerBlock 构建完整模型，在小数据集上验证。
