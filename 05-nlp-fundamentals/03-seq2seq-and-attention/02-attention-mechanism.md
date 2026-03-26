# 注意力机制原理 / Attention Mechanism

## 1. 背景（Background）
> 注意力机制是 Transformer 核心创新，让模型动态关注输入的不同部分。"Attention Is All You Need" 改变了整个 AI 领域。

## 2-3. 知识点与内容
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    
    Args:
        Q: Shape [B, seq_len, d_k]  — 我想找什么
        K: Shape [B, seq_len, d_k]  — 有什么特征
        V: Shape [B, seq_len, d_v]  — 实际内容
    
    Returns:
        output: Shape [B, seq_len, d_v]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, s, s]
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
```

## 4. 详细推理
- Q·K^T 计算每对 token 的相关性分数
- 除以 √d_k 防止 softmax 进入饱和区
- softmax 归一化为概率分布（注意力权重）
- 乘以 V 得到加权输出

## 5-6. 例题/习题
**练习：** 实现注意力机制并可视化注意力权重热力图。
