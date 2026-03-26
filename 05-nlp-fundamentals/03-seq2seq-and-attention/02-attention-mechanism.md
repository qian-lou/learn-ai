# 注意力机制原理 / Attention Mechanism

## 1. 背景（Background）

> **为什么要学这个？**
>
> 注意力机制（Attention）是 **Transformer 的核心创新**，也是整个大模型时代的基石。"Attention Is All You Need"（2017）这篇论文的标题精准概括了这一点。
>
> Attention 的核心思想极为优雅：让模型在处理每个位置时，可以**动态地"关注"输入的任何部分**。对于 Java 工程师来说，Attention 就像是一个**动态查询系统**——Query 是你的问题，Key 是索引，Value 是数据。你用 Query 和所有 Key 计算相关度，然后按相关度加权获取 Value。
>
> **在整个体系中的位置：** Attention 是从 RNN 到 Transformer 的核心桥梁。理解 Attention，就理解了 Transformer 的心脏。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | 类比 |
|------|------|------|
| Query (Q) | "我想查什么" | 数据库 SELECT 查询 |
| Key (K) | "有哪些索引" | 数据库索引字段 |
| Value (V) | "实际数据" | 数据库返回的行 |
| 注意力权重 | Q 和 K 的相关度 | 查询匹配度 |
| 缩放因子 √d_k | 防止 softmax 饱和 | 归一化 |

**核心公式：**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

其中：
  Q: [B, T_q, d_k]  — 查询矩阵
  K: [B, T_k, d_k]  — 键矩阵
  V: [B, T_k, d_v]  — 值矩阵
  Output: [B, T_q, d_v]
```

## 3. 内容（Content）

### 3.1 Attention 的种类

```
1. Additive Attention（Bahdanau, 2014）
   score(q, k) = v^T · tanh(W₁q + W₂k)
   → 最早的 Attention，用于 RNN Seq2Seq

2. Dot-Product Attention（Luong, 2015）
   score(q, k) = q^T · k
   → 更简单、更快

3. Scaled Dot-Product Attention（Vaswani, 2017）
   score(q, k) = q^T · k / √d_k        ← Transformer 用这个！
   → 缩放防止 softmax 饱和

4. Multi-Head Attention（Vaswani, 2017）
   多个独立的 Attention 并行计算 → 捕获不同类型的关注模式
```

### 3.2 Scaled Dot-Product Attention 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention.
    
    Attention(Q,K,V) = softmax(QK^T / √d_k) · V
    
    Args:
        Q: 查询 / Query. Shape: [B, ..., T_q, d_k]
        K: 键 / Key. Shape: [B, ..., T_k, d_k]
        V: 值 / Value. Shape: [B, ..., T_k, d_v]
        mask: 注意力掩码 / Attention mask.
    
    Returns:
        (output, attention_weights)
    """
    d_k = Q.size(-1)
    
    # 1. 计算注意力分数 / Compute attention scores
    # QK^T: [B, ..., T_q, d_k] × [B, ..., d_k, T_k] → [B, ..., T_q, T_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（如有）/ Apply mask if any
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax 归一化 / Softmax normalization
    attn_weights = F.softmax(scores, dim=-1)  # [B, ..., T_q, T_k]
    
    # 4. 加权求和 / Weighted sum
    output = torch.matmul(attn_weights, V)  # [B, ..., T_q, d_v]
    
    return output, attn_weights
```

### 3.3 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """多头注意力 / Multi-Head Attention.
    
    多个独立的 attention head 并行计算，
    每个 head 关注不同的子空间特征。
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q, K, V 投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query/key/value: Shape [B, T, d_model]
        Returns:
            output: Shape [B, T, d_model]
        """
        B, T, _ = query.shape
        
        # 1. 线性投影 + 拆分头
        Q = self.W_q(query).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, n_heads, T, d_head]
        
        # 2. Scaled Dot-Product Attention
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # output: [B, n_heads, T, d_head]
        
        # 3. 合并头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights
```

### 3.4 三种 Attention 模式

```
1. Self-Attention（自注意力）:
   Q = K = V = 同一序列
   每个位置关注同一序列的其他位置
   用于: BERT Encoder, GPT Decoder

2. Cross-Attention（交叉注意力）:
   Q = 解码器序列, K = V = 编码器输出
   解码器关注编码器的输出
   用于: T5/BART 的翻译/摘要

3. Causal (Masked) Self-Attention:
   Q = K = V = 同一序列，但下三角掩码
   每个位置只能看到前面的位置
   用于: GPT 自回归生成

掩码示例（序列长度=4）:
  ┌───┬───┬───┬───┐       ┌───┬───┬───┬───┐
  │ 1 │ 1 │ 1 │ 1 │       │ 1 │ 0 │ 0 │ 0 │
  │ 1 │ 1 │ 1 │ 1 │       │ 1 │ 1 │ 0 │ 0 │
  │ 1 │ 1 │ 1 │ 1 │       │ 1 │ 1 │ 1 │ 0 │
  │ 1 │ 1 │ 1 │ 1 │       │ 1 │ 1 │ 1 │ 1 │
  └───┴───┴───┴───┘       └───┴───┴───┴───┘
   Bidirectional            Causal (GPT)
   (BERT)                   (只看左边)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么要除以 √d_k？

```
问题：当 d_k 很大时，Q·K^T 的值会变得很大

假设 Q 和 K 的每个元素独立服从 N(0,1)：
  q·k = Σ(qᵢ × kᵢ), i=1...d_k
  E[q·k] = 0
  Var[q·k] = d_k  ← 方差与 d_k 成正比！

当 d_k = 64 时，scores 的标准差 ≈ √64 = 8
→ softmax 输入值很大 → 梯度趋近于 0（饱和区）

除以 √d_k 后：
  Var[q·k / √d_k] = 1  ← 方差回到 1，softmax 在有效区间
```

### 4.2 多头的意义

```
为什么用多个 head 而不是一个大 head？

单头 (d_model=768):
  一组 Q, K, V → 一种关注模式

多头 (8 heads, d_head=96):
  Head 1: 关注语法关系（主语-谓语）
  Head 2: 关注指代关系（it → the cat）
  Head 3: 关注位置关系（相邻词）
  ...
  
  每个 head 学习不同的关注模式
  → 模型能同时捕获多种语言现象
```

## 5. 例题（Worked Examples）

### 例题：可视化 Attention 权重

```python
# 注意力权重可视化
import matplotlib.pyplot as plt

# 模拟 attention weights
sentence = ["I", "love", "machine", "learning"]
attn = torch.softmax(torch.randn(4, 4), dim=-1)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(attn.numpy(), cmap='Blues')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(sentence)
ax.set_yticklabels(sentence)
plt.colorbar(im)
plt.title("Attention Weights")
plt.show()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现 Scaled Dot-Product Attention 并验证输出形状。

**练习 2：** 解释 Causal Mask 如何保证 GPT 的自回归特性。

### 进阶题

**练习 3：** 实现完整的 Multi-Head Attention，并测试 n_heads=1, 4, 8 对模型效果的影响。

**练习 4：** 用 BERT 提取 Attention 权重（`output_attentions=True`），可视化模型在不同层对句子的关注模式。
