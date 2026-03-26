# 位置编码 / Positional Encoding

## 1. 背景（Background）

> **为什么要学这个？**
>
> Transformer 没有 RNN 的递归结构，也没有 CNN 的滑动窗口——它对输入的顺序**完全无感知**。如果不加位置信息，"I love you" 和 "you love I" 对 Transformer 来说是一样的！位置编码就是解决这个问题。
>
> 对于 Java 工程师来说，位置编码就像数组的索引——没有索引，数组就退化为集合（Set），失去了顺序信息。
>
> **在整个体系中的位置：** 位置编码从正弦编码（原始 Transformer）→ 可学习编码（BERT/GPT）→ RoPE（LLaMA/Qwen）→ ALiBi（BLOOM），不断演进。

## 2. 知识点（Key Concepts）

| 位置编码方式 | 代表模型 | 可外推* | 类型 |
|-------------|---------|---------|------|
| 正弦位置编码 | 原始 Transformer | 有限 | 绝对 |
| 可学习位置编码 | BERT, GPT-2 | ❌ | 绝对 |
| RoPE（旋转位置编码）| LLaMA, Qwen | ✅ | 相对 |
| ALiBi | BLOOM | ✅ | 相对 |

*外推：能否处理训练时没见过的更长序列

## 3. 内容（Content）

### 3.1 正弦位置编码（Sinusoidal）

```python
import torch
import math

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """原始 Transformer 的正弦位置编码.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        seq_len: 序列长度 / Sequence length.
        d_model: 模型维度 / Model dimension.
    
    Returns:
        位置编码矩阵 / Positional encoding matrix. Shape: [seq_len, d_model]
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 计算频率分母 / Compute frequency denominator
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
    
    return pe

# 测试
pe = sinusoidal_positional_encoding(100, 512)
print(f"位置编码矩阵: {pe.shape}")  # [100, 512]
```

### 3.2 可视化位置编码

```python
import matplotlib.pyplot as plt

pe = sinusoidal_positional_encoding(50, 128)

plt.figure(figsize=(12, 4))
plt.imshow(pe.numpy().T, aspect='auto', cmap='RdBu')
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.show()

# 观察：低维度（底部）频率高，高维度（顶部）频率低
# → 低维编码位置的精细变化，高维编码位置的宏观模式
```

### 3.3 可学习位置编码

```python
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码（BERT/GPT-2 使用）.
    
    每个位置学习一个独立的向量，通过训练优化。
    缺点：最大长度固定，无法外推到更长序列。
    """
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        Args:
            x: 输入 / Input. Shape: [B, N, D]
        """
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)  # 加法而非拼接
```

### 3.4 RoPE（旋转位置编码）

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """RoPE 旋转位置编码（LLaMA/Qwen/ChatGLM 使用）.
    
    核心思想：将位置信息编码为向量的旋转角度
    相对位置 = 两个向量旋转角度之差
    
    Args:
        q, k: Shape [B, H, N, d_head]
        cos, sin: Shape [1, 1, N, d_head]
    """
    # 将向量拆成两半，分别旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """将向量后半部分取负后与前半部分交换."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

```
RoPE 的优势：

1. 相对位置编码
   注意力分数只取决于 token 之间的相对距离
   而非绝对位置 → 更好的泛化

2. 远程衰减
   距离越远的 token 对，注意力自然衰减
   → 编码了 "近处更重要" 的归纳偏置

3. 长度外推
   配合 NTK-Aware Scaling / YaRN 等扩展
   LLaMA-2 从 4K → 32K+ 上下文
   LLaMA-3 支持 128K 上下文
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么正弦编码能工作？

```
正弦编码的关键性质：

1. 唯一性：不同位置的编码不同
2. 有界性：值在 [-1, 1] 范围内
3. 相对位置可线性变换：
   PE(pos+k) 可以表示为 PE(pos) 的线性函数
   → 模型可以学习到相对位置关系

类比：二进制计数
  pos=0: 000  →  sin 和 cos 在不同频率上的组合
  pos=1: 001      类似于不同位频翻转速度不同
  pos=2: 010
  pos=3: 011
  低位变化快，高位变化慢 ≈ 低维频率高，高维频率低
```

### 4.2 绝对 vs 相对位置编码

```
绝对位置编码（BERT/GPT-2）：
  每个位置有固定编码：PE(0), PE(1), PE(2), ...
  问题：训练时 max_len=512 → 推理时无法处理 513+ 的位置

相对位置编码（RoPE/ALiBi）：
  编码的是 token 之间的距离
  "A 和 B 距离 3" 比 "A 在位置 5" 更有泛化性
  → 可以外推到训练时没见过的更长序列

现代大模型全部使用相对位置编码：
  LLaMA: RoPE (训练 4K, 外推到 128K)
  BLOOM: ALiBi
  Gemini: RoPE
```

## 5. 例题（Worked Examples）

### 例题：验证位置编码的距离特性

```python
pe = sinusoidal_positional_encoding(100, 64)

# 计算不同位置之间的余弦相似度
from torch.nn.functional import cosine_similarity

# 相邻位置相似度高
sim_1 = cosine_similarity(pe[10:11], pe[11:12]).item()
# 远距离位置相似度低
sim_50 = cosine_similarity(pe[10:11], pe[60:61]).item()
print(f"距离 1 的相似度: {sim_1:.4f}")
print(f"距离 50 的相似度: {sim_50:.4f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现正弦位置编码并可视化热力图。

**练习 2：** 解释如果不加位置编码，Transformer 会出什么问题。

### 进阶题

**练习 3：** 实现 RoPE，验证注意力分数只依赖相对位置（改变绝对位置但保持相对位置不变，注意力分数不变）。

**练习 4：** 用可学习位置编码训练一个小 Transformer，然后尝试输入超过 `max_len` 的序列，观察效果退化。
