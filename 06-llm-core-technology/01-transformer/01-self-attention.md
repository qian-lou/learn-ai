# 自注意力机制详解 / Self-Attention in Detail

## 1. 背景（Background）

> **为什么要学这个？**
>
> 自注意力（Self-Attention）是 Transformer 的**核心引擎**。它让序列中的每个位置都能直接"看到"所有其他位置，彻底解决了 RNN 的长距离依赖和顺序计算问题。可以说，**理解自注意力就是理解大模型的一半**。
>
> 对于 Java 工程师来说，自注意力就像是一个**全连接的消息传递系统**——每个节点（token）可以向所有其他节点发送查询，接收相关信息，然后根据相关性做加权聚合。
>
> **在整个体系中的位置：** 自注意力是 Multi-Head Attention 的基础，而 Multi-Head Attention 是每个 Transformer Block 的核心组件。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | 计算复杂度 |
|------|------|-----------|
| Self-Attention | Q=K=V 来自同一序列 | O(N²·D) |
| Cross-Attention | Q 来自解码器，K=V 来自编码器 | O(N·M·D) |
| Causal Attention | 带掩码的自注意力（GPT） | O(N²·D) |
| 缩放点积 | QK^T / √d_k | 防止梯度消失 |

**核心公式：**
```
Self-Attention(X) = softmax(XWq · (XWk)^T / √d_k) · XWv

输入: X ∈ ℝ^{N×D}（N 个 token，D 维表示）
输出: Z ∈ ℝ^{N×D}（每个 token 融合了全局信息）
```

## 3. 内容（Content）

### 3.1 自注意力的直觉理解

```
句子: "The cat sat on the mat because it was tired"

对于 "it"，自注意力回答：
  "it" 指代谁？→ 关注 "cat"（注意力权重最高）
  在干什么？  → 关注 "tired"
  在哪里？    → 关注 "mat"

每个 token 都能"看到"所有其他 token
→ 通过学习的 Q/K/V 权重，自动学会关注重要的位置

对比 RNN：
  RNN 到 "it" 时，"cat" 的信息已经被压缩了多层
  Self-Attention 直接连接 "it" 和 "cat"，无信息损失
```

### 3.2 从零实现自注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """从零实现自注意力 / Self-Attention from scratch.
    
    Args:
        d_model: 模型维度 / Model dimension.
        dropout: Dropout 概率 / Dropout probability.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # 查询投影
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # 键投影
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # 值投影
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列 / Input sequence. Shape: [B, N, D]
            mask: 注意力掩码 / Attention mask. Shape: [B, 1, N, N] or [1, 1, N, N]
        
        Returns:
            输出 / Output. Shape: [B, N, D]
        """
        # 1. 线性投影 / Linear projection
        Q = self.W_q(x)  # Shape: [B, N, D]
        K = self.W_k(x)  # Shape: [B, N, D]
        V = self.W_v(x)  # Shape: [B, N, D]
        
        # 2. 计算注意力分数 / Compute attention scores
        # QK^T / √d_k
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores Shape: [B, N, N] — N×N 的注意力矩阵
        
        # 3. 应用掩码 / Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax 归一化 / Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加权求和 / Weighted sum
        output = torch.matmul(attn_weights, V)  # [B, N, D]
        
        return output

# 测试 / Test
attn = SelfAttention(d_model=64)
x = torch.randn(2, 10, 64)  # 2 个样本，10 个 token，64 维
out = attn(x)
print(f"输入: {x.shape} → 输出: {out.shape}")  # [2, 10, 64]
```

### 3.3 Causal Mask（因果掩码）

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """创建因果掩码（GPT 使用）/ Create causal mask for GPT.
    
    每个位置只能看到它自己和前面的位置，看不到后面。
    
    Returns:
        mask: Shape [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

# 示例：序列长度=4
mask = create_causal_mask(4)
print(mask.squeeze())
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
# 位置 i 只能看到 ≤ i 的位置
```

### 3.4 注意力矩阵的可视化理解

```
注意力矩阵 A[i][j] 的含义：
  位置 i 对位置 j 的关注程度

示例: "I love NLP"
       I    love   NLP
I    [0.7   0.2   0.1]   ← I 主要关注自己
love [0.3   0.4   0.3]   ← love 均匀关注
NLP  [0.1   0.3   0.6]   ← NLP 主要关注自己

每行之和 = 1（softmax 归一化）
纵向对比反映了每个位置被关注的程度
```

## 4. 详细推理（Deep Dive）

### 4.1 自注意力的计算复杂度

```
Self-Attention 的复杂度分析：

QK^T 矩阵乘法: [B, N, D] × [B, D, N] = [B, N, N]
  时间复杂度: O(N²·D)
  空间复杂度: O(N²)（存储 N×N 注意力矩阵）

当 N（序列长度）很大时：
  N = 512:   注意力矩阵 = 262K 元素 ✅
  N = 2048:  注意力矩阵 = 4.2M 元素 ✅
  N = 32768: 注意力矩阵 = 1.07B 元素 ❌ 显存爆炸！
  N = 128K:  注意力矩阵 = 16.4B 元素 ❌❌ 

→ 这就是为什么需要 FlashAttention、稀疏注意力等优化
→ FlashAttention 不改变算法，只优化 GPU 内存访问模式
```

### 4.2 为什么 Self-Attention 比 RNN 好？

```
对比维度        Self-Attention     RNN
──────────────────────────────────────
长距离依赖       O(1) 直接连接      O(N) 逐步传递
并行计算         ✅ 完全并行        ❌ 必须顺序
计算复杂度       O(N²·D)           O(N·D²)
最大路径长度     O(1)              O(N)
训练速度         快（GPU 利用率高）  慢（顺序瓶颈）

当 N < D 时（大模型通常如此），Self-Attention 更高效
GPT-3: D=12288, 典型 N=2048 → N < D ✅
```

## 5. 例题（Worked Examples）

### 例题：验证注意力权重的性质

```python
import torch

# 验证自注意力的关键性质
attn = SelfAttention(d_model=64)
x = torch.randn(1, 5, 64)

# 手动提取注意力权重
Q = attn.W_q(x)
K = attn.W_k(x)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5)
weights = F.softmax(scores, dim=-1)

# 性质 1: 每行和为 1
print(f"行和: {weights.sum(dim=-1)}")  # 全为 1.0

# 性质 2: 所有值 ≥ 0
print(f"最小值: {weights.min().item():.6f}")  # ≥ 0

# 性质 3: 带 causal mask 后，上三角为 0
mask = create_causal_mask(5)
scores_masked = scores.masked_fill(mask.squeeze(0).squeeze(0) == 0, float('-inf'))
weights_causal = F.softmax(scores_masked, dim=-1)
print(f"因果掩码后上三角:\n{weights_causal.squeeze().detach()}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 从零实现自注意力，验证输入 [B, N, D] → 输出 [B, N, D] 的形状。

**练习 2：** 解释为什么 Q、K、V 需要三个独立的投影矩阵，而不能共用一个？

### 进阶题

**练习 3：** 实现一个自注意力并输出注意力权重矩阵，用热力图可视化。

**练习 4：** 比较自注意力在有和没有因果掩码时的输出差异，解释 GPT 为什么需要因果掩码。

> **答案：** 因果掩码确保自回归性——生成 token t 时不能看到 t+1, t+2, ... 的信息，否则训练时会"偷看答案"。
