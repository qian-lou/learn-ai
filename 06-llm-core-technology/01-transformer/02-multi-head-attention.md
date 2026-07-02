# 多头注意力 / Multi-Head Attention

## 1. 背景（Background）

> **为什么要学这个？**
>
> Multi-Head Attention 是 Transformer 的**核心组件**。它将自注意力扩展为多个并行的"注意力头"，每个头关注输入的不同方面——有的头关注语法关系，有的关注语义关系，有的关注位置关系。
>
> 类比：单头注意力像一只眼睛看世界，多头注意力像一个复眼——从多个角度同时观察，综合得到更丰富的信息。
>
> 对于 Java 工程师来说，多头 ≈ 一次 fork-join：把 d_model 切成 h 份交给 h 个"worker"（头）各算各的注意力，最后 concat 合并——类似 `parallelStream().map(...)` 并行处理再汇总。
>
> **在整个体系中的位置：** Multi-Head Attention 是 Transformer Block 的第一个子层。GPT-3 有 96 个头，BERT-base 有 12 个头。

## 2. 知识点（Key Concepts）

| 模型 | d_model | n_heads | d_head | 层数 |
|------|---------|---------|--------|------|
| BERT-base | 768 | 12 | 64 | 12 |
| BERT-large | 1024 | 16 | 64 | 24 |
| GPT-2 | 768 | 12 | 64 | 12 |
| GPT-3 | 12288 | 96 | 128 | 96 |
| LLaMA-7B | 4096 | 32 | 128 | 32 |

**公式：** `MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · W_O`

## 3. 内容（Content）

### 3.1 多头注意力的工作原理

```
Multi-Head Attention 过程：

输入 X: [B, N, d_model=768]

Step 1: 线性投影 + 拆分头
  Q = X · W_q → [B, N, 768] → reshape → [B, 12, N, 64]
  K = X · W_k → [B, N, 768] → reshape → [B, 12, N, 64]
  V = X · W_v → [B, N, 768] → reshape → [B, 12, N, 64]

Step 2: 每个头独立做 Scaled Dot-Product Attention
  head_1: Attention(Q₁, K₁, V₁) → [B, N, 64]
  head_2: Attention(Q₂, K₂, V₂) → [B, N, 64]
  ...
  head_12: Attention(Q₁₂, K₁₂, V₁₂) → [B, N, 64]

Step 3: 拼接所有头 + 输出投影
  Concat → [B, N, 768]  →  W_O → [B, N, 768]

关键：12 个头**共用一次**大矩阵乘法，通过 reshape 实现并行
  实际上不是做 12 次小矩阵乘法，而是做 1 次大矩阵乘法后 reshape
```

### 3.2 从零实现 Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 完整实现.
    
    Args:
        d_model: 模型维度 / Model dimension.
        n_heads: 注意力头数 / Number of attention heads.
        dropout: Dropout 概率 / Dropout probability.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 一次大投影代替多次小投影（效率更高）
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Shape [B, N, D]
            mask: Shape [B, 1, N, N] or [1, 1, N, N]
        Returns:
            output: Shape [B, N, D]
        """
        B, N, D = x.shape
        
        # 1. 一次投影得到 Q, K, V
        qkv = self.W_qkv(x)  # [B, N, 3D]
        Q, K, V = qkv.chunk(3, dim=-1)  # 各 [B, N, D]
        
        # 2. 拆分头: [B, N, D] → [B, H, N, d_head]
        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention（每个头独立计算）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: [B, H, N, N]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, d_head]
        
        # 4. 合并头: [B, H, N, d_head] → [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # 5. 输出投影
        output = self.W_o(attn_output)  # [B, N, D]
        
        return output

# 测试
mha = MultiHeadAttention(d_model=768, n_heads=12)
x = torch.randn(2, 128, 768)
out = mha(x)
print(f"输入: {x.shape} → 输出: {out.shape}")  # [2, 128, 768]
print(f"参数量: {sum(p.numel() for p in mha.parameters()):,}")  # 2,359,296
```

### 3.3 GQA（Grouped Query Attention）

```
现代大模型的注意力优化：

MHA (Multi-Head Attention):
  每个头有独立的 Q, K, V
  GPU 显存: O(3 × n_heads × d_head)

MQA (Multi-Query Attention):
  所有头共享一组 K, V，只有 Q 是独立的
  GPU 显存: O(n_heads × d_head + 2 × d_head)
  → 推理速度提升，但质量略降

GQA (Grouped Query Attention) — LLaMA-2/3 使用:
  将 n_heads 分成 n_groups 组
  每组内共享 K, V，Q 独立
  介于 MHA 和 MQA 之间的折中
  
  例: 32 heads, 8 groups → 每 4 个 Q head 共享 1 组 KV
  显存减少 75%，质量几乎无损
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么需要多个头？

```
单头注意力的限制：
  比如 "The animal didn't cross the street because it was too tired"
  需要同时理解：
  - "it" 指代 "animal"（语义关系）
  - "tired" 修饰 "it"（语法关系）
  - "cross" 和 "street" 搭配（搭配关系）
  
  单头：一次只能建模一种关系
  多头：每个头专注一种关系，综合后得到完整理解

实验发现不同head 学到的模式：
  Head 3: 关注相邻位置（局部特征）
  Head 7: 关注动词-宾语关系
  Head 11: 关注指代关系（it→animal）
```

### 4.2 参数量计算

```
Multi-Head Attention 参数量:
  W_q: d_model × d_model = D²
  W_k: d_model × d_model = D²
  W_v: d_model × d_model = D²
  W_o: d_model × d_model = D²
  总计: 4D²

BERT-base (D=768): 4 × 768² = 2,359,296 ≈ 2.4M
GPT-3 (D=12288): 4 × 12288² = 604M（仅注意力部分！）
```

## 5. 例题（Worked Examples）

### 例题：对比不同头数的效果

```python
# 对比 head 数量对参数量和推理时间的影响
for n_heads in [1, 4, 8, 12, 16]:
    mha = MultiHeadAttention(d_model=768, n_heads=n_heads)
    params = sum(p.numel() for p in mha.parameters())
    print(f"heads={n_heads:2d}, d_head={768//n_heads:3d}, params={params:,}")
# 参数量相同！区别在于每个头的维度和多样性
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 从零实现 Multi-Head Attention，验证 n_heads=1 时退化为 Self-Attention。

*参考答案*：当 `n_heads=1` 时，`d_head = d_model`，reshape 后头维度为 1，`Q/K/V` 的形状从 `[B, N, D]` 变为 `[B, 1, N, D]`，注意力计算等价于单头自注意力。验证方法是用同一份 `W_qkv/W_o` 权重，分别走"多头分支"和"单头朴素自注意力"，比较输出。

```python
import torch, math
import torch.nn.functional as F

torch.manual_seed(0)
B, N, D = 2, 5, 16
x = torch.randn(B, N, D)  # Shape: [B, N, D]
Wq, Wk, Wv, Wo = (torch.randn(D, D) for _ in range(4))

# 分支 A：单头朴素自注意力 / Plain single-head self-attention
Q, K, V = x @ Wq, x @ Wk, x @ Wv
attn = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(D), dim=-1)
out_plain = (attn @ V) @ Wo  # Shape: [B, N, D]

# 分支 B：MHA(n_heads=1)，用同一组权重走多头路径
# MHA path with a single head, sharing the same weights
n_heads, d_head = 1, D // 1  # d_head == d_model
Qh = (x @ Wq).view(B, N, n_heads, d_head).transpose(1, 2)  # Shape: [B, 1, N, D]
Kh = (x @ Wk).view(B, N, n_heads, d_head).transpose(1, 2)  # Shape: [B, 1, N, D]
Vh = (x @ Wv).view(B, N, n_heads, d_head).transpose(1, 2)  # Shape: [B, 1, N, D]
scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_head)   # Shape: [B, 1, N, N]
attn_h = F.softmax(scores, dim=-1)
ctx = (attn_h @ Vh).transpose(1, 2).contiguous().view(B, N, D)  # 合并头 -> [B, N, D]
out_mha = ctx @ Wo  # Shape: [B, N, D]

# n_heads=1 时头维度为 1，多头路径与单头朴素自注意力逐元素相等
# With a single head, both branches produce identical outputs
assert torch.allclose(out_plain, out_mha, atol=1e-6)
```

**练习 2：** 为什么 d_model 必须能被 n_heads 整除？

*参考答案*：多头是把 `d_model` 维向量**等分**给 `n_heads` 个头，每个头维度 `d_head = d_model / n_heads`。代码里 `view(B, N, n_heads, d_head)` 要求 `n_heads × d_head == d_model`，若不整除则 reshape 无法把 `d_model` 拆成整齐的 `[n_heads, d_head]`，且拼接后也无法还原回 `d_model`。因此实现中用 `assert d_model % n_heads == 0` 强制约束。

### 进阶题

**练习 3：** 实现 GQA（Grouped Query Attention），对比 MHA 和 GQA 的推理速度。

*参考答案*：GQA 让 Q 保留 `n_heads` 个头，K/V 只用 `n_kv_heads`（`n_kv_heads < n_heads`）个头，再把每个 KV 头**广播**给同组的多个 Q 头。核心是 `repeat_kv`：

```python
def repeat_kv(kv, n_rep):
    """把 KV 头复制 n_rep 份 / Repeat each KV head n_rep times.
    kv: [B, n_kv_heads, N, d_head] -> [B, n_kv_heads*n_rep, N, d_head]
    """
    B, H_kv, N, Dh = kv.shape
    # 扩展中间维度后再展平 / Expand then flatten, no real copy in attention
    return kv[:, :, None, :, :].expand(B, H_kv, n_rep, N, Dh).reshape(B, H_kv * n_rep, N, Dh)
# n_rep = n_heads // n_kv_heads
```

推理加速来源于 **KV Cache 显存大幅缩小**（KV 头数从 `n_heads` 降到 `n_kv_heads`），显存带宽是自回归解码的瓶颈，故 GQA 在长上下文/大 batch 解码时吞吐明显高于 MHA，质量几乎无损。注意：本节代码的 `[2, 128, 768]` 短序列前向计算量主要在矩阵乘，GQA 的优势需在**带 KV Cache 的逐 token 解码**场景才显著。

**练习 4：** 提取 BERT 不同层、不同头的注意力权重，分析它们分别学到了什么模式。

*参考答案*：加载时设 `output_attentions=True`，`outputs.attentions` 是长度为层数的 tuple，每个元素形状 `[B, n_heads, N, N]`。

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
inputs = tok("The cat sat on the mat", return_tensors="pt")
attn = model(**inputs).attentions  # tuple(len=12), 每个 [B, 12, N, N]
layer3_head7 = attn[3][0, 7]       # Shape: [N, N]
```

典型发现（与 Clark et al. 2019 "What Does BERT Look At?" 一致）：部分头关注**相邻 token**（局部），部分头几乎全部权重落在 `[SEP]`/`[CLS]` 上（"无操作"头），深层出现指代消解、动宾、介词搭配等**语法/语义**模式；浅层偏表层位置关系，深层偏长程依赖。
