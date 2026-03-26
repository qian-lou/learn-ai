# Transformer 完整架构 / Complete Transformer Architecture

## 1. 背景（Background）

> **为什么要学这个？**
>
> Transformer 是**现代 AI 的基石**。从 GPT-4 到 Sora，从 AlphaFold 到 Stable Diffusion，几乎所有最先进的 AI 模型都基于 Transformer 架构。理解其完整架构，就掌握了理解一切大模型的钥匙。
>
> 对于 Java 工程师来说，Transformer 就像 **Spring 框架**——一个高度模块化的架构，由标准化的组件（Attention、FFN、LayerNorm、残差连接）组合而成。每个组件各司其职，通过标准接口连接。
>
> **在整个体系中的位置：** 这是 Transformer 系列的汇总，将前几节的组件组装成完整模型。

## 2. 知识点（Key Concepts）

| 组件 | 功能 | 类比 |
|------|------|------|
| Embedding | token → 向量 | 字典查找 |
| Positional Encoding | 注入位置信息 | 数组索引 |
| Multi-Head Attention | 动态全局信息聚合 | 全文检索 |
| Feed-Forward Network | 非线性特征变换 | MLP |
| Layer Normalization | 稳定训练 | 批归一化 |
| Residual Connection | 梯度高速公路 | 快捷方式 |

## 3. 内容（Content）

### 3.1 完整架构图

```
┌─────────────── Transformer Architecture ───────────────┐
│                                                        │
│  ┌── Encoder ──────────┐   ┌── Decoder ──────────────┐ │
│  │                     │   │                         │ │
│  │  Input Embedding    │   │  Output Embedding       │ │
│  │  + Pos Encoding     │   │  + Pos Encoding         │ │
│  │                     │   │                         │ │
│  │  ┌───────────────┐  │   │  ┌───────────────────┐  │ │
│  │  │ Multi-Head    │  │   │  │ Masked Multi-Head │  │ │
│  │  │ Self-Attention│  │   │  │ Self-Attention     │  │ │
│  │  │ + Add & Norm  │  │   │  │ + Add & Norm      │  │ │
│  │  ├───────────────┤  │   │  ├───────────────────┤  │ │
│  │  │ Feed-Forward  │  │   │  │ Cross Attention   │  │ │
│  │  │ Network       │  │   │  │ (Q=Dec, KV=Enc)  │  │ │
│  │  │ + Add & Norm  │  │   │  │ + Add & Norm      │  │ │
│  │  └───────────────┘  │   │  ├───────────────────┤  │ │
│  │       × N 层        │   │  │ Feed-Forward      │  │ │
│  │                     │   │  │ Network           │  │ │
│  └─────────────────────┘   │  │ + Add & Norm      │  │ │
│                            │  └───────────────────┘  │ │
│                            │       × N 层            │ │
│                            │                         │ │
│                            │  Linear + Softmax       │ │
│                            └─────────────────────────┘ │
└────────────────────────────────────────────────────────┘

三种变体：
  BERT  = 只用 Encoder（双向理解）→ 分类、NER、问答
  GPT   = 只用 Decoder（自回归生成）→ 对话、写作、推理
  T5    = 完整 Encoder-Decoder → 翻译、摘要
```

### 3.2 Transformer Block 详解

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """单个 Transformer Block.
    
    组成: Multi-Head Attention → Add & Norm → FFN → Add & Norm
    
    Args:
        d_model: 模型维度 / Model dimension.
        n_heads: 注意力头数 / Number of heads.
        d_ff: FFN 中间维度 / FFN intermediate dimension.
        dropout: Dropout 概率 / Dropout probability.
    """
    
    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        # 子层 1: Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # 子层 2: Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),       # 扩展 4 倍
            nn.GELU(),                       # 激活函数
            nn.Linear(d_ff, d_model),        # 投影回原维度
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Shape [B, N, D]
        Returns:
            x: Shape [B, N, D]
        """
        # Pre-Norm 变体（GPT-2/LLaMA 使用，比 Post-Norm 更稳定）
        # 子层 1: Attention + 残差
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)  # 残差连接
        
        # 子层 2: FFN + 残差
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # 残差连接
        
        return x
```

### 3.3 FFN 的作用

```
Feed-Forward Network 的角色：

  FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂
  
  W₁: [d_model, d_ff]    即 [768, 3072] → 扩展 4 倍
  W₂: [d_ff, d_model]    即 [3072, 768] → 压缩回来

为什么需要 FFN？
  Attention 是线性操作（加权求和）
  FFN 引入非线性变换 → 提供特征变换能力
  
  类比：
  Attention = "信息收集"（你看到了什么？）
  FFN       = "信息处理"（你理解了什么？）

现代变体（LLaMA 使用 SwiGLU）：
  SwiGLU(x) = (x · W₁ ⊙ Swish(x · W_gate)) · W₂
  → 比标准 FFN 效果更好
```

### 3.4 Pre-Norm vs Post-Norm

```
Post-Norm（原始 Transformer）：
  x → Attention → x + Attention(x) → LayerNorm → FFN → ...
  问题：深层模型训练不稳定

Pre-Norm（GPT-2/LLaMA 等现代模型）：
  x → LayerNorm → Attention → x + Attention(LN(x)) → LayerNorm → FFN → ...
  优势：训练更稳定，不需要 warmup

几乎所有现代大模型都使用 Pre-Norm
```

## 4. 详细推理（Deep Dive）

### 4.1 Transformer 参数量计算

```
单个 Transformer Block 参数量：

  Multi-Head Attention: 4 × D² (W_q, W_k, W_v, W_o)
  FFN: 2 × D × D_ff (W₁, W₂)
  LayerNorm: 2 × 2D (两个 LN, 各有 γ 和 β)
  
  BERT-base (D=768, D_ff=3072):
    Attention: 4 × 768² = 2,359,296
    FFN: 2 × 768 × 3072 = 4,718,592
    LN: 4 × 768 = 3,072
    每层总计: ≈ 7.08M
    
  12 层 × 7.08M + Embedding(30522×768) ≈ 110M ✅

  GPT-3 (D=12288, D_ff=49152, 96 层):
    每层: 4 × 12288² + 2 × 12288 × 49152 ≈ 1.81B
    96 层: ≈ 175B ✅
```

### 4.2 残差连接为什么关键？

```
没有残差连接：
  信号经过 96 层变换后可能完全改变
  梯度需要反向传播 96 层 → 梯度消失

有残差连接：
  x_out = x + F(x)
  梯度: ∂x_out/∂x = 1 + ∂F(x)/∂x
  
  即使 ∂F(x)/∂x 很小，梯度仍有 "1" 保底
  → 梯度可以畅通无阻地流过任意深度
  
  这就是为什么 Transformer 可以堆到 96+ 层
  而没有残差连接的网络通常 < 20 层
```

## 5. 例题（Worked Examples）

### 例题：构建完整 GPT-style 模型

```python
class MiniGPT(nn.Module):
    """最小化 GPT 模型 / Minimal GPT model."""
    
    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 n_layers=4, d_ff=1024, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, idx):
        B, N = idx.shape
        x = self.embedding(idx) + self.pos_encoding(torch.arange(N, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=idx.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, N, vocab_size]
        return logits

model = MiniGPT(vocab_size=10000)
params = sum(p.numel() for p in model.parameters())
print(f"参数量: {params:,}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 画出 BERT、GPT、T5 三种模型的架构差异图。

**练习 2：** 计算 LLaMA-7B（32 层，d=4096，d_ff=11008）的参数量。

### 进阶题

**练习 3：** 实现 Pre-Norm 和 Post-Norm 两种 TransformerBlock，对比训练稳定性。

**练习 4：** 用 SwiGLU 替换标准 FFN，对比模型效果。
