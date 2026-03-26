# 从零实现 Transformer / Transformer from Scratch

## 1. 背景（Background）

> **为什么要学这个？**
>
> 阅读论文和看代码是两回事——**动手实现**才能真正理解 Transformer 的每一个细节。本节将从零构建一个完整的 GPT-style Transformer，不使用任何高级封装（如 `nn.TransformerEncoder`）。
>
> 这也是 AI 面试的高频考点：能否从零写出一个可运行的 Transformer？
>
> **在整个体系中的位置：** 这是前四节知识的综合实践。完成这个实现后，你将具备阅读 LLaMA、GPT 等开源模型源码的能力。

## 2. 知识点（Key Concepts）

```
完整 GPT 的组件清单：

1. Token Embedding (nn.Embedding)
2. Position Embedding (nn.Embedding 或 RoPE)
3. N × Transformer Block:
   a. LayerNorm
   b. Multi-Head Causal Self-Attention
   c. Residual Connection
   d. LayerNorm
   e. Feed-Forward Network (SwiGLU/GELU)
   f. Residual Connection
4. Final LayerNorm
5. Language Model Head (nn.Linear → vocab_size)
```

## 3. 内容（Content）

### 3.1 完整 GPT 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 组件 1: Multi-Head Causal Self-Attention
# Component 1: Multi-Head Causal Self-Attention
# ============================================================

class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力.
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int,
                 max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 预计算因果掩码 / Precompute causal mask
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len))
    
    def forward(self, x):
        B, N, D = x.shape
        
        # QKV 投影 + 拆分头
        qkv = self.qkv_proj(x)  # [B, N, 3D]
        Q, K, V = qkv.split(D, dim=-1)
        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, H, N, d_head]
        
        # Scaled Dot-Product Attention + Causal Mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:, :, :N, :N] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 加权聚合 + 合并头
        out = torch.matmul(attn, V)  # [B, H, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.resid_dropout(self.out_proj(out))


# ============================================================
# 组件 2: Feed-Forward Network
# Component 2: Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    """前馈网络（GELU 版本）."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 组件 3: Transformer Block (Pre-Norm)
# Component 3: Transformer Block (Pre-Norm)
# ============================================================

class Block(nn.Module):
    """Transformer Block with Pre-Norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Attention + 残差
        x = x + self.ffn(self.ln2(x))   # FFN + 残差
        return x


# ============================================================
# 完整 GPT 模型
# Complete GPT Model
# ============================================================

class GPT(nn.Module):
    """从零实现的 GPT 模型 / GPT model from scratch.
    
    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 层数
        d_ff: FFN 中间维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # 嵌入层
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享: embedding 和 output head 共用权重
        self.token_emb.weight = self.head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: token IDs. Shape: [B, N]
            targets: 目标 token IDs. Shape: [B, N]
        
        Returns:
            logits: Shape [B, N, vocab_size]
            loss: 交叉熵损失（如提供 targets）
        """
        B, N = idx.shape
        
        # Token + Position Embedding
        tok_emb = self.token_emb(idx)  # [B, N, D]
        pos_emb = self.pos_emb(torch.arange(N, device=idx.device))  # [N, D]
        x = self.dropout(tok_emb + pos_emb)
        
        # N 层 Transformer Block
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)  # [B, N, vocab_size]
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        
        return logits, loss
```

### 3.2 训练循环

```python
# ============================================================
# 训练 Mini GPT / Train Mini GPT
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT(
    vocab_size=256,     # 字符级
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    max_len=64,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 准备数据（字符级语言建模）
text = "To be or not to be, that is the question. " * 200
data = torch.tensor([ord(c) for c in text], dtype=torch.long)

# 训练
for step in range(200):
    idx = torch.randint(0, len(data) - 64, (32,))
    x = torch.stack([data[i:i+64] for i in idx]).to(device)
    targets = torch.stack([data[i+1:i+65] for i in idx]).to(device)
    
    logits, loss = model(x, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

### 3.3 文本生成

```python
@torch.no_grad()
def generate(model, start, max_new_tokens=100, temperature=0.8):
    """自回归生成 / Autoregressive generation."""
    model.eval()
    idx = torch.tensor([[ord(c) for c in start]], device=device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -64:])  # 最多看 64 个 token
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    
    return ''.join(chr(i) for i in idx[0].tolist() if 0 < i < 128)

print(generate(model, "To be"))
```

## 4. 详细推理（Deep Dive）

### 4.1 权重共享（Weight Tying）

```
GPT/LLaMA 中 embedding 和 output head 共享权重：
  self.token_emb.weight = self.head.weight

为什么？
  Embedding: token_id → vector (查表)
  Output Head: vector → logits (分类)
  
  它们是"互逆"操作，共享权重减少参数量
  vocab_size=50000, d_model=4096 → 节省 200M 参数
```

### 4.2 与真实 LLaMA 的差异

```
本节实现 vs LLaMA 的区别:

  位置编码: Learned Embedding → RoPE
  FFN: GELU → SwiGLU
  归一化: LayerNorm → RMSNorm
  注意力: MHA → GQA
  精度: FP32 → BF16  
  训练: 单卡 → 分布式（FSDP/DeepSpeed）
  
  但核心架构完全一致！
```

## 5. 例题（Worked Examples）

### 例题：计算模型 FLOPs

```python
def estimate_flops(vocab_size, d_model, n_layers, n_heads, seq_len, d_ff):
    """估算前向传播 FLOPs."""
    # Attention: 4 * N * D^2 + 2 * N^2 * D
    attn_flops = 4 * seq_len * d_model**2 + 2 * seq_len**2 * d_model
    # FFN: 2 * N * D * D_ff
    ffn_flops = 2 * seq_len * d_model * d_ff
    # 每层
    layer_flops = attn_flops + ffn_flops
    total = n_layers * layer_flops
    return total

flops = estimate_flops(50000, 4096, 32, 32, 2048, 11008)
print(f"LLaMA-7B 单次前向 FLOPs: {flops/1e12:.1f} TFLOPs")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 运行本节的 GPT 代码，在莎士比亚文本上训练，生成一段文本。

**练习 2：** 修改模型超参数（层数、维度），观察参数量和生成质量的变化。

### 进阶题

**练习 3：** 将 FFN 替换为 SwiGLU，将 LayerNorm 替换为 RMSNorm，使其更接近 LLaMA。

**练习 4：** 添加 KV Cache 优化推理速度——在生成时缓存已计算的 K 和 V。

> **提示：** KV Cache 是推理加速的核心技术，避免重复计算前面 token 的 K 和 V。
