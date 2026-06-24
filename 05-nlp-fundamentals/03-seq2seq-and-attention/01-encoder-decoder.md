# 编码器-解码器架构 / Encoder-Decoder Architecture

## 1. 背景（Background）

> **为什么要学这个？**
>
> Encoder-Decoder 是 **Transformer 的基础架构**。Encoder 将输入序列编码为上下文表示，Decoder 根据上下文生成输出序列。几乎所有 NLP 生成任务（翻译、摘要、问答）都基于这个框架。
>
> 对于 Java 工程师来说，Encoder-Decoder 就像是**编解码器模式**——Encoder 就是序列化器（Serializer），将输入"编码"为中间表示；Decoder 就是反序列化器（Deserializer），将中间表示"解码"为输出。
>
> **模型变体：** BERT = 纯 Encoder（理解任务），GPT = 纯 Decoder（生成任务），T5/BART = Encoder-Decoder（翻译/摘要）。
>
> **在整个体系中的位置：** RNN Seq2Seq → RNN + Attention → Transformer Encoder-Decoder → 纯 Decoder（GPT）。

## 2. 知识点（Key Concepts）

| 架构 | 代表模型 | 输入 | 输出 | 典型任务 |
|------|---------|------|------|----------|
| Encoder-only | BERT | 文本 | 向量表示 | 分类、NER |
| Decoder-only | GPT | 前缀 | 续写文本 | 生成、对话 |
| Encoder-Decoder | T5, BART | 源序列 | 目标序列 | 翻译、摘要 |

## 3. 内容（Content）

### 3.1 RNN Seq2Seq 基本结构

```
Seq2Seq 架构：

Encoder（编码器）:
  x₁ → x₂ → x₃ → [EOS]
  ↓     ↓     ↓      ↓
 RNN → RNN → RNN → RNN → context vector (c)
                            ↓
Decoder（解码器）:          ↓
  [BOS] → y₁ → y₂ → y₃ → [EOS]
   ↓      ↓    ↓    ↓
  RNN → RNN → RNN → RNN
   ↓      ↓    ↓    ↓
   y₁     y₂   y₃  [EOS]（预测的输出）

问题：
  整个输入序列被压缩为一个固定大小的 context vector
  → 信息瓶颈！长序列信息严重丢失
  → 这就是 Attention 机制要解决的问题
```

### 3.2 PyTorch 实现 Seq2Seq

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """LSTM 编码器 / LSTM Encoder."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, src):
        """
        Args:
            src: 源序列 / Source sequence. Shape: [B, src_len]
        Returns:
            outputs: 每步输出 / Step outputs. Shape: [B, src_len, H]
            (h, c): 最终隐藏状态 / Final hidden state.
        """
        embedded = self.embedding(src)          # [B, src_len, D]
        outputs, (h, c) = self.lstm(embedded)   # [B, src_len, H]
        return outputs, (h, c)


class Decoder(nn.Module):
    """LSTM 解码器 / LSTM Decoder."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, tgt, hidden):
        """
        Args:
            tgt: 目标序列 / Target sequence. Shape: [B, 1]（一步）
            hidden: 编码器最终状态 / Encoder final state.
        Returns:
            output: 预测 logits / Prediction logits. Shape: [B, 1, V]
        """
        embedded = self.embedding(tgt)         # [B, 1, D]
        output, hidden = self.lstm(embedded, hidden)  # [B, 1, H]
        logits = self.fc(output)               # [B, 1, V]
        return logits, hidden


class Seq2Seq(nn.Module):
    """完整 Seq2Seq 模型 / Complete Seq2Seq model."""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Teacher Forcing: 训练时以一定概率使用真实目标作为下一步输入
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        _, hidden = self.encoder(src)
        
        # 第一个输入是 [BOS] token
        input_token = tgt[:, 0:1]
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher Forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t:t+1]  # 使用真实目标
            else:
                input_token = output.argmax(-1)  # 使用模型预测
        
        return outputs
```

### 3.3 Teacher Forcing

```
Teacher Forcing 训练策略：

没有 Teacher Forcing（自回归）：
  Input:  [BOS]     → 预测 y₁*   → 预测 y₂*   → ...
  如果 y₁* 错了，后续全错（错误累积/Exposure Bias）

有 Teacher Forcing：
  Input:  [BOS]     → y₁(真实)   → y₂(真实)   → ...
  训练时使用真实目标作为下一步输入
  → 训练更稳定、收敛更快

问题：训练时用真实目标，推理时用预测结果 → 训练/推理不一致
解决：用 teacher_forcing_ratio 在 0~1 之间随机切换
```

## 4. 详细推理（Deep Dive）

### 4.1 信息瓶颈问题

```
Seq2Seq 的根本问题：

  输入: "I love natural language processing very much"（7 个词）
    ↓
  压缩为一个固定大小的 context vector（如 256 维）
    ↓
  解码器必须从 256 维向量中恢复所有信息

  序列越长，信息损失越严重
  这就像把一本书的内容压缩到一个句子中

解决方案 → Attention 机制：
  不只看最终的 context vector
  而是让 Decoder 的每一步都可以"回头看" Encoder 的每一步
  → 信息不再被压缩到固定大小
```

## 5. 例题（Worked Examples）

### 例题：构建数字翻译模型

```python
# 翻译数字序列（如 "1 2 3" → "3 2 1" 反转任务）
# 这是验证 Seq2Seq 模型的经典 toy task

src_vocab_size = 12  # 0-9 + BOS + EOS
tgt_vocab_size = 12

encoder = Encoder(src_vocab_size, 32, 64)
decoder = Decoder(tgt_vocab_size, 32, 64)
device = torch.device('cpu')
model = Seq2Seq(encoder, decoder, device)

# 训练数据: "1 2 3 EOS" → "BOS 3 2 1 EOS"
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现一个 Seq2Seq 模型完成序列反转任务（输入 [1,2,3]，输出 [3,2,1]）。

*参考答案*：

直接复用本文 3.2 节的 `Encoder/Decoder/Seq2Seq`，只需准备"反转"数据并训练：

```python
import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 10, 11          # 0-9 是数字, 10/11 是特殊符
V = 12

def make_batch(batch=64, length=5):
    """生成 (src, tgt)：tgt 是 src 数字段的反转 / build reversal pairs."""
    digits = torch.randint(0, 10, (batch, length))      # [B, L]
    src = torch.cat([digits, torch.full((batch, 1), EOS)], 1)        # [B, L+1]
    rev = torch.flip(digits, dims=[1])                  # 反转数字段
    tgt = torch.cat([torch.full((batch, 1), BOS), rev,
                     torch.full((batch, 1), EOS)], 1)   # [B, L+2]: BOS rev EOS
    return src, tgt

device = torch.device('cpu')
model = Seq2Seq(Encoder(V, 32, 64), Decoder(V, 32, 64), device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss(ignore_index=PAD)

for step in range(2000):
    src, tgt = make_batch()
    logits = model(src, tgt, teacher_forcing_ratio=0.5)   # [B, T, V]
    # 用 tgt[:,1:] 作为标签（错位一位）/ shift target by one
    loss = crit(logits[:, 1:].reshape(-1, V), tgt[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```

要点：序列反转是验证 Seq2Seq 是否跑通的经典 toy task，模型必须把整段输入"记住"后再倒序吐出，正好考验 context vector 的承载能力；损失计算时目标要**错位一位**（用 `tgt[:,1:]`，因为第 0 位是 BOS 输入而非预测目标），并用 `ignore_index` 跳过 PAD。短序列上很快能达到接近 100% 的正确率。

**练习 2：** 解释 Teacher Forcing 的优缺点，以及 Exposure Bias 问题。

*参考答案*：

**Teacher Forcing（训练时用真实目标 yₜ₋₁ 作为下一步输入）：**

优点：
- **收敛快、训练稳定**：每一步的输入都是正确的，模型不会因前一步预测错误而让后续输入全盘崩坏。
- **可并行/高效**：标签已知，训练时不依赖上一步的采样结果（在 Transformer 中整条序列可一次并行算完）。

缺点：
- **训练-推理不一致**：训练时喂的是"真实历史"，推理时只能喂"模型自己生成的历史"，两种分布不同。

**Exposure Bias（曝光偏差）：**
- 定义：模型在训练中**从未见过自己生成的（可能有误的）前缀**，只见过真实前缀；推理时一旦某步预测错误，就进入了训练时没暴露过的状态分布，模型不知如何纠正，导致**误差沿序列累积、越错越离谱**。
- 这正是"训练用真实目标、推理用预测结果"带来的根本矛盾。

**常见缓解手段：**
- **Scheduled Sampling**：训练中以一定概率（如本文的 `teacher_forcing_ratio`）改用模型自己的预测作为输入，让模型提前适应自身错误，并在训练过程中逐步降低该概率。
- 序列级训练目标（如最小化风险 / RL with BLEU 等）直接在生成序列上优化。
- Transformer 时代曝光偏差影响相对减弱（强语言建模 + beam search 部分缓解），但本质问题依然存在。

### 进阶题

**练习 3：** 对比不同 teacher_forcing_ratio（0.0, 0.5, 1.0）对训练效果的影响。

*参考答案*：

固定模型与数据，只改 `teacher_forcing_ratio`，记录收敛速度（训练 loss 曲线）和**推理时**（全自回归，ratio=0）的序列准确率。

| ratio | 训练表现 | 推理表现 | 说明 |
|-------|---------|---------|------|
| **1.0**（全用真实）| 收敛**最快**、训练 loss 最低 | 受 **Exposure Bias** 影响，推理时易累积误差，泛化可能偏弱 | 训练/推理差距最大 |
| **0.0**（全自回归）| 收敛**最慢**，早期一步错步步错，训练困难甚至学不动 | 训练即推理，无分布差距，但因训练太难效果未必好 | 难优化 |
| **0.5**（混合）| 收敛较快且较稳 | **通常最佳**：兼顾"训练好优化"和"提前暴露自身错误" | 实践常用折中 |

结论：`ratio=1.0` 训练最快但最易曝光偏差；`ratio=0.0` 最贴近推理却最难训练；**中间值（如 0.5，即 Scheduled Sampling）往往综合最好**。进一步常用做法是**从高 ratio 退火到低 ratio**——前期多用真实目标快速学到基本能力，后期多用自身预测来适应推理分布。最终评估一定要在 `ratio=0`（真实推理设置）下进行，否则会高估模型。

**练习 4：** 将 Encoder 改为 Bidirectional LSTM，观察翻译质量的变化。

*参考答案*：

双向 Encoder 的关键是处理两个维度变化：(1) 输出维度变成 `2*H`；(2) 前后向各有一份 (h, c)，要合并后再交给单向 Decoder。

```python
import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    """双向 LSTM 编码器 / Bidirectional LSTM Encoder."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        # 把 2*H 投影回 H，供单向 Decoder 使用 / project 2H -> H
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):                          # src: [B, src_len]
        emb = self.embedding(src)
        outputs, (h, c) = self.lstm(emb)             # outputs: [B, L, 2H]; h/c: [2, B, H]
        # 拼接前向(h[0])与后向(h[1]) 再投影 / concat both directions
        h = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], 1))).unsqueeze(0)  # [1,B,H]
        c = torch.tanh(self.bridge_c(torch.cat([c[0], c[1]], 1))).unsqueeze(0)  # [1,B,H]
        return outputs, (h, c)
```

质量变化与原因：**通常翻译/序列任务质量提升**。单向 Encoder 编码位置 t 时只看过 `x₁..xₜ`；双向则同时看到**左右两侧的完整上下文**，每个位置的表示信息更充分（这正是 BERT 双向优于 GPT 单向理解的同款直觉）。代价：参数和计算约翻倍，且必须把双向的 (h,c) 合并维度匹配 Decoder。注意 Decoder 仍须保持单向（自回归生成不能看未来）。配合 Attention 时，Encoder 的 `[B, L, 2H]` 输出还能作为更丰富的 attention 记忆，收益更明显。
