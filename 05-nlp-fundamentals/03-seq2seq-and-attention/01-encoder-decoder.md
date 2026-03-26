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

**练习 2：** 解释 Teacher Forcing 的优缺点，以及 Exposure Bias 问题。

### 进阶题

**练习 3：** 对比不同 teacher_forcing_ratio（0.0, 0.5, 1.0）对训练效果的影响。

**练习 4：** 将 Encoder 改为 Bidirectional LSTM，观察翻译质量的变化。
