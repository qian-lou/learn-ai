# 序列预测实战 / Sequence Prediction Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 用 LSTM 做文本生成——**预测下一个 token**——这就是 GPT 的前身。GPT 只是用 Transformer Decoder 替换了 LSTM，核心范式完全一致。理解 LSTM 语言模型的训练和生成流程，能帮助你直接理解 GPT/LLaMA 等大模型的工作原理。
>
> **在整个体系中的位置：** 这是 RNN 系列的最后一个实战，也是通往 Transformer 和大模型的桥梁。掌握了"预测下一个 token"的范式，学习 GPT 就只需要理解 Attention 机制即可。

## 2. 知识点（Key Concepts）

| 概念 | LSTM 语言模型 | GPT（对比） |
|------|-------------|------------|
| 核心任务 | 预测下一个 token | 预测下一个 token ✅ |
| 序列编码器 | LSTM | Transformer Decoder |
| 训练方式 | Teacher Forcing | Teacher Forcing ✅ |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss ✅ |
| 生成策略 | Greedy/Top-K/Top-P | Greedy/Top-K/Top-P ✅ |
| 并行训练 | ❌ 顺序处理 | ✅ 完全并行 |

**核心范式（LSTM 和 GPT 完全一样）：**
```
训练：
  输入: [I, love, machine]
  目标: [love, machine, learning]
  → 每个位置预测下一个 token

生成：
  输入: [I]
  → 预测: love → 输入: [I, love]
  → 预测: machine → 输入: [I, love, machine]
  → 预测: learning → ...
```

## 3. 内容（Content）

### 3.1 字符级语言模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 字符级 LSTM 语言模型
# Character-level LSTM Language Model
# ============================================================

class CharLSTM(nn.Module):
    """字符级语言模型 / Character-level language model.
    
    Args:
        vocab_size: 字符表大小 / Character vocabulary size.
        embed_dim: 嵌入维度 / Embedding dimension.
        hidden_dim: LSTM 隐藏维度 / LSTM hidden dimension.
        num_layers: LSTM 层数 / Number of LSTM layers.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: token 序列 / Token sequence. Shape: [B, T]
            hidden: LSTM 隐藏状态 / Hidden state.

        Returns:
            logits: 预测 logits / Prediction logits. Shape: [B, T, V]
            hidden: 更新后的隐藏状态 / Updated hidden state.
        """
        embed = self.embedding(x)           # Shape: [B, T, D]
        output, hidden = self.lstm(embed, hidden)  # Shape: [B, T, H]
        logits = self.fc(output)            # Shape: [B, T, V]
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态 / Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
```

### 3.2 数据准备

```python
# ============================================================
# 文本数据准备流程
# Text data preparation pipeline
# ============================================================

def prepare_char_data(text: str, seq_length: int = 100):
    """将文本转换为训练数据 / Convert text to training data.
    
    Args:
        text: 原始文本 / Raw text.
        seq_length: 序列长度 / Sequence length.
    
    Returns:
        (输入序列, 目标序列, 字符映射) / (inputs, targets, char maps).
    """
    # 构建字符表 / Build character vocabulary
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    # 编码文本 / Encode text
    encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    
    # 切分为 (输入, 目标) 对
    # Split into (input, target) pairs
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_length, seq_length):
        inputs.append(encoded[i:i+seq_length])
        targets.append(encoded[i+1:i+seq_length+1])
    
    return torch.stack(inputs), torch.stack(targets), char_to_idx, idx_to_char

# 示例 / Example
text = "To be or not to be, that is the question. " * 100
X, Y, c2i, i2c = prepare_char_data(text, seq_length=50)
print(f"词表大小: {len(c2i)}, 样本数: {len(X)}")
```

### 3.3 训练循环

```python
# ============================================================
# 训练字符级语言模型
# Training character-level language model
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(c2i)
model = CharLSTM(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(X.size(0), device)
    
    inputs = X.to(device)     # [num_samples, seq_len]
    targets = Y.to(device)    # [num_samples, seq_len]
    
    optimizer.zero_grad()
    # 截断 BPTT: detach 隐藏状态
    hidden = tuple(h.detach() for h in hidden)
    
    logits, hidden = model(inputs, hidden)
    # logits: [N, T, V] → [N*T, V]
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 3.4 文本生成策略

```python
# ============================================================
# 三种生成策略 / Three generation strategies
# ============================================================

@torch.no_grad()
def generate(model, start_text, length, strategy='top_k',
             temperature=1.0, top_k=10, top_p=0.9):
    """文本生成 / Text generation.
    
    Args:
        strategy: 'greedy', 'top_k', or 'top_p'
        temperature: 温度参数（越高越随机）
        top_k: Top-K 采样的 K 值
        top_p: Top-P（Nucleus）采样的阈值
    """
    model.eval()
    device = next(model.parameters()).device
    
    tokens = [c2i[ch] for ch in start_text]
    input_seq = torch.tensor([tokens[-1]], device=device).unsqueeze(0)
    hidden = model.init_hidden(1, device)
    
    generated = list(start_text)
    
    for _ in range(length):
        logits, hidden = model(input_seq, hidden)
        logits = logits[0, -1, :] / temperature  # Shape: [V]
        
        if strategy == 'greedy':
            # 贪心: 选概率最大的 / Greedy: pick highest prob
            next_token = logits.argmax().item()
        
        elif strategy == 'top_k':
            # Top-K: 只从前 K 个候选中采样
            # Top-K: Sample from top K candidates
            top_k_logits, top_k_indices = logits.topk(top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[idx].item()
        
        elif strategy == 'top_p':
            # Top-P (Nucleus): 累积概率达到 p 时截止
            # Top-P: Truncate when cumulative prob reaches p
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = sorted_probs.cumsum(dim=0)
            mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, 1).item()
            next_token = sorted_indices[idx].item()
        
        generated.append(i2c[next_token])
        input_seq = torch.tensor([[next_token]], device=device)
    
    return ''.join(generated)

# 生成示例 / Generation examples
print("=== Greedy ===")
print(generate(model, "To be", 100, strategy='greedy'))
print("\n=== Top-K ===")
print(generate(model, "To be", 100, strategy='top_k', top_k=5))
print("\n=== Top-P ===")
print(generate(model, "To be", 100, strategy='top_p', top_p=0.9))
```

## 4. 详细推理（Deep Dive）

### 4.1 Temperature 的作用

```
Temperature 控制概率分布的"锐利程度"：

  原始 logits: [2.0, 1.0, 0.5]
  
  T=0.1 (低温): softmax → [0.99, 0.01, 0.00] — 几乎确定性
  T=1.0 (默认): softmax → [0.59, 0.24, 0.17] — 正常分布
  T=2.0 (高温): softmax → [0.42, 0.32, 0.26] — 更均匀/随机

  T → 0: 等价于 Greedy（确定性最强）
  T → ∞: 等价于均匀随机采样

GPT 默认 T=1.0，需要创意时用 T=0.7~0.9
```

### 4.2 从 LSTM 到 GPT 的距离

```
LSTM 语言模型 → GPT 只需要两步改变：

1. 将 LSTM 替换为 Transformer Decoder（多头自注意力）
   LSTM: 顺序处理，因果性通过时间步保证
   GPT:  并行处理，因果性通过 causal mask 保证

2. 扩大规模
   LSTM: ~100M 参数，~1M 训练数据
   GPT-3: 175B 参数，~300B token 训练数据

其他完全一样：
  - 预测下一个 token ✅
  - CrossEntropyLoss ✅ 
  - 采样策略（Greedy/Top-K/Top-P）✅
  - Teacher Forcing 训练 ✅
```

## 5. 例题（Worked Examples）

### 例题：多样性采样对比

```python
# 对比不同 temperature 的输出多样性
for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\n--- Temperature = {temp} ---")
    for _ in range(3):
        text = generate(model, "To", 50, strategy='top_p', temperature=temp)
        print(f"  {text}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 LSTM 训练一个字符级语言模型（基于莎士比亚文本），生成 500 个字符的文本。

**练习 2：** 对比 Greedy、Top-K(K=5) 和 Top-P(P=0.9) 三种策略生成的文本质量。

### 进阶题

**练习 3：** 将字符级模型改为**词级（word-level）**模型，对比两者的效果。

**练习 4：** 实现 **Repetition Penalty**（重复惩罚），降低已生成 token 的概率，观察生成文本的变化。

> **参考答案：**
> ```python
> # 在采样前对已出现的 token 做惩罚
> for token_id in set(generated_tokens):
>     logits[token_id] /= repetition_penalty  # penalty > 1.0
> ```
