# LSTM 与 GRU / LSTM and GRU

## 1. 背景（Background）

> **为什么要学这个？**
>
> LSTM（长短期记忆网络）通过**门控机制**解决了 RNN 的梯度消失问题。GRU（门控循环单元）是 LSTM 的简化版本，参数更少且效果相当。对于 Java 工程师来说，门控机制类似于**阀门控制系统**——遗忘门决定"丢弃哪些旧信息"，输入门决定"写入哪些新信息"，输出门决定"输出哪些信息"。
>
> 虽然 Transformer 已经取代了 LSTM，但 LSTM 的**门控思想**（选择性信息保留）仍然影响着现代架构设计。
>
> **在整个体系中的位置：** LSTM 是序列建模从 RNN 到 Transformer 的中间站。它证明了门控机制的有效性，为 Attention 机制的出现奠定了基础。

## 2. 知识点（Key Concepts）

| 组件 | LSTM | GRU | 说明 |
|------|------|-----|------|
| 遗忘门 (Forget Gate) | ✅ fₜ | ❌ | 决定丢弃哪些信息 |
| 输入门 (Input Gate) | ✅ iₜ | ❌ | 决定存储哪些新信息 |
| 输出门 (Output Gate) | ✅ oₜ | ❌ | 决定输出哪些信息 |
| 更新门 (Update Gate) | ❌ | ✅ zₜ | 合并遗忘+输入 |
| 重置门 (Reset Gate) | ❌ | ✅ rₜ | 控制历史信息 |
| 记忆单元 (Cell State) | ✅ cₜ | ❌ | 长期记忆通道 |
| 参数量 | 4 × (D² + D·H) | 3 × (D² + D·H) | GRU 少 25% |

## 3. 内容（Content）

### 3.1 LSTM 结构详解

```
LSTM Cell 内部结构：

         cₜ₋₁ ────────× ────────⊕───────→ cₜ (记忆通道)
                       │          │
                    fₜ(遗忘)   iₜ⊙c̃ₜ(写入)
                       │          │
  ┌────────────────────┤          ├────────────┐
  │     ┌──────┐  ┌────┴───┐ ┌───┴──┐  ┌─────┐│
  │     │Forget│  │ Input  │ │ Cell │  │Output││
  │     │ Gate │  │ Gate   │ │Update│  │ Gate ││
  │     │  fₜ  │  │  iₜ   │ │  c̃ₜ │  │  oₜ ││
  │     └──┬───┘  └───┬───┘ └──┬───┘  └──┬──┘│
  │        │          │        │         │    │
  └────────┼──────────┼────────┼─────────┼────┘
           │          │        │         │
      hₜ₋₁ + xₜ  hₜ₋₁ + xₜ  hₜ₋₁ + xₜ  │
                                         │
                                   oₜ ⊙ tanh(cₜ)
                                         │
                                         ▼
                                        hₜ

公式：
  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)      遗忘门
  iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)      输入门
  c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)   候选记忆
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ        更新记忆
  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)      输出门
  hₜ = oₜ ⊙ tanh(cₜ)               隐藏状态
```

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch LSTM 使用
# Using PyTorch LSTM
# ============================================================
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True,  # 双向 LSTM
    dropout=0.1,
)

x = torch.randn(32, 50, 128)  # [B, T, D]
# 初始状态 / Initial states
h0 = torch.zeros(4, 32, 256)  # [num_layers*2(双向), B, hidden]
c0 = torch.zeros(4, 32, 256)  # cell state

output, (h_n, c_n) = lstm(x, (h0, c0))
print(f"输出: {output.shape}")      # [32, 50, 512] — 256*2(双向)
print(f"最终 h: {h_n.shape}")       # [4, 32, 256]
print(f"最终 c: {c_n.shape}")       # [4, 32, 256]


# ============================================================
# 从零实现 LSTM Cell
# LSTM Cell from scratch
# ============================================================
class SimpleLSTMCell(nn.Module):
    """从零实现 LSTM Cell."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # 4 个门共享一个大矩阵乘法（效率更高）
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)  # [B, input+hidden]
        gates = self.gates(combined)               # [B, 4*hidden]
        
        # 拆分四个门
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # 输入门
        f = torch.sigmoid(f)   # 遗忘门
        g = torch.tanh(g)      # 候选记忆
        o = torch.sigmoid(o)   # 输出门
        
        c = f * c_prev + i * g  # 更新记忆
        h = o * torch.tanh(c)   # 输出隐藏状态
        return h, c
```

### 3.2 GRU 结构

```python
import torch.nn as nn

# ============================================================
# GRU: LSTM 的简化版
# GRU: Simplified version of LSTM
# ============================================================
# GRU 用更新门(z)和重置门(r)替代 LSTM 的三个门
# 没有独立的 cell state，参数量减少 25%

gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
)

x = torch.randn(32, 50, 128)
output, h_n = gru(x)  # GRU 没有 cell state
print(f"GRU 输出: {output.shape}")  # [32, 50, 512]

# GRU 公式：
# zₜ = σ(Wz·[hₜ₋₁, xₜ])        更新门
# rₜ = σ(Wr·[hₜ₋₁, xₜ])        重置门
# h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁, xₜ])   候选状态
# hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ      最终状态
```

### 3.3 LSTM vs GRU vs RNN 对比

```
选择建议：
  RNN:  几乎不用了（梯度消失太严重）
  LSTM: 经典可靠，适合需要长期记忆的任务
  GRU:  参数更少，速度更快，效果与 LSTM 接近
  
  实际上：现在直接用 Transformer，除非有特殊限制
  
参数量对比（input=128, hidden=256）：
  RNN:   (128+256)*256 + 256     = 98,560
  GRU:   3 × [(128+256)*256 + 256] = 295,680
  LSTM:  4 × [(128+256)*256 + 256] = 394,240
  注：以上为教科书公式（单 bias）；PyTorch nn.LSTM/GRU 每门含 ih/hh 两个 bias，
      实际参数多 K×hidden（K=4/3），用 count_parameters 验证会略高
```

## 4. 详细推理（Deep Dive）

### 4.1 LSTM 如何解决梯度消失？

```
关键：cell state 提供了"信息高速公路"

RNN 的梯度路径（每步都要经过 tanh 和矩阵乘法）：
  ∂hₜ/∂hₜ₋₁ = diag(1-hₜ²) · Wₕₕ  ← 每步都衰减

LSTM 的梯度路径（通过 cell state 直通）：
  ∂cₜ/∂cₜ₋₁ = fₜ                    ← 遗忘门的值！
  
  如果 fₜ ≈ 1（遗忘门打开），梯度直接通过，不衰减
  如果 fₜ ≈ 0，信息被主动遗忘（这是期望的行为）
  
  LSTM 偏置初始化技巧：将遗忘门偏置初始化为 1~2
  → 初始时 fₜ ≈ σ(1~2) ≈ 0.73~0.88 → 梯度畅通
```

## 5. 例题（Worked Examples）

### 例题：BiLSTM 情感分类

```python
class SentimentClassifier(nn.Module):
    """BiLSTM 情感分类器 / BiLSTM Sentiment Classifier."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # [B, T, D]
        output, (h_n, _) = self.lstm(embedded)       # h_n: [2, B, H]
        # 拼接双向最终状态
        hidden = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, 2H]
        return self.classifier(self.dropout(hidden))
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 对比 LSTM 三个门在训练过程中的激活值分布，验证遗忘门通常接近 1。

*参考答案*：

思路：在 `SimpleLSTMCell.forward` 里把 `i, f, o` 三个门的值记录下来（detach 后存盘），训练若干步后画直方图对比。

```python
import torch

@torch.no_grad()
def collect_gate_stats(cell, x_seq, h, c):
    """逐步前向并收集三个门的均值 / Collect gate means over time.

    x_seq: [T, B, input_size]
    """
    means = {"i": [], "f": [], "o": []}
    for x_t in x_seq:                                   # x_t: [B, input_size]
        combined = torch.cat([x_t, h], dim=1)
        i, f, g, o = cell.gates(combined).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        means["i"].append(i.mean().item())
        means["f"].append(f.mean().item())             # 关注遗忘门 / focus on forget
        means["o"].append(o.mean().item())
    return means
# 训练后画 hist(means["f"])，可观察到 f 的分布整体偏向 0.7~1.0
```

预期结论：在能学好长依赖的 LSTM 上，**遗忘门 f 的激活整体偏高（多数接近 1）**，因为模型需要把有用信息长期保留在 cell state 中；而输入门 i、输出门 o 的分布通常更分散（在 0~1 之间按需开合）。这与梯度路径 `∂cₜ/∂cₜ₋₁ = fₜ` 一致：f≈1 才能让梯度沿记忆通道无衰减地传播。

**练习 2：** 用 GRU 替换 BiLSTM 情感分类器中的 LSTM，对比参数量和准确率。

*参考答案*：

只需把 `nn.LSTM` 换成 `nn.GRU`，并注意 GRU 没有 cell state（只返回 `h_n`）：

```python
import torch
import torch.nn as nn

class GRUSentimentClassifier(nn.Module):
    """把 BiLSTM 换成 BiGRU / Swap BiLSTM for BiGRU."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):                       # x: [B, T]
        embedded = self.dropout(self.embedding(x))    # [B, T, D]
        output, h_n = self.gru(embedded)              # GRU 只有 h_n（无 cell）
        hidden = torch.cat([h_n[0], h_n[1]], dim=1)   # [B, 2H]
        return self.classifier(self.dropout(hidden))
```

对比结论：
- **参数量**：循环层 GRU 比 LSTM **少约 25%**（3 组门权重 vs 4 组）。本文 3.3 节给出 input=128, hidden=256 时 GRU 295,680 vs LSTM 394,240，恰好少 1/4。
- **准确率**：在情感分类这类中短序列任务上，**两者差距通常很小**（常在 ±1% 内波动）。GRU 参数更少、训练更快，资源受限时是更划算的选择；LSTM 在需要超长期记忆的任务上偶尔略占优。

### 进阶题

**练习 3：** 实现遗忘门偏置初始化为 1.0 的 LSTM，对比默认初始化在长序列上的效果差异。

*参考答案*：

PyTorch 的 `nn.LSTM` 把 4 个门偏置拼成一个张量，顺序为 `[i, f, g, o]`，遗忘门 f 位于**第 2 个 1/4 段**。注意 PyTorch 有 `bias_ih` 和 `bias_hh` 两套偏置，二者相加才是有效偏置，所以一般把其中一套的 f 段设为 1.0：

```python
import torch.nn as nn

def set_forget_bias(lstm: nn.LSTM, value: float = 1.0):
    """将遗忘门偏置初始化为 value / Init forget-gate bias.

    PyTorch 门顺序 = [input, forget, cell, output]，f 在 [H:2H] 区间。
    """
    for name, param in lstm.named_parameters():
        if "bias_ih" in name:                  # 只改一套即可生效
            H = param.size(0) // 4
            with torch.no_grad():
                param[H:2 * H].fill_(value)    # forget gate segment

lstm = nn.LSTM(128, 256, batch_first=True)
set_forget_bias(lstm, 1.0)
```

长序列上的差异：默认偏置为 0 时 `f = σ(0) = 0.5`，初期记忆每步约衰减一半，长序列上梯度仍偏弱、训练慢且容易丢失早期信息；把遗忘门偏置设为 1（甚至 1~2）后，初期 `f ≈ σ(1) ≈ 0.73`，cell state 默认"倾向于保留"，梯度沿记忆通道更畅通。结论：**遗忘门偏置初始化为正值能加速收敛、显著改善长依赖任务的表现**，是一个被广泛采用的小技巧（早期实现甚至默认这么做）。

**练习 4：** 从零实现 GRU Cell，参照 LSTM Cell 的实现方式。

*参考答案*：

GRU 有更新门 z、重置门 r 和候选状态 h̃。注意重置门 r 作用在 `h_prev` 上后才参与候选计算，所以候选用的权重要单独算（不能与 z、r 共享同一个矩阵乘）：

```python
import torch
import torch.nn as nn

class SimpleGRUCell(nn.Module):
    """从零实现 GRU Cell / GRU Cell from scratch."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # z 和 r 两个门共享一次矩阵乘 / gates z & r in one matmul
        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        # 候选状态 h̃ 单独的权重（输入用 r⊙h_prev）/ candidate weights
        self.cand = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):                  # x:[B,in]  h_prev:[B,hidden]
        combined = torch.cat([x, h_prev], dim=1)
        z, r = self.gates(combined).chunk(2, dim=1)
        z = torch.sigmoid(z)                       # 更新门 / update gate
        r = torch.sigmoid(r)                       # 重置门 / reset gate
        # 候选状态：用 r 门控历史 / candidate uses reset-gated history
        cand_in = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.cand(cand_in))
        # 最终状态：在旧状态与候选间插值 / interpolate old vs new
        h = (1 - z) * h_prev + z * h_tilde
        return h
```

对照公式（与本文 3.2 节一致）：`zₜ=σ(Wz·[h,x])`、`rₜ=σ(Wr·[h,x])`、`h̃ₜ=tanh(W·[r⊙h, x])`、`hₜ=(1−z)⊙h + z⊙h̃`。相比 LSTM Cell：少了一个独立 cell state c，把"遗忘+输入"合并为一个更新门 z，因此参数量更少。
