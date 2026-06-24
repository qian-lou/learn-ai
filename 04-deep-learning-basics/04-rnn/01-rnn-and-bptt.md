# 循环神经网络原理与 BPTT / RNN and BPTT

## 1. 背景（Background）

> **为什么要学这个？**
>
> 循环神经网络（RNN）是深度学习中处理**序列数据（Sequence Data）**的开山鼻祖。尽管目前在大语言模型（LLM）中，Transformer 已经成为统治级的核心架构，但 RNN 的自回归、状态传递和序列依赖的底层思想，仍然深刻影响着当前的语言模型（如 GPT 的自回归解码逻辑）。
>
> 对于 Java 后端工程师来说，RNN 的状态传递机制极其类似于 **Stateful Stream Processing（状态流处理）**系统（例如 Apache Flink / Kafka Streams）。每个传入的时间步（Token）类似于流中的事件，更新当前状态（Hidden State）并输出响应，然后将状态向下游传递。
>
> **在整个体系中的位置：** RNN 是处理时序数据的基础，也是引入门控机制（LSTM/GRU）以及注意力机制（Attention）的基石。

---

## 2. 知识点（Key Concepts）

| 概念 | 英文名称 | 说明 | 空间复杂度 |
| :--- | :--- | :--- | :--- |
| 隐状态 | Hidden State | 存储过去所有时间步历史信息的向量表示 | O(B * H) |
| 随时间反向传播 | BPTT (Backpropagation Through Time) | 在展开后的序列计算图上沿着时间轴进行反向梯度传播的算法 | O(B * T * H) |
| 梯度消失/爆炸 | Vanishing/Exploding Gradient | 序列过长时，反向传播链式法则累乘矩阵导致梯度变零或趋于无穷大 | - |

**核心状态演化方程：**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

---

## 3. 内容（Content）

### 3.1 RNN 计算图展开

```
输入序列:    x_1     ──→    x_2     ──→    x_3
             │              │              │
             ▼              ▼              ▼
隐状态:     h_0  ──→ h_1  ──→ h_2  ──→ h_3
             │              │              │
             ▼              ▼              ▼
输出序列:    y_1            y_2            y_3
```

### 3.2 从零使用 PyTorch Tensor 构建 RNN 单元

以下代码展示如何使用 PyTorch 最基础的 Tensor 操作（而不使用 `nn.RNN`）从零编写 RNN 前向传播算法，加深对权重变换的直观理解。

```python
import torch
from typing import Tuple

class VanillaRNNCell:
    """从零实现的 Vanilla RNN Cell / Vanilla RNN Cell from scratch.
    
    Time: O(H^2 + I * H) - 矩阵投影乘法 / Matrix projections.
    Space: O(H) - 隐状态存储 / Hidden state memory.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重与偏置参数 / Initialize parameters
        # 启用 requires_grad，让 autograd 跟踪参数，才能做 BPTT 反向传播
        # Enable requires_grad so autograd can track params for BPTT
        # Shape: [H, I]
        self.W_xh = (torch.randn(hidden_dim, input_dim) * 0.01).requires_grad_(True)
        # Shape: [H, H]
        self.W_hh = (torch.randn(hidden_dim, hidden_dim) * 0.01).requires_grad_(True)
        # Shape: [H]
        self.b_h = torch.zeros(hidden_dim, requires_grad=True)
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """执行单步前向传播 / Perform a single step forward.
        
        Args:
            x: 当前步输入 / Input tensor. Shape: [B, I]
            h_prev: 上一步隐状态 / Previous hidden state. Shape: [B, H]
            
        Returns:
            当前步隐状态 / Current hidden state. Shape: [B, H]
        """
        # 矩阵相乘，使用转置匹配维度 / Matrix multiplication for state projection
        # projection Shape: [B, H]
        state_proj = torch.matmul(h_prev, self.W_hh.t())
        input_proj = torch.matmul(x, self.W_xh.t())
        
        # 结合偏置并经过 tanh 激活 / Combine and activate
        # h_next Shape: [B, H]
        h_next = torch.tanh(state_proj + input_proj + self.b_h)
        return h_next

# 运行验证单步前向
cell = VanillaRNNCell(input_dim=10, hidden_dim=20)
x_t = torch.randn(2, 10)     # Shape: [B, I]
h_0 = torch.zeros(2, 20)     # Shape: [B, H]
h_1 = cell.forward(x_t, h_0)  # Shape: [B, H]
print(f"输出隐状态形状 / Output Hidden shape: {list(h_1.shape)}")
```

---

## 4. 详细推理（Deep Dive）

### 4.1 BPTT（随时间反向传播）与梯度消失问题

在 RNN 中，在第 $T$ 个时间步的损失 $L_T$ 对隐状态 $h_1$ 的梯度公式包含连乘项：
$$\frac{\partial L_T}{\partial h_1} = \frac{\partial L_T}{\partial h_T} \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}}$$
每一项的 Jacobian 矩阵 $\frac{\partial h_t}{\partial h_{t-1}}$ 包含权重矩阵 $W_{hh}^T$ 的转置与激活函数导数。
当序列步数 $T$ 很大时：
- 若 $W_{hh}$ 特征值最大值小于 1，连乘后梯度会以**指数级速度衰减为 0**（梯度消失 / Vanishing Gradient）。
- 若 $W_{hh}$ 特征值最大值大于 1，连乘后梯度会**呈几何级数爆涨**（梯度爆炸 / Exploding Gradient）。
这解释了为什么循环神经网络处理不了长文本，也是后续诞生门控网络 LSTM/GRU 以及完全抛弃递归的 Transformer 的核心原因。

---

## 5. 例题（Worked Examples）

### 例题 1：使用 PyTorch 内置 nn.RNN 实现简易序列字符预测器 / Sequence prediction using nn.RNN

本例展示如何配置并运行 PyTorch 的 `nn.RNN` 模块，在一组序列上进行完整的前向计算。

```python
import torch
import torch.nn as nn

class SequencePredictor(nn.Module):
    """基于 nn.RNN 的时序预测器 / Sequence predictor based on nn.RNN.
    
    Time: O(S * H^2) per batch / Speed per sequence.
    Space: O(S * H) - 计算图节点存储 / Activation memory.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # batch_first=True 要求输入格式为 [B, S, I] / Expects [B, S, I] input format.
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x Shape: [B, S, I]
        # output Shape: [B, S, H] - 包含每个时间步隐状态的集合
        # h_n Shape: [1, B, H] - 仅包含最后一个时间步的隐状态
        output, h_n = self.rnn(x)
        
        # 将每个时间步的输出通过全连接层分类预测 / Project to vocabulary space
        # logits Shape: [B, S, O]
        logits = self.fc(output)
        return logits, h_n

# 验证模型 / Validate model
model = SequencePredictor(input_dim=8, hidden_dim=16, output_dim=4)
x_batch = torch.randn(2, 5, 8)  # 2 个样本，序列长度为 5，输入维度为 8
logits, hn = model(x_batch)

print(f"输出预测形状 / Output logits shape: {list(logits.shape)}") # [2, 5, 4]
print(f"最终隐状态形状 / Last hidden state shape: {list(hn.shape)}")  # [1, 2, 16]
```

---

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释在 PyTorch 的 `nn.RNN` 中，`batch_first=True` 和默认 `batch_first=False` 时，输入张量形状（Shape）的差异。
*参考答案*：
- `batch_first=True` 时，输入的张量形状为 `[Batch_Size, Sequence_Length, Input_Dimension]`。
- 默认 `batch_first=False` 时，输入的张量形状为 `[Sequence_Length, Batch_Size, Input_Dimension]`。在很多底层循环计算中，后者更符合 GPU 的内存加速习惯，但前者更符合后端开发人员对数据的逻辑认知。

### 进阶题
**练习 2**：已知循环神经网络中梯度爆炸可以通过**梯度裁剪（Gradient Clipping）**有效治理。说明梯度裁剪的运作公式，并在 PyTorch 中编写代码对模型参数进行裁剪。
*参考答案*：
梯度裁剪的公式为：如果梯度范数超过阈值，则按比例缩小：
$$g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|}\right)$$
PyTorch 代码：
```python
# 假设 model 为神经网络，optimizer 为优化器
# logits, loss = model(x, y)
# loss.backward()
# 在 optimizer.step() 调用前进行梯度裁剪，防止爆炸 / Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step()
```
