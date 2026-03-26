# RNN 原理与 BPTT / RNN and BPTT

## 1. 背景（Background）
> RNN 是序列建模的经典架构，虽被 Transformer 取代，但理解其序列建模思想有助于理解语言模型。

## 2-3. 知识点与内容
```python
import torch.nn as nn
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
# Input: [B, Seq, 128] -> Output: [B, Seq, 256]
# 问题：长序列上梯度消失/爆炸 → 这就是 Transformer 要解决的问题
```

## 4-6. 推理/例题/习题
**练习：** 理解为什么 RNN 无法处理长序列（梯度消失）。
