# LSTM 与 GRU / LSTM and GRU

## 1. 背景（Background）
> LSTM 通过门控机制解决 RNN 的梯度消失问题。GRU 是 LSTM 的简化版。

## 2-3. 知识点与内容
```python
import torch.nn as nn
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
# Bidirectional: Output Shape [B, Seq, 512] (256*2)
# 三个门：遗忘门、输入门、输出门，控制信息的保留和丢弃
```

## 4-6. 推理/例题/习题
**练习：** 用 BiLSTM 做 IMDB 情感分类。
