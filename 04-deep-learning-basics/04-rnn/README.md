# 04-rnn — 循环神经网络

> **所属阶段**：阶段四 · 深度学习基础
> **学习目标**：理解序列建模原理，掌握 LSTM/GRU 门控机制，打通到 GPT 的桥梁
> **预估时长**：4-5 天（本子模块）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [rnn-and-bptt](./01-rnn-and-bptt.md) | RNN 原理与 BPTT | 隐状态传递、计算图时间展开、BPTT 连乘导致梯度消失/爆炸、梯度裁剪、nn.RNN 用法 |
| 02 | [lstm-and-gru](./02-lstm-and-gru.md) | LSTM 与 GRU | 遗忘/输入/输出门、cell state 信息高速公路、GRU 简化、遗忘门偏置初始化技巧 |
| 03 | [sequence-prediction-practice](./03-sequence-prediction-practice.md) | 序列预测实战 | 字符级语言模型、Teacher Forcing、截断 BPTT、Greedy/Top-K/Top-P 采样、温度参数 |

---

## 🔑 知识点详解

### 01 · RNN 原理与 BPTT
- **核心概念**：RNN 用一个随时间传递的隐状态 `h_t` 压缩全部历史，逐步吃入序列；BPTT 是在按时间展开的计算图上做反向传播。
- **关键公式/API**：`h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)`，`y_t = W_hy·h_t + b_y`；梯度含连乘 `∏ ∂h_t/∂h_{t-1}`；`nn.RNN(input_size, hidden_size, batch_first=True)` 返回 `(output[B,S,H], h_n[layers,B,H])`。
- **易错点**：① 长序列上 `W_hh` 反复连乘导致梯度**指数消失/爆炸**（这是 RNN 处理不了长文本的根因）；② `batch_first=True` 输入是 `[B,S,I]`，默认 `False` 是 `[S,B,I]`，喂错维度会静默出错；③ 梯度爆炸要用 `clip_grad_norm_` 治理，但裁剪治不了梯度消失。
- **Java 视角**：RNN 的状态传递 ≈ 有状态流处理（Flink/Kafka Streams）——每个时间步像一个事件，更新当前状态并输出，再把状态传给下游。
- **前置**：01-neural-network-theory/03（BPTT 就是在时间展开图上跑反向传播）。

### 02 · LSTM 与 GRU
- **核心概念**：LSTM 用门控 + 独立的 cell state 提供"信息高速公路"，让梯度沿记忆通道近乎无衰减传播，解决 RNN 的梯度消失。
- **关键公式/API**：`fₜ,iₜ,oₜ = σ(W·[hₜ₋₁,xₜ])`，`cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ`，`hₜ = oₜ⊙tanh(cₜ)`；GRU 合并为更新门 z、重置门 r，无 cell state，参数少约 25%；`nn.LSTM/nn.GRU(..., bidirectional=True)`。
- **易错点**：① 梯度沿 cell state 的路径是 `∂cₜ/∂cₜ₋₁ = fₜ`——遗忘门 ≈1 时才畅通，故常把遗忘门偏置初始化为 1~2 加速长依赖学习；② LSTM 返回 `(output, (h_n, c_n))` 两个状态，GRU 只返回 `(output, h_n)`，替换时别忘改解包；③ 双向 LSTM 输出维度是 `hidden×2`，且 `h_n` 第一维为 `num_layers×2`。
- **Java 视角**：门控 ≈ 阀门控制系统——遗忘门决定丢弃哪些旧信息、输入门决定写入哪些新信息、输出门决定放出哪些信息。
- **前置**：01（LSTM 是为修复 RNN 梯度消失而生）。

### 03 · 序列预测实战
- **核心概念**：LSTM 语言模型"预测下一个 token"就是 GPT 的前身——把 LSTM 换成 Transformer Decoder、放大规模，范式完全一致。
- **关键 API**：`Embedding → LSTM → Linear(→vocab)`；训练用 Teacher Forcing + `CrossEntropyLoss(logits.view(-1,V), targets.view(-1))`；长序列用截断 BPTT（`hidden = tuple(h.detach() ...)`）；生成用 `logits/temperature` + Greedy/Top-K(`topk`)/Top-P(`sort+cumsum`) + `multinomial` 采样。
- **易错点**：① 自回归生成前要用整段前缀**预热隐状态**（除最后一个 token 外全部喂入），否则丢上下文；② 不 detach 隐状态会让计算图无限增长、显存爆掉；③ 温度 T→0 退化为 Greedy、T→∞ 趋于均匀随机，需创意时用 0.7~0.9。
- **Java 视角**：自回归生成 ≈ 有状态的流式生成器——每次产出一个 token 并更新内部状态，作为下一次输入。
- **前置**：02（用 LSTM 作序列编码器）、02-pytorch/03（训练循环）。

---

## 🎯 学习要点

- **理解 RNN 梯度消失的数学根源**：`∏ ∂h_t/∂h_{t-1}` 连乘，`W_hh` 特征值 <1 指数衰减、>1 指数爆炸——这是 LSTM/GRU 乃至 Transformer 出现的动机。
- **说清 LSTM 为何能缓解梯度消失**：核心是 cell state 的梯度路径 `∂cₜ/∂cₜ₋₁ = fₜ`，遗忘门开着就直通，而非像 RNN 那样每步过 tanh + 矩阵乘。
- **掌握遗忘门偏置初始化技巧**：初始化为正值让 cell state 默认"倾向保留"，是长依赖任务收敛更快的经典小技巧。
- **打通 LSTM → GPT 的范式**："预测下一个 token + CrossEntropyLoss + Teacher Forcing + Top-K/Top-P 采样"在两者中完全相同，学 GPT 只剩理解 Attention。
- **动手实现三种采样并对比**：Greedy/Top-K/Top-P 配合温度 T，观察生成文本在确定性与多样性上的差异。
- **理性看待 RNN 的历史地位**：Transformer 已在多数任务上超越 LSTM/GRU，但它们仍是理解序列建模、状态传递与自回归解码的必经之路。

---

## 🔗 关联

- **上一模块**：[03-cnn](../03-cnn/README.md)（从空间特征提取转向时间序列建模）
- **下一模块**：[阶段五 · 自然语言处理基础](../../05-nlp-fundamentals/README.md)，尤其 [03-seq2seq-and-attention](../../05-nlp-fundamentals/03-seq2seq-and-attention/README.md)（Seq2Seq 建立在 LSTM/GRU 之上，注意力机制正是为突破循环的长依赖瓶颈而生）
- **本阶段总览**：[阶段四 · 深度学习基础](../README.md)
- **关联 Day**：本模块的"下一 token 预测 + Top-K/Top-P/温度采样"范式，直接支撑 [agent-course Day-02 model-params](../../agent-course/Day-02-model-params.md) 中对 temperature/top_p 等解码参数的理解。
