# 阶段四：深度学习基础

> **预估周期**：3-4 周
> **核心目标**：神经网络原理 + PyTorch 全流程 + CNN/RNN
> **主线**：单神经元 → MLP → 反向传播/优化 → PyTorch 工程化 → CNN(空间) / RNN(时间) → 通往 Transformer

---

## 🗺️ 本阶段主线

一句话串联：**先懂"网络怎么算、梯度怎么回传、参数怎么更新"，再用 PyTorch 把它写成能训练的代码，最后分别攻下图像（CNN）和序列（RNN）两大经典范式，为阶段五的 Transformer 铺路。**

- **01 理论**回答"为什么有效"——非线性激活突破线性限制、反向传播算梯度、优化器决定步长。
- **02 PyTorch**回答"怎么落地"——Tensor、nn.Module、训练循环、GPU/混合精度、模型持久化，这套工程范式后续一直复用。
- **03 CNN**是空间特征提取的代表，其**残差连接**被 Transformer 直接继承。
- **04 RNN**是时间序列建模的开端，其**"预测下一个 token + 采样"**范式正是 GPT 的前身。

---

## 📋 模块大纲

### [01-neural-network-theory](./01-neural-network-theory/) — 神经网络理论

从感知机到反向传播，建立"前向计算 → 反向求梯度 → 优化器更新"的完整心智模型。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [perceptron-and-mlp](./01-neural-network-theory/01-perceptron-and-mlp.md) | 感知机与多层感知机 | XOR 不可分、隐藏层+非线性突破限制、通用近似定理，MLP 即 Transformer 的 FFN |
| 02 | [activation-functions](./01-neural-network-theory/02-activation-functions.md) | 激活函数 | ReLU/Sigmoid/Tanh/GELU/SiLU/Softmax 公式与导数、梯度消失成因、温度参数 |
| 03 | [backpropagation](./01-neural-network-theory/03-backpropagation.md) | 反向传播算法 | 链式法则、计算图、动态图 vs 静态图、autograd、梯度累积/裁剪/检查点 |
| 04 | [optimization-algorithms](./01-neural-network-theory/04-optimization-algorithms.md) | 优化算法 | SGD→Adam→AdamW 演进、Warmup+Cosine 调度、GPT/LLaMA 超参配方 |

### [02-pytorch](./02-pytorch/) — PyTorch 框架

AI 领域主流深度学习框架，掌握模型构建与训练全流程（本阶段工程主干）。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [tensor-basics](./02-pytorch/01-tensor-basics.md) | Tensor 基础与自动求导 | 创建/形状操作/广播/matmul、autograd、stride 与 contiguous |
| 02 | [nn-module](./02-pytorch/02-nn-module.md) | nn.Module 模型构建 | 继承 Module、ModuleList/Dict、参数冻结、train/eval、初始化 |
| 03 | [training-loop](./02-pytorch/03-training-loop.md) | 训练循环与 DataLoader | Dataset/DataLoader、损失函数、五步循环、EarlyStopping、梯度累积 |
| 04 | [gpu-acceleration](./02-pytorch/04-gpu-acceleration.md) | GPU 加速与混合精度 | 设备管理、显存监控、AMP autocast、GradScaler、FP16/BF16/TF32 |
| 05 | [model-save-and-load](./02-pytorch/05-model-save-and-load.md) | 模型保存与加载 | state_dict、Checkpoint、weights_only、safetensors、分片保存 |

### [03-cnn](./03-cnn/) — 卷积神经网络

图像处理的核心架构，理解卷积、池化、残差连接与迁移学习。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [convolution-theory](./03-cnn/01-convolution-theory.md) | 卷积原理 | 卷积/池化、输出尺寸公式、感受野、参数共享、卷积 vs 全连接 vs Attention |
| 02 | [classic-architectures](./03-cnn/02-classic-architectures.md) | 经典架构 | LeNet→ResNet 演进、残差连接、Bottleneck、Pre/Post-Norm |
| 03 | [image-classification-practice](./03-cnn/03-image-classification-practice.md) | 图像分类实战 | CIFAR-10/MNIST 端到端、数据增强、迁移学习三策略、Mixup |

### [04-rnn](./04-rnn/) — 循环神经网络

序列数据处理的经典方法，NLP 与自回归生成的重要前置知识。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [rnn-and-bptt](./04-rnn/01-rnn-and-bptt.md) | RNN 原理与 BPTT | 隐状态传递、时间展开、BPTT 连乘导致梯度消失/爆炸、梯度裁剪 |
| 02 | [lstm-and-gru](./04-rnn/02-lstm-and-gru.md) | LSTM 与 GRU | 遗忘/输入/输出门、cell state 信息高速公路、GRU 简化、遗忘门偏置技巧 |
| 03 | [sequence-prediction-practice](./04-rnn/03-sequence-prediction-practice.md) | 序列预测实战 | 字符级语言模型、Teacher Forcing、截断 BPTT、Greedy/Top-K/Top-P 采样 |

---

## 🎯 学习要点

- **手推一次反向传播 + autograd 对照**：这是整个阶段最该内化的能力，能解释梯度消失/爆炸、梯度累积/裁剪等一切训练技巧。
- **把五步训练循环练成肌肉记忆**：`zero_grad → forward → loss → backward → step`，从 MNIST 到 GPT 结构一致，是后续所有项目的骨架。
- **形状注释是硬习惯**：每个 Tensor 标 `# Shape: [B, S, D]`，尤其多头注意力的 split/merge，读写大模型代码全靠它。
- **抓住两条贯穿到 Transformer 的主线**：CNN 的**残差连接**、RNN 的**下一 token 预测 + 采样**范式，正是 GPT 能堆深层、能自回归生成的直接来源。
- **记住大模型标准配方**：AdamW + Warmup + Cosine Decay + 梯度裁剪 + 混合精度(优先 BF16)，并能对照 GPT-3/LLaMA 的具体超参与显存账。
- **跑通两个端到端实战**：CIFAR-10 图像分类（自训 CNN → 迁移 ResNet）与字符级 LSTM 语言模型（训练 → Top-K/Top-P 生成），把理论转成可运行、可复现的结果。

---

## 🔗 关联

- **上一阶段**：[阶段三 · 机器学习基础](../03-machine-learning-basics/README.md)（线性/逻辑回归是单层网络的特例，本阶段将其推广到多层非线性）
- **下一阶段**：[阶段五 · 自然语言处理基础](../05-nlp-fundamentals/README.md)（词嵌入、Seq2Seq 与注意力，直接承接本阶段 RNN 的序列建模）
- **关联 Day**：本阶段的张量/训练思维支撑 [agent-course Day-16 embedding-basics](../agent-course/Day-16-embedding-basics.md) 起的向量化实践；RNN 的采样范式支撑 [Day-02 model-params](../agent-course/Day-02-model-params.md) 中 temperature/top_p 等解码参数的理解。
