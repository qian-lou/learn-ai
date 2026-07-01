# 03-cnn — 卷积神经网络

> **所属阶段**：阶段四 · 深度学习基础
> **学习目标**：理解卷积操作原理，掌握经典 CNN 架构与图像分类完整流程
> **预估时长**：5-6 天（本子模块）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [convolution-theory](./01-convolution-theory.md) | 卷积原理 | 卷积/池化运算、输出尺寸公式、感受野、参数共享、1×1 卷积、卷积 vs 全连接 vs Attention |
| 02 | [classic-architectures](./02-classic-architectures.md) | 经典架构 | LeNet→AlexNet→VGG→GoogLeNet→ResNet 演进、残差连接、Bottleneck、Pre/Post-Norm |
| 03 | [image-classification-practice](./03-image-classification-practice.md) | 图像分类实战 | CIFAR-10/MNIST 端到端训练、数据增强、迁移学习三策略、Mixup |

---

## 🔑 知识点详解

### 01 · 卷积原理
- **核心概念**：卷积 = 一个小卷积核在输入上滑动、逐位置做"逐元素乘再求和"，靠**局部感受野 + 参数共享**高效提取平移不变特征。
- **关键公式/API**：输出尺寸 `Output = (Input - Kernel + 2·Padding) / Stride + 1`；`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`，输入形状 `[B, C, H, W]`；`nn.MaxPool2d`、`nn.AdaptiveAvgPool2d(1)`（全局平均池化）。
- **易错点**：① 卷积参数量 = `out_ch × in_ch × K² (+out_ch bias)`，与输入空间尺寸**无关**（这正是参数共享的省法）；② `padding=1` 配 `3×3, stride=1` 才能保持尺寸不变，别记混；③ `1×1` 卷积不改变空间尺寸，只在通道维做线性变换（用于升/降维）。
- **Java 视角**：卷积 ≈ 滑动窗口过滤器，类似用正则在字符串上做定长滑窗匹配、每个窗口提取局部特征。
- **前置**：01-neural-network-theory（卷积是特殊的、带权重共享的线性层 + 激活）。

### 02 · 经典架构
- **核心概念**：CNN 演进史的最大遗产是**残差连接** `H(x) = F(x) + x`——被 Transformer 直接采用，是深层网络能训练的关键。
- **关键公式/API**：残差块 `out = ReLU(BN(Conv(...)) + identity)`；Bottleneck 用 `1×1 降维 → 3×3 → 1×1 升维` 控制 FLOPs；`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`。
- **易错点**：① ResNet 解决的是**优化困难/退化**（深层训练误差更高），不是过拟合——56 层网络理论上能表示 20 层的解，只是 SGD 学不到恒等映射；② 残差分支形状要对齐，下采样/改通道时 shortcut 需配 `1×1 conv`；③ 现代大模型用 **Pre-Norm**（`x + F(LN(x))`，恒等通路干净、常免 warmup），原始 Transformer 是 Post-Norm。
- **Java 视角**：残差连接 ≈ 在深层调用链里加一条"旁路直连"，让原始输入/信号绕过中间处理直达下游，避免层层衰减。
- **前置**：01（残差块由卷积 + BN + 激活组成）。

### 03 · 图像分类实战
- **核心概念**：图像分类是深度学习的 Hello World——串起数据增强、DataLoader、CNN、训练/评估的完整流水线，范式可直接迁移到 NLP。
- **关键 API**：`transforms.Compose([RandomCrop, RandomHorizontalFlip, ToTensor, Normalize(mean,std)])`；`torchvision.datasets.CIFAR10/MNIST`；迁移学习靠控制 `param.requires_grad` + 替换 `model.fc`；`classification_report`/`confusion_matrix`。
- **易错点**：① 数据增强**只加在训练集**，测试集保持原样，否则评估失真；② 归一化必须用对应数据集的 mean/std（CIFAR-10 与 ImageNet 不同），预训练模型要 Resize 到 224 并用 ImageNet 统计量，否则严重掉点；③ 全微调时 backbone 要用更小学习率，否则会破坏预训练特征。
- **Java 视角**：训练流水线 ≈ ETL/责任链——DataLoader 是分批拉数的 Iterator，预处理→前向→反向→评估是流水线上依次流过的工序。
- **前置**：02-pytorch 全部（训练循环、GPU、保存）、本模块 01+02。

---

## 🎯 学习要点

- **卷积是"局部连接 + 参数共享"的特征提取器**：理解它相对全连接省了多少参数（224×224 图一层可省近 1800 倍），以及为什么这对自然图像有效。
- **背熟输出尺寸公式并会反推**：能手算 `stride=2` 下采样、ResNet 首层 `7×7/s2/p3`、ViT Patch Embedding `16×16/s16` 各自的输出尺寸。
- **说清残差连接为何能训深层网络**：关键在 `∂xₗ₊₁/∂xₗ = ∂F/∂xₗ + 1` 里那个"+1"让梯度不消失——这直接解释了 GPT-3 为何能堆 96 层。
- **打通 CNN 与 Transformer 的桥**：ViT 的 Patch Embedding 本质是一个大 stride 卷积；残差连接、归一化在两者中一脉相承。
- **跑通一个端到端项目**：在 CIFAR-10 上从零训练 CNN 到 85%+，再用预训练 ResNet 迁移学习冲 90%+，对比自训 vs 迁移的成本与效果。
- **掌握迁移学习三策略取舍**：只训分类头 / 全微调 / 逐层解冻，能根据数据量与域相似度选择。

---

## 🔗 关联

- **上一模块**：[02-pytorch](../02-pytorch/README.md)（本模块直接复用其训练循环与保存机制）
- **下一模块**：[04-rnn](../04-rnn/README.md)（从空间序列转向时间序列建模）
- **本阶段总览**：[阶段四 · 深度学习基础](../README.md)
