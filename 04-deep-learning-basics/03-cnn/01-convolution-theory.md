# 卷积原理 / Convolution Theory

## 1. 背景（Background）

> **为什么要学这个？**
>
> 卷积神经网络（CNN）虽然在 NLP 领域已被 Transformer 取代，但理解卷积操作对学习大模型仍然重要。首先，**Vision Transformer（ViT）的 Patch Embedding 本质上就是一个卷积操作**；其次，卷积的核心思想——**局部特征提取+参数共享**——贯穿整个深度学习。
>
> 对于 Java 工程师来说，卷积可以理解为一个**滑动窗口过滤器**——想象一个小窗口在图像（或文本序列）上滑动，每到一个位置就提取局部特征。这和 Java 中用正则表达式在字符串上做滑动窗口匹配很类似。
>
> **在整个体系中的位置：** CNN 是 MLP 和 Transformer 之间的桥梁。理解卷积的"局部感受野"和"权重共享"，才能理解 Transformer 用 Attention 替代卷积的本质原因。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | 与 Transformer 的关系 |
|------|------|----------------------|
| 卷积核 (Kernel) | 在输入上滑动的小滤波器 | Attention 的 Q/K/V 投影 |
| 步长 (Stride) | 滑动窗口的移动步幅 | Patch Embedding 的 patch_size |
| 填充 (Padding) | 边缘补零以保持尺寸 | Padding mask |
| 池化 (Pooling) | 降低空间分辨率 | Mean/CLS Pooling |
| 感受野 (Receptive Field) | 每个输出看到的输入范围 | Attention 的全局感受野 |
| 通道数 (Channels) | 特征图的深度 | 嵌入维度 d_model |

**核心公式：**
```
输出尺寸 = (输入尺寸 - 卷积核大小 + 2 × 填充) / 步长 + 1
Output = (Input - Kernel + 2 × Padding) / Stride + 1
```

## 3. 内容（Content）

### 3.1 卷积操作详解

```
2D 卷积操作（以 3×3 卷积核为例）：

输入（5×5）:              卷积核（3×3）:
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │     │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┤     └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤     操作：逐元素乘法后求和
│ 1 │ 0 │ 1 │ 0 │ 1 │     Result = Σ(input ⊙ kernel)
└───┴───┴───┴───┴───┘

输出（3×3）:    (5-3+0)/1 + 1 = 3
```

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch 中的卷积操作
# Convolution in PyTorch
# ============================================================

# 2D 卷积层 / 2D Convolution layer
conv = nn.Conv2d(
    in_channels=3,       # 输入通道数（RGB=3）/ Input channels
    out_channels=64,     # 输出通道数（即卷积核个数）/ Output channels
    kernel_size=3,       # 卷积核大小 3×3 / Kernel size
    stride=1,            # 步长 / Stride
    padding=1,           # 填充（padding=1 保持尺寸不变）/ Padding
    bias=True,           # 是否包含偏置 / Include bias
)

# 输入: [B, C_in, H, W] = [1, 3, 224, 224]
x = torch.randn(1, 3, 224, 224)
output = conv(x)  # Shape: [1, 64, 224, 224]

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"卷积核形状: {conv.weight.shape}")  # [64, 3, 3, 3]
print(f"参数量: {sum(p.numel() for p in conv.parameters())}")
# 参数量 = 64 × 3 × 3 × 3 + 64 (bias) = 1,792


# ============================================================
# 不同参数的效果 / Effects of different parameters
# ============================================================
# stride=2: 输出尺寸减半（下采样）
# stride=2: Output size halved (downsampling)
conv_stride2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
out_s2 = conv_stride2(x)
print(f"stride=2 输出: {out_s2.shape}")  # [1, 64, 112, 112]

# 1×1 卷积: 只在通道维度做线性变换（不改变空间尺寸）
# 1×1 conv: Linear transform only in channel dim
conv_1x1 = nn.Conv2d(3, 64, kernel_size=1)
out_1x1 = conv_1x1(x)
print(f"1×1 卷积输出: {out_1x1.shape}")  # [1, 64, 224, 224]
```

### 3.2 池化操作

```python
import torch.nn as nn

# ============================================================
# 池化层：降低空间分辨率
# Pooling: Reduce spatial resolution
# ============================================================

# 最大池化 / Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
x = torch.randn(1, 64, 224, 224)
out = max_pool(x)  # Shape: [1, 64, 112, 112]

# 平均池化 / Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 全局平均池化（GAP）：将整个特征图压缩为一个值
# Global Average Pooling: Compress entire feature map to single value
gap = nn.AdaptiveAvgPool2d(1)
out_gap = gap(x)  # Shape: [1, 64, 1, 1] — 每个通道一个值
out_flat = out_gap.view(out_gap.size(0), -1)  # Shape: [1, 64]
```

### 3.3 卷积 vs 全连接 vs Attention

```
三种特征提取方式对比：

全连接 (MLP):
  - 每个输出与所有输入连接
  - 参数量: O(N²)
  - 无空间结构先验
  
卷积 (CNN):
  - 每个输出只与局部输入连接（局部感受野）
  - 参数共享：同一卷积核在所有位置使用
  - 参数量: O(K²·C_in·C_out)，与输入大小无关
  - 平移不变性

Attention (Transformer):
  - 每个输出与所有输入连接（全局感受野）
  - 参数量: O(D²)，与序列长度无关
  - 位置编码提供位置信息
  - 内容感知（Content-aware）

结论：
  CNN = 局部 + 参数共享 → 高效但受限
  Attention = 全局 + 内容感知 → 强大但计算量大
  ViT 的 Patch Embedding = 用卷积把图像切块后交给 Attention
```

## 4. 详细推理（Deep Dive）

### 4.1 参数共享为什么有效？

```
全连接处理 224×224 图像：
  参数量 = 224 × 224 × 64 = 3,211,264（仅一层！）

3×3 卷积处理同样图像：
  参数量 = 3 × 3 × 3 × 64 + 64 = 1,792

减少了 1,796 倍！

原因：自然图像的统计特性具有平移不变性
  - 检测"边缘"的特征在图像任何位置都一样
  - 同一个卷积核可以在所有位置复用
```

### 4.2 感受野计算

```python
# ============================================================
# 感受野：每个输出像素"看到"的输入范围
# Receptive field: Input range each output pixel "sees"
# ============================================================

# 一层 3×3 卷积: 感受野 = 3×3
# 两层 3×3 卷积: 感受野 = 5×5
# 三层 3×3 卷积: 感受野 = 7×7

# 这就是 VGG 的设计思想：
# Two 3×3 convolutions = One 5×5 convolution
# 但参数量: 2×(3²×C²) = 18C² < 25C² = 5²×C²
# 而且多了一次非线性激活！
```

## 5. 例题（Worked Examples）

### 例题 1：手算卷积输出尺寸

```python
# ============================================================
# 计算各种配置的输出尺寸
# Calculate output size for various configurations
# ============================================================

def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """计算卷积输出尺寸 / Calculate convolution output size."""
    return (input_size - kernel_size + 2 * padding) // stride + 1

# 常见场景 / Common scenarios
configs = [
    (224, 3, 1, 1),   # 3×3, stride=1, pad=1 → 保持尺寸
    (224, 3, 2, 1),   # 3×3, stride=2, pad=1 → 减半
    (224, 7, 2, 3),   # 7×7, stride=2, pad=3 → ResNet 第一层
    (224, 16, 16, 0), # 16×16, stride=16 → ViT patch embedding
]

for inp, k, s, p in configs:
    out = conv_output_size(inp, k, s, p)
    print(f"Input={inp}, Kernel={k}, Stride={s}, Pad={p} → Output={out}")
```

### 例题 2：构建简单 CNN

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    """简单 CNN 分类器 / Simple CNN classifier."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # [B,1,28,28] → [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                   # [B,32,14,14]
            nn.Conv2d(32, 64, 3, padding=1),   # [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                   # [B,64,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # [B, 64*7*7]
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 计算以下卷积的输出尺寸和参数量：输入 [B, 3, 32, 32]，Conv2d(3, 16, kernel_size=5, stride=1, padding=2)。

> **答案：** 输出 [B, 16, 32, 32]，参数量 = 16×3×5×5 + 16 = 1,216

**练习 2：** 解释为什么 ViT 使用 `Conv2d(3, 768, kernel_size=16, stride=16)` 作为 Patch Embedding？

### 进阶题

**练习 3：** 实现一个深度可分离卷积（Depthwise Separable Convolution），对比其与标准卷积的参数量差异。

> **参考答案：**
> ```python
> class DepthwiseSeparableConv(nn.Module):
>     def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
>         super().__init__()
>         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
>         self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
>     
>     def forward(self, x):
>         return self.pointwise(self.depthwise(x))
> # 参数量: in_ch×K² + in_ch×out_ch（远小于 in_ch×out_ch×K²）
> ```
