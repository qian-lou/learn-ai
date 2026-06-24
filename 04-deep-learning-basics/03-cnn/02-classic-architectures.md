# 经典架构（LeNet/ResNet/VGG）/ Classic CNN Architectures

## 1. 背景（Background）

> **为什么要学这个？**
>
> CNN 架构的演进历程（LeNet → AlexNet → VGG → GoogLeNet → ResNet）浓缩了深度学习最重要的设计思想。其中最关键的创新——**残差连接（Residual Connection）**——直接被 Transformer 采用，成为所有大模型的标配。
>
> 理解 ResNet 的残差连接为什么有效，就理解了 Transformer 为什么可以堆叠 96 层（GPT-3）仍然能训练。
>
> **在整个体系中的位置：** CNN 架构演进是深度学习工程智慧的结晶。VGG 的"更深更好"、GoogLeNet 的"多尺度"、ResNet 的"残差连接"——这些思想在 Transformer 时代依然适用。

## 2. 知识点（Key Concepts）

| 架构 | 年份 | 层数 | 核心创新 | 与 Transformer 的联系 |
|------|------|------|----------|----------------------|
| LeNet-5 | 1998 | 5 | 第一个实用 CNN | 基础卷积结构 |
| AlexNet | 2012 | 8 | ReLU + Dropout + GPU | 深度学习复兴的起点 |
| VGG | 2014 | 16/19 | 3×3 小卷积堆叠 | 统一设计原则 |
| GoogLeNet | 2014 | 22 | Inception 多尺度 | 多头的灵感来源 |
| ResNet | 2015 | 50/101/152 | **残差连接** | **Transformer 核心组件** ✅ |

## 3. 内容（Content）

### 3.1 架构演进

```
CNN 架构演进树：

LeNet (1998)          → AlexNet (2012)       → VGG (2014)
  5 层                   8 层                   16-19 层
  手写数字               ImageNet 冠军          更深更好
  
                                              → GoogLeNet (2014)
                                                 22 层
                                                 Inception 模块
                                                 
                                              → ResNet (2015) ★
                                                 152 层！
                                                 残差连接突破深度限制
                                                 
ResNet 的核心洞察：
  深度模型退化不是过拟合导致的，
  而是"优化困难"——梯度传播不畅导致深层网络反而更差
  
  解决方案：让网络学习残差 F(x) = H(x) - x
  即 H(x) = F(x) + x（跳跃连接）
```

### 3.2 ResNet 残差块实现

```python
import torch
import torch.nn as nn

# ============================================================
# ResNet 核心：残差块 / ResNet core: Residual Block
# ============================================================
class ResidualBlock(nn.Module):
    """ResNet 基本残差块 / Basic Residual Block.
    
    output = ReLU(BatchNorm(Conv(ReLU(BatchNorm(Conv(x))))) + x)
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 核心：输出 = 变换(输入) + 输入
        # Core: output = transform(input) + input
        identity = x
        out = self.block(x)
        out = out + identity  # 残差连接！/ Residual connection!
        return self.relu(out)


# ============================================================
# 完整的 ResNet-18 简化版
# Simplified ResNet-18
# ============================================================
class SimpleResNet(nn.Module):
    """简化版 ResNet / Simplified ResNet."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 初始层 / Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # [B,3,224,224]->[B,64,112,112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # [B,64,56,56]
        )
        # 残差块堆叠 / Stacked residual blocks
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        # 全局平均池化 + 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)        # [B, 64, 56, 56]
        x = self.layer1(x)       # [B, 64, 56, 56]
        x = self.avgpool(x)      # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        return self.fc(x)        # [B, num_classes]
```

### 3.3 使用预训练模型（迁移学习）

```python
import torchvision.models as models
import torch.nn as nn

# ============================================================
# 加载 ImageNet 预训练的 ResNet-18
# Load ImageNet pretrained ResNet-18
# ============================================================
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 替换最后的全连接层（迁移到自己的任务）
# Replace last FC layer (transfer to your task)
num_classes = 5  # 你的分类数
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 冻结特征提取层（只训练分类头）
# Freeze feature extractor (only train classifier head)
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc.parameters():
    param.requires_grad = True

print(f"可训练参数: {sum(p.numel() for p in resnet.parameters() if p.requires_grad):,}")
```

### 3.4 ResNet 与 Transformer 的关系

```
ResNet Block              vs           Transformer Block
─────────────                          ─────────────────
     x                                      x
     │                                       │
  Conv3×3                                LayerNorm
     │                                       │
  BatchNorm                              Attention
     │                                       │
    ReLU                                  + (残差)
     │                                       │
  Conv3×3                                LayerNorm
     │                                       │
  BatchNorm                                 FFN
     │                                       │
  + (残差) ← identity skip                + (残差) ← identity skip
     │                                       │
    ReLU                                   output
     │
   output

核心相同点：残差连接 (output = F(x) + x)
  → 让梯度可以"直通"到最前面的层
  → 这就是为什么 Transformer 可以堆 96 层（GPT-3）
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么残差连接能训练更深的网络？

```
没有残差连接（深度退化问题）：
  梯度: ∂L/∂x₁ = ∂L/∂xₙ · ∂xₙ/∂xₙ₋₁ · ... · ∂x₂/∂x₁
  如果每个 ∂xᵢ/∂xᵢ₋₁ < 1 → 梯度指数衰减 → 前面的层学不到东西

有残差连接：
  xₗ₊₁ = F(xₗ) + xₗ
  ∂xₗ₊₁/∂xₗ = ∂F/∂xₗ + 1  ← 这个 "+1" 是关键！
  
  梯度至少为 1，不会消失！
  即使 F 的梯度很小，信号仍然可以通过 identity 直接传到前面的层

类比：
  想象你在一个深井（100层网络）底部喊话，
  没有残差连接 → 声音层层衰减，最上面听不到
  有残差连接 → 相当于每层都有扩音器，声音直达井口
```

## 5. 例题（Worked Examples）

### 例题：实现带下采样的残差块

```python
class BottleneckBlock(nn.Module):
    """ResNet Bottleneck 块（用于 ResNet-50/101/152）."""
    
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),  # 1×1 降维
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False),  # 3×3
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),  # 1×1 升维
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 解释 ResNet 论文中的观察：56 层的 plain network 比 20 层的表现更差，这不是过拟合（训练误差也更高）。这说明了什么？

*参考答案*：

这说明问题是**优化困难（degradation problem）**，而非过拟合或表达能力不足。

- 过拟合的特征是"训练误差低、测试误差高"；而这里 56 层连**训练误差**都更高，所以不是过拟合。
- 表达能力也不是瓶颈：56 层网络理论上可以表示 20 层网络的解（多出的 36 层只要学成恒等映射 identity，就退化为 20 层），所以 56 层的最优解不可能比 20 层差。
- 唯一的解释是：**SGD 难以让普通堆叠层学到恒等映射**。深层网络梯度传播不畅，优化器找不到那个"至少不更差"的解。
- ResNet 的对策：把映射改写为 `H(x) = F(x) + x`，让"学恒等映射"变成"把残差 F(x) 学成 0"——后者对优化器容易得多（权重趋近 0 即可），从而解决退化问题。

**练习 2：** 用 `torchvision.models.resnet18(pretrained=True)` 对 CIFAR-10 做迁移学习，只训练最后的全连接层。

*参考答案*：

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet 预训练要求 224×224 输入与 ImageNet 归一化
# Pretrained model expects 224x224 + ImageNet normalization
tf = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tf)
loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# 1. 冻结全部 backbone / Freeze backbone
for p in model.parameters():
    p.requires_grad = False
# 2. 替换分类头（新层默认 requires_grad=True）/ Replace head
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 3. 只把 fc 参数交给优化器 / Only optimize the head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

要点：冻结 backbone 后只训练 `model.fc`，仅几千个参数即可，速度快、不易过拟合；务必把输入 resize 到 224 并用 ImageNet 的均值/方差归一化，否则与预训练分布不匹配会严重掉点。

### 进阶题

**练习 3：** 实现 Pre-Norm 版本的残差连接（Transformer 常用）：`output = x + F(LayerNorm(x))`，对比 Post-Norm 版本的训练稳定性。

*参考答案*：

```python
import torch.nn as nn

class PreNormBlock(nn.Module):
    """Pre-Norm: output = x + F(LayerNorm(x))."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):  # x: [B, T, dim]
        return x + self.ff(self.norm(x))   # 归一化在残差分支内部 / norm inside branch

class PostNormBlock(nn.Module):
    """Post-Norm: output = LayerNorm(x + F(x)) —— 原始 Transformer."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.norm(x + self.ff(x))   # 归一化在残差相加之后 / norm after add
```

稳定性对比（结论）：
- **Pre-Norm 更稳**。残差通路 `x` 完全不经过 LayerNorm，是一条干净的恒等高速公路，深层堆叠时梯度不衰减，常常**无需 warmup** 就能训练几十上百层。
- **Post-Norm** 的恒等通路被 LayerNorm 包裹，深层时输出方差随层数累积放大，需要**学习率 warmup + 仔细初始化**才能稳定，但收敛后峰值效果有时略高。
- 现代大模型（GPT 系列等）几乎都采用 Pre-Norm，正是为了堆叠深层时的训练稳定性。

**练习 4：** 计算 ResNet-50 的总参数量和总 FLOPs（对于 224×224 输入）。

*参考答案*：

ResNet-50 的公认数值（来自原论文及标准统计）：
- **参数量 ≈ 25.6 M（约 2560 万）**
- **FLOPs ≈ 4.1 GFLOPs**（224×224 输入，统计一次前向的乘加，常记为 ~3.8–4.1 GFLOPs，口径不同略有差异）

不必手算，直接用工具核对即可：

```python
import torchvision.models as models
from thop import profile          # pip install thop
import torch

model = models.resnet50()
# 参数量 / Parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.1f} M")          # ≈ 25.6 M

# FLOPs（thop 统计 MACs，乘加各算一次）/ FLOPs
x = torch.randn(1, 3, 224, 224)   # Shape: [B, C, H, W]
macs, params = profile(model, inputs=(x,), verbose=False)
print(f"MACs: {macs/1e9:.2f} G")  # ≈ 4.1 G (FLOPs≈2×MACs，业界常直接报 MACs)
```

说明：业界口径常把一次 MAC（乘加）记作 1 FLOP，因此 ResNet-50 常被报为 "~4 GFLOPs"。Bottleneck 结构用 1×1 卷积降/升维，正是为了在加深到 50 层时把 FLOPs 控制在 ~4G 这个量级。
