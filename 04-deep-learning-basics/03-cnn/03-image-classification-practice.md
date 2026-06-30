# 图像分类实战 / Image Classification Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 图像分类是深度学习的 **"Hello World"**——通过一个完整的实战项目，串联数据加载、预处理、模型构建、训练循环、评估指标等所有组件。掌握了图像分类的完整流程后，迁移到 NLP 和大模型只需要替换数据和模型结构，核心训练范式是一样的。
>
> 对于 Java 工程师来说，这条训练流水线 ≈ 一条 ETL/责任链：`DataLoader` 像分批拉数据的 `Iterator`，预处理→前向→反向→评估各是流水线上一道工序，按 batch 反复流过。
>
> **在整个体系中的位置：** 这是深度学习实战的第一个端到端项目，建立起模型训练的完整心智模型。

## 2. 知识点（Key Concepts）

| 环节 | 工具/方法 | 说明 |
|------|-----------|------|
| 数据集 | MNIST, CIFAR-10 | 经典基准数据集 |
| 数据增强 | transforms.Compose | 提升泛化能力 |
| 模型选择 | 自定义 CNN / 预训练模型 | 从简单到复杂 |
| 损失函数 | CrossEntropyLoss | 分类任务标配 |
| 评估指标 | Accuracy, Confusion Matrix | 衡量模型表现 |
| 超参数调优 | 学习率、batch_size | 影响训练效果 |

## 3. 内容（Content）

### 3.1 数据加载与预处理

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================================================
# CIFAR-10 数据集准备
# CIFAR-10 dataset preparation
# ============================================================

# 训练集数据增强 / Training data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # 随机水平翻转
    transforms.RandomCrop(32, padding=4),       # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 均值
                         (0.2470, 0.2435, 0.2616)),  # CIFAR-10 标准差
])

# 测试集不做增强 / No augmentation for test set
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

### 3.2 构建 CNN 模型

```python
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    """CIFAR-10 分类 CNN / CIFAR-10 Classification CNN.
    
    目标: 在 CIFAR-10 上达到 85%+ 准确率。
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B,3,32,32] → [B,64,16,16]
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: [B,64,16,16] → [B,128,8,8]
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: [B,128,8,8] → [B,256,4,4]
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
```

### 3.3 完整训练流程

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 训练 / Training
for epoch in range(50):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    train_acc = correct / total
    
    # 评估 / Evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total
    
    print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

### 3.4 评估与分析

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 收集预测结果 / Collect predictions
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# 分类报告 / Classification report
print(classification_report(all_labels, all_preds, target_names=classes))

# 混淆矩阵 / Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("混淆矩阵 / Confusion Matrix:")
print(cm)
```

## 4. 详细推理（Deep Dive）

### 4.1 数据增强为什么有效？

```
数据增强 = 人工扩大训练集
  原始: 50,000 张图
  增强后: 每张有多种变体 → 等效于更大的数据集

效果: 
  无增强: ~80% 准确率
  有增强: ~88% 准确率

常用技术:
  翻转/旋转 → 学习旋转不变性
  裁剪/缩放 → 学习尺度不变性
  颜色抖动 → 学习光照不变性
  Mixup/CutMix → 大模型时代的高级增强
```

## 5. 例题（Worked Examples）

### 例题：使用预训练 ResNet 做迁移学习

```python
import torchvision.models as models

# 加载预训练 ResNet-18 / Load pretrained ResNet-18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(512, 10)  # 10 classes for CIFAR-10
resnet = resnet.to(device)

# 通常准确率可以轻松超过 90%！
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 在 MNIST 上训练一个简单 CNN （2 层卷积），达到 99%+ 准确率。

*参考答案*：

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MnistCNN(nn.Module):
    """2 层卷积即可在 MNIST 上达到 99%+ / Two conv layers reach 99%+."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [B,32,14,14]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # [B,64,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 10))

    def forward(self, x):  # x: [B, 1, 28, 28]
        return self.classifier(self.features(x))

tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])  # MNIST 均值方差
train = torchvision.datasets.MNIST('./data', True, download=True, transform=tf)
test = torchvision.datasets.MNIST('./data', False, download=True, transform=tf)
train_loader = DataLoader(train, 128, shuffle=True)
test_loader = DataLoader(test, 256)

model = MnistCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
for epoch in range(5):                       # 约 5 epoch 即可破 99%
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()

# 评估 / Evaluate
model.eval(); correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += y.size(0)
print(f"Test Acc: {correct/total:.4f}")      # ≈ 0.99x
```

要点：MNIST 简单，2 层卷积 + 1 个全连接、约 5 个 epoch 的 Adam 即可稳定超过 99%。归一化用 MNIST 的统计量 (0.1307, 0.3081)，加一点 Dropout 防过拟合。

**练习 2：** 对比有/无数据增强的训练效果差异。

*参考答案*：

实验设计：固定模型、优化器、epoch 数，只切换 train_transform，对比两条曲线。

```python
import torchvision.transforms as T

NORM = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
# A. 无增强 / No augmentation
plain = T.Compose([T.ToTensor(), NORM])
# B. 有增强 / With augmentation
aug = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(), NORM,
])
# 分别用 plain / aug 作为训练集 transform，测试集始终用 plain，
# 训练同一个 CIFAR10Net，记录每个 epoch 的 train_acc 和 test_acc。
```

预期观察（CIFAR-10 上的典型现象）：

| 配置 | 训练准确率 | 测试准确率 | 现象 |
|------|-----------|-----------|------|
| 无增强 | 接近 100% | ~80% | 训练-测试差距大，**明显过拟合** |
| 有增强 | ~95% | ~88% | 差距收窄，**泛化更好** |

解释：增强相当于在每个 epoch 给模型看"略有不同"的样本，等效于扩大数据集、注入对平移/翻转的不变性先验，从而压低训练准确率但显著抬高测试准确率。注意增强**只加在训练集**，测试集保持原样。

### 进阶题

**练习 3：** 在 CIFAR-10 上使用预训练 ResNet-18 做迁移学习，对比以下策略的效果：
1. 只训练分类头（冻结 backbone）
2. 全部微调（fine-tune）
3. 逐层解冻（Gradual Unfreezing）

*参考答案*：

三种策略的实现核心是"控制哪些参数 `requires_grad=True`"：

```python
import torchvision.models as models
import torch.nn as nn

def build():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

# 策略 1：只训练分类头 / Frozen backbone
m1 = build()
for p in m1.parameters(): p.requires_grad = False
for p in m1.fc.parameters(): p.requires_grad = True   # 仅 fc 可训练

# 策略 2：全部微调 / Full fine-tune（建议 backbone 用更小学习率）
m2 = build()                                           # 所有参数默认可训练
opt2 = torch.optim.Adam([
    {"params": m2.fc.parameters(), "lr": 1e-3},
    {"params": (p for n, p in m2.named_parameters() if not n.startswith("fc")), "lr": 1e-4},
])

# 策略 3：逐层解冻 / Gradual unfreezing
# 先只训 fc，再依次解冻 layer4 -> layer3 -> ...，每解冻一组降低学习率
```

效果对比（CIFAR-10 典型结论）：

| 策略 | 训练成本 | 测试准确率 | 适用场景 |
|------|---------|-----------|----------|
| 只训分类头 | 最低 | 较低 | 数据极少 / 域接近 ImageNet |
| 全部微调 | 最高 | **通常最高** | 数据充足、追求精度 |
| 逐层解冻 | 中等 | 接近全微调且更稳 | 数据中等、怕灾难性遗忘 |

要点：全微调上限最高，但学习率要小（尤其 backbone），否则会破坏预训练特征；逐层解冻是两者的折中，先稳住高层再逐步释放低层，能缓解小数据上的过拟合与遗忘。

**练习 4：** 实现 Mixup 数据增强并观察其对测试准确率的影响。

*参考答案*：

Mixup 核心：对两个样本及其 one-hot 标签做凸组合，`λ ~ Beta(α, α)`。损失 = 对两套标签的加权交叉熵。

```python
import numpy as np
import torch
import torch.nn.functional as F

def mixup_batch(x, y, alpha=1.0):
    """对一个 batch 做 mixup / Mix one batch.

    Args:
        x: 图像 / Images. Shape: [B, C, H, W]
        y: 整数标签 / Int labels. Shape: [B]
    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha)        # λ ∈ (0,1)
    idx = torch.randperm(x.size(0), device=x.device)  # 随机配对 / random pairing
    mixed_x = lam * x + (1 - lam) * x[idx]    # Shape: [B, C, H, W]
    return mixed_x, y, y[idx], lam

def mixup_loss(logits, y_a, y_b, lam):
    # 两套标签的加权 CE / weighted CE over both labels
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

# 训练循环里 / In the training loop:
# mixed_x, y_a, y_b, lam = mixup_batch(images, labels, alpha=1.0)
# loss = mixup_loss(model(mixed_x), y_a, y_b, lam)
```

对测试准确率的影响：Mixup 是一种**强正则化**。它通过在样本/标签空间做线性插值，鼓励模型在样本之间表现得更"线性"、不对单点过度自信，因而：

- 降低过拟合，测试准确率通常**提升约 1–2 个百分点**（CIFAR-10 量级），并改善模型校准（置信度更可靠）、增强对噪声标签和对抗扰动的鲁棒性。
- 代价：训练准确率看起来更低、收敛更慢，因为模型在拟合"混合"的软目标；通常需要训练更多 epoch 才能体现增益。
