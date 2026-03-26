# 经典架构（LeNet/ResNet/VGG）/ Classic CNN Architectures

## 1. 背景（Background）
> CNN 架构演进：LeNet → AlexNet → VGG → ResNet。残差连接（ResNet）是现代深度学习核心技巧，Transformer 也大量使用。

## 2-3. 知识点与内容
```python
import torchvision.models as models
resnet = models.resnet18(pretrained=True)

# ResNet 核心：残差连接 output = F(x) + x
# 这和 Transformer 的结构完全一致！
# Transformer: output = LayerNorm(Attention(x) + x)
```

## 4-6. 推理/例题/习题
**练习：** 实现一个 ResBlock，理解梯度直通的原理。
