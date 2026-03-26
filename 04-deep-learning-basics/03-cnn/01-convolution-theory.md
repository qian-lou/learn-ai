# 卷积原理 / Convolution Theory

## 1. 背景（Background）
> CNN 虽已被 Transformer 超越，但理解卷积有助于理解"局部特征提取"思想。Vision Transformer 的 Patch Embedding 借鉴了卷积。

## 2-3. 知识点与内容
```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# Input: [B, 3, 224, 224] -> Output: [B, 64, 224, 224]
# 输出尺寸: output_size = (input - kernel + 2*padding) / stride + 1
```

## 4-6. 推理/例题/习题
**练习：** 手动计算不同卷积参数下的输出尺寸和参数量。
