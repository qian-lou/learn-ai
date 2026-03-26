# GPU 加速与混合精度 / GPU Acceleration and Mixed Precision

## 1. 背景（Background）
> 大模型训练必须使用 GPU。混合精度（FP16/BF16）节省显存并加速计算。

## 2-3. 知识点与内容
```python
import torch
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
scaler = GradScaler()

for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    with autocast():  # 自动使用 FP16 计算
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 4-6. 推理/例题/习题
**练习：** 对比 FP32 和混合精度训练的速度和显存占用差异。
