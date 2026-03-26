# 训练循环与 DataLoader / Training Loop and DataLoader

## 1. 背景（Background）
> 训练循环是深度学习核心范式：前向传播 → 计算损失 → 反向传播 → 更新参数。

## 2-3. 知识点与内容
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()           # 1. 清零梯度
        output = model(batch_x)         # 2. 前向传播
        loss = criterion(output, batch_y)  # 3. 计算损失
        loss.backward()                 # 4. 反向传播
        optimizer.step()                # 5. 更新参数
```

## 4-6. 推理/例题/习题
**练习：** 实现完整训练循环，包含验证集评估和 Early Stopping。
