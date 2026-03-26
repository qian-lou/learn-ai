# nn.Module 模型构建 / Building Models with nn.Module

## 1. 背景（Background）
> `nn.Module` 是 PyTorch 模型的基类，所有自定义模型都继承自它。类似 Java 中继承一个抽象基类。

## 2-3. 知识点与内容
```python
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Shape: [B, L] -> [B, L, D]
        self.fc = nn.Linear(embed_dim, n_classes)             # Shape: [B, D] -> [B, C]
    
    def forward(self, x):
        embedded = self.embedding(x)        # Shape: [B, L, D]
        pooled = embedded.mean(dim=1)       # Shape: [B, D] 平均池化
        return self.fc(pooled)              # Shape: [B, C]

model = TextClassifier(10000, 256, 5)
print(sum(p.numel() for p in model.parameters()))  # 总参数量
```

## 4-6. 推理/例题/习题
**练习：** 构建一个包含 BatchNorm、Dropout 和残差连接的模型。
