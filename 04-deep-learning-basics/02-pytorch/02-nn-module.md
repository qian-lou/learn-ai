# nn.Module 模型构建 / Building Models with nn.Module

## 1. 背景（Background）

> **为什么要学这个？**
>
> `nn.Module` 是 PyTorch 中构建所有神经网络模型的**基类**。对于 Java 工程师来说，它就像是一个**抽象基类（AbstractClass）**——你继承它，实现 `forward()` 方法（类似 Java 中重写 `execute()` 或 `process()` 方法），然后 PyTorch 自动处理参数管理、设备迁移、序列化等工作。
>
> 理解 `nn.Module` 的工作机制，是构建从简单 MLP 到复杂 Transformer 模型的基础。所有大模型（GPT、BERT、LLaMA）本质上都是由多个 `nn.Module` 子类组成的树形结构。
>
> **在整个体系中的位置：** `nn.Module` 是 PyTorch 模型层的基石。理解它之后，才能构建复杂模型、使用预训练模型、进行模型微调。

## 2. 知识点（Key Concepts）

| PyTorch 概念 | Java 类比 | 说明 |
|-------------|-----------|------|
| `nn.Module` | 抽象基类 `AbstractModel` | 所有模型的基类 |
| `forward()` | 重写的 `process()` 方法 | 前向传播逻辑 |
| `parameters()` | `getFields()` 获取成员 | 返回所有可训练参数 |
| `nn.Linear` | 一个方法调用 `y = W*x + b` | 全连接层 |
| `nn.Sequential` | Builder 模式链式调用 | 按顺序堆叠层 |
| `model.train()` / `model.eval()` | 设置运行模式 | 控制 Dropout/BN 行为 |
| `state_dict()` | `serialize()` 序列化 | 导出模型参数 |

**核心要点：**
- 所有模型都继承 `nn.Module` 并实现 `forward()`
- **不要**直接调用 `forward()`，用 `model(x)` 调用（会触发钩子）
- 子模块必须注册为**类属性**（`self.layer = ...`），否则 `parameters()` 找不到
- `model.train()` 和 `model.eval()` 会影响 Dropout 和 BatchNorm 的行为

## 3. 内容（Content）

### 3.1 基本模型构建

```python
import torch
import torch.nn as nn

# ============================================================
# 方式 1：继承 nn.Module（推荐，最灵活）
# Method 1: Inherit nn.Module (recommended, most flexible)
# ============================================================
class SimpleClassifier(nn.Module):
    """简单分类器 / Simple classifier.
    
    Args:
        input_dim: 输入维度 / Input dimension.
        hidden_dim: 隐藏层维度 / Hidden dimension.
        num_classes: 类别数 / Number of classes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()  # 必须调用！/ Must call!
        # 所有层都注册为类属性
        # All layers registered as class attributes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward pass.
        
        Args:
            x: 输入特征 / Input features. Shape: [B, input_dim]
        
        Returns:
            分类 logits / Classification logits. Shape: [B, num_classes]
        """
        x = self.fc1(x)      # Shape: [B, input_dim] -> [B, hidden_dim]
        x = self.relu(x)     # Shape: [B, hidden_dim]
        x = self.dropout(x)  # Shape: [B, hidden_dim]（训练时随机置零）
        x = self.fc2(x)      # Shape: [B, hidden_dim] -> [B, num_classes]
        return x

# 使用方式 / Usage
model = SimpleClassifier(784, 256, 10)
x = torch.randn(32, 784)    # 一个 batch 的输入
logits = model(x)            # 不要调用 model.forward(x)！
print(f"输出形状: {logits.shape}")  # [32, 10]


# ============================================================
# 方式 2：nn.Sequential（简单堆叠）
# Method 2: nn.Sequential (simple stacking)
# ============================================================
model_seq = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)
logits = model_seq(x)  # 效果同上
```

### 3.2 常用层一览

```python
import torch.nn as nn

# ============================================================
# 线性层 / Linear layers
# ============================================================
linear = nn.Linear(768, 3072)         # y = xW^T + b, Shape: [B, 768] -> [B, 3072]
linear_no_bias = nn.Linear(768, 3072, bias=False)  # 无偏置（大模型常用）

# ============================================================
# 嵌入层（NLP 核心）/ Embedding layer (NLP core)
# ============================================================
# 将离散 token ID 映射为连续向量
# Map discrete token IDs to continuous vectors
embedding = nn.Embedding(
    num_embeddings=50000,  # 词表大小 / Vocabulary size
    embedding_dim=768,     # 嵌入维度 / Embedding dimension
)
token_ids = torch.tensor([[0, 512, 1024]])  # Shape: [1, 3]
vectors = embedding(token_ids)               # Shape: [1, 3, 768]

# ============================================================
# 归一化层 / Normalization layers
# ============================================================
# LayerNorm: Transformer 的标配
# LayerNorm: Standard for Transformer
layer_norm = nn.LayerNorm(768)  # 在最后一维归一化

# BatchNorm: CNN 常用
batch_norm = nn.BatchNorm1d(256)

# RMSNorm: LLaMA 使用（更简单，无减均值）
# RMSNorm: Used by LLaMA (simpler, no mean subtraction)
# PyTorch 2.4+ 内置 nn.RMSNorm

# ============================================================
# 正则化层 / Regularization layers
# ============================================================
dropout = nn.Dropout(p=0.1)  # 训练时随机置零 10% 的神经元

# ============================================================
# 激活函数层 / Activation layers
# ============================================================
relu = nn.ReLU()
gelu = nn.GELU()
silu = nn.SiLU()
```

### 3.3 模型参数管理

```python
import torch.nn as nn

model = SimpleClassifier(784, 256, 10)

# ============================================================
# 1. 查看参数 / Inspect parameters
# ============================================================
# 打印所有参数名和形状
# Print all parameter names and shapes
for name, param in model.named_parameters():
    print(f"{name:20s}: shape={param.shape}, requires_grad={param.requires_grad}")

# 输出:
# fc1.weight          : shape=torch.Size([256, 784]), requires_grad=True
# fc1.bias            : shape=torch.Size([256]),      requires_grad=True
# fc2.weight          : shape=torch.Size([10, 256]),  requires_grad=True
# fc2.bias            : shape=torch.Size([10]),       requires_grad=True

# 统计总参数量 / Count total parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total:,}, 可训练: {trainable:,}")

# ============================================================
# 2. 冻结参数（迁移学习/微调常用）
# Freeze parameters (common in transfer learning/fine-tuning)
# ============================================================
# 冻结第一层 / Freeze first layer
for param in model.fc1.parameters():
    param.requires_grad = False

# 只有 fc2 的参数会更新
# Only fc2 parameters will be updated
trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"冻结后可训练参数: {trainable_after:,}")

# ============================================================
# 3. 模型设备管理 / Model device management
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # 将所有参数移到 GPU

# 确保输入也在同一设备上！
# Ensure inputs are on the same device!
x = torch.randn(32, 784).to(device)
output = model(x)
```

### 3.4 高级模型构建模式

```python
import torch
import torch.nn as nn

# ============================================================
# 1. 残差连接（ResNet/Transformer 核心）
# Residual connection (core of ResNet/Transformer)
# ============================================================
class ResidualBlock(nn.Module):
    """残差块 / Residual block.
    
    output = LayerNorm(x + SubLayer(x))
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接: 输出 = 输入 + 变换（输入）
        # Residual: output = input + transform(input)
        return x + self.dropout(self.ffn(self.norm(x)))


# ============================================================
# 2. nn.ModuleList（动态层列表）
# nn.ModuleList (dynamic list of layers)
# ============================================================
class TransformerEncoder(nn.Module):
    """简化版 Transformer 编码器 / Simplified Transformer encoder."""
    
    def __init__(self, d_model: int, d_ff: int, n_layers: int):
        super().__init__()
        # 必须用 ModuleList，不能用普通 list！
        # Must use ModuleList, NOT a regular list!
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)    # Shape: [B, S, D] -> [B, S, D]
        return self.norm(x)

encoder = TransformerEncoder(d_model=768, d_ff=3072, n_layers=12)
print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")


# ============================================================
# 3. nn.ModuleDict（动态层字典）
# nn.ModuleDict (dynamic dict of layers)
# ============================================================
class MultiTaskModel(nn.Module):
    """多任务模型 / Multi-task model."""
    
    def __init__(self, d_model: int, tasks: dict[str, int]):
        super().__init__()
        self.shared = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, n_classes)
            for name, n_classes in tasks.items()
        })
    
    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        shared_out = self.shared(x)
        return self.heads[task](shared_out)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么不能直接用 Python list 存储子模块？

```python
# ❌ 错误做法 / Wrong approach
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]  # 普通 list！
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

bad = BadModel()
print(list(bad.parameters()))  # 空！参数没有注册！
# model.to('cuda') 也不会移动这些层的参数

# ✅ 正确做法 / Correct approach
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
```

### 4.2 model.train() vs model.eval() 的影响

```
model.train()                    model.eval()
─────────────                    ────────────
Dropout: 激活（随机置零）          Dropout: 关闭（全部通过）
BatchNorm: 使用 batch 统计量      BatchNorm: 使用全局统计量
梯度: 正常计算                    梯度: 仍然计算（需配合 no_grad）

常见误区：
  model.eval() 不会停止梯度计算！
  推理时应该同时使用：
    model.eval()
    with torch.no_grad():
        output = model(x)
```

### 4.3 模型参数初始化

```python
import torch.nn as nn
import torch.nn.init as init

# ============================================================
# 自定义初始化（影响训练收敛速度）
# Custom initialization (affects convergence speed)
# ============================================================
class InitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 768)
        
        # Xavier 初始化（适合 Sigmoid/Tanh）
        init.xavier_uniform_(self.fc.weight)
        
        # Kaiming 初始化（适合 ReLU）
        # init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        
        # 偏置初始化为零
        init.zeros_(self.fc.bias)

# 大模型通常使用缩放后的正态分布初始化
# LLMs typically use scaled normal initialization
# std = 0.02 (GPT-2) 或 1/sqrt(d_model)（Transformer 原始版）
```

## 5. 例题（Worked Examples）

### 例题 1：构建文本分类模型

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    """简单文本分类器（Embedding + Mean Pooling + FC）."""
    
    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # [B, L] -> [B, L, D]
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, n_classes),
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: token ID 序列 / Token ID sequence. Shape: [B, L]
        
        Returns:
            分类 logits / Classification logits. Shape: [B, n_classes]
        """
        embedded = self.embedding(input_ids)   # Shape: [B, L, D]
        pooled = embedded.mean(dim=1)          # Shape: [B, D] — 均值池化
        normed = self.norm(pooled)             # Shape: [B, D]
        return self.classifier(normed)         # Shape: [B, n_classes]

# 测试 / Test
model = TextClassifier(vocab_size=30000, embed_dim=256, n_classes=5)
dummy_input = torch.randint(0, 30000, (4, 128))  # 4 个样本，长度 128
output = model(dummy_input)
print(f"输出形状: {output.shape}")        # [4, 5]
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 例题 2：打印模型结构

```python
# PyTorch 内置的模型打印
print(model)
# 输出结构化的模型树

# 更详细的参数统计
from torchinfo import summary  # pip install torchinfo
summary(model, input_data=dummy_input)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 构建一个 MLP 模型，结构为 `784 → 512 → ReLU → Dropout(0.3) → 256 → ReLU → 10`，并统计参数量。

**练习 2：** 解释为什么调用 `model(x)` 而不是 `model.forward(x)`。提示：查看 `nn.Module.__call__` 的源码。

> **答案：** `__call__` 在调用 `forward()` 前后会执行注册的 hooks（前向钩子和后向钩子），直接调用 `forward()` 会跳过这些钩子。

### 进阶题

**练习 3：** 构建一个包含**残差连接**和 **LayerNorm** 的 Transformer Block：

```python
class TransformerBlock(nn.Module):
    """单个 Transformer Block.
    
    结构: x → LayerNorm → Self-Attention → Residual →
              LayerNorm → FFN → Residual
    """
    # TODO: 实现
    pass
```

**练习 4：** 实现一个自定义的 `nn.Module`，它接收一个参数 `scale`（标量），在 `forward` 中将输入乘以 `scale`。确保 `scale` 被注册为可训练参数（使用 `nn.Parameter`）。

> **参考答案：**
> ```python
> class ScaleLayer(nn.Module):
>     def __init__(self, dim: int, init_scale: float = 1.0):
>         super().__init__()
>         self.scale = nn.Parameter(torch.full((dim,), init_scale))
>
>     def forward(self, x: torch.Tensor) -> torch.Tensor:
>         return x * self.scale  # 逐元素缩放
> ```
