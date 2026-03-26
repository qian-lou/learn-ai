# 训练循环与 DataLoader / Training Loop and DataLoader

## 1. 背景（Background）

> **为什么要学这个？**
>
> 训练循环是深度学习的**核心范式**——前向传播→计算损失→反向传播→更新参数。对于 Java 工程师来说，训练循环就像是一个**事件循环**或**消息处理循环**：不断地读取数据（事件/消息），处理它们（前向传播），评估结果（损失），然后调整行为（参数更新）。
>
> `DataLoader` 则是 PyTorch 的**数据管道**，负责批量加载、打乱、并行读取数据。它类似于 Java 中的 `BlockingQueue` + 线程池的组合——在后台异步加载数据，前台消费数据进行训练。
>
> **在整个体系中的位置：** 训练循环是将模型、数据、损失函数和优化器串联起来的"胶水代码"。无论是训练 MNIST 分类器还是 GPT-3，训练循环的核心结构都是一样的。

## 2. 知识点（Key Concepts）

| 组件 | 作用 | Java 类比 |
|------|------|-----------|
| Dataset | 数据集的抽象，实现 `__getitem__` 和 `__len__` | `List<T>` 接口 |
| DataLoader | 批量加载数据，支持打乱和并行 | `ExecutorService` + `Iterator` |
| Loss Function | 衡量预测与标签的差距 | 结果校验器 |
| Optimizer | 根据梯度更新参数 | 自动调参策略 |
| Epoch | 遍历完整个数据集一次 | 完成一轮数据处理 |
| Batch | 一次训练的数据子集 | 批量处理的一批 |

**训练循环的 5 步固定流程：**
```
1. optimizer.zero_grad()    # 清零梯度（避免累积）
2. output = model(x)        # 前向传播
3. loss = criterion(y, ŷ)   # 计算损失
4. loss.backward()           # 反向传播（计算梯度）
5. optimizer.step()          # 更新参数
```

## 3. 内容（Content）

### 3.1 Dataset 与 DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# ============================================================
# 方式 1：TensorDataset（最简单，适合内存能放下的数据）
# Method 1: TensorDataset (simplest, for data that fits in memory)
# ============================================================
X = torch.randn(1000, 784)        # 1000 个样本，784 维特征
y = torch.randint(0, 10, (1000,)) # 10 分类标签

dataset = TensorDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size=32,     # 每批 32 个样本
    shuffle=True,       # 每 epoch 随机打乱顺序
    num_workers=4,      # 4 个子进程并行加载数据
    pin_memory=True,    # 锁页内存，加速 CPU→GPU 传输
    drop_last=True,     # 丢弃最后不足 batch_size 的数据
)

# ============================================================
# 方式 2：自定义 Dataset（灵活处理任何数据格式）
# Method 2: Custom Dataset (flexible for any data format)
# ============================================================
class TextDataset(Dataset):
    """自定义文本数据集 / Custom text dataset.
    
    Args:
        texts: 文本列表 / List of texts.
        labels: 标签列表 / List of labels.
        max_length: 最大序列长度 / Maximum sequence length.
    """
    
    def __init__(self, texts: list[str], labels: list[int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self) -> int:
        """返回数据集大小 / Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本 / Get a single sample.
        
        Args:
            idx: 样本索引 / Sample index.
        
        Returns:
            (token_ids, label) 元组 / Tuple of (token_ids, label).
        """
        text = self.texts[idx]
        label = self.labels[idx]
        # 这里简化处理，实际应使用 tokenizer
        # Simplified here, should use tokenizer in practice
        token_ids = torch.zeros(self.max_length, dtype=torch.long)
        return token_ids, torch.tensor(label, dtype=torch.long)


# ============================================================
# 划分训练集和验证集 / Split train and validation sets
# ============================================================
full_dataset = TensorDataset(X, y)
train_size = int(0.8 * len(full_dataset))  # 80% 训练
val_size = len(full_dataset) - train_size   # 20% 验证

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

### 3.2 损失函数

```python
import torch.nn as nn

# ============================================================
# 常用损失函数 / Common loss functions
# ============================================================

# 1. 交叉熵损失（分类任务标配）
# Cross-Entropy Loss (standard for classification)
criterion = nn.CrossEntropyLoss()
# 输入: logits [B, C]（未经过 softmax！）
# 目标: 类别索引 [B]
logits = torch.randn(4, 10)          # 4 个样本，10 个类
labels = torch.tensor([1, 5, 3, 7])  # 正确类别
loss = criterion(logits, labels)

# 2. 二元交叉熵（二分类 / 多标签）
# Binary Cross-Entropy (binary / multi-label)
bce = nn.BCEWithLogitsLoss()
pred = torch.randn(4, 1)     # 预测 logits
target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
loss_bce = bce(pred, target)

# 3. MSE 损失（回归任务）
# MSE Loss (for regression)
mse = nn.MSELoss()
pred_val = torch.randn(4, 1)
true_val = torch.randn(4, 1)
loss_mse = mse(pred_val, true_val)

# 4. 语言模型损失（下一个 token 预测）
# Language model loss (next token prediction)
# 本质就是交叉熵，但在 vocab 维度上计算
vocab_size = 50000
lm_logits = torch.randn(4, 128, vocab_size)  # [B, S, V]
lm_targets = torch.randint(0, vocab_size, (4, 128))  # [B, S]
# 需要 reshape: [B*S, V] vs [B*S]
lm_loss = nn.CrossEntropyLoss()(
    lm_logits.view(-1, vocab_size),
    lm_targets.view(-1)
)
```

### 3.3 完整训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 完整的训练循环模板
# Complete training loop template
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch / Train one epoch.
    
    Args:
        model: 模型 / Model.
        dataloader: 数据加载器 / Data loader.
        criterion: 损失函数 / Loss function.
        optimizer: 优化器 / Optimizer.
        device: 计算设备 / Device.
    
    Returns:
        平均损失 / Average loss.
    """
    model.train()  # 开启训练模式（激活 Dropout/BN）
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 0. 数据移到设备 / Move data to device
        data, target = data.to(device), target.to(device)
        
        # 1. 清零梯度 / Zero gradients
        optimizer.zero_grad()
        
        # 2. 前向传播 / Forward pass
        output = model(data)
        
        # 3. 计算损失 / Compute loss
        loss = criterion(output, target)
        
        # 4. 反向传播 / Backward pass
        loss.backward()
        
        # （可选）梯度裁剪 / (Optional) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. 更新参数 / Update parameters
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """评估模型 / Evaluate model.
    
    Returns:
        (平均损失, 准确率) / (average loss, accuracy).
    """
    model.eval()  # 关闭 Dropout/BN
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


# ============================================================
# 主训练流程 / Main training process
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据 / Prepare data
X_train = torch.randn(800, 784)
y_train = torch.randint(0, 10, (800,))
X_val = torch.randn(200, 784)
y_val = torch.randint(0, 10, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# 模型、损失、优化器
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 训练循环 / Training loop
num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型 / Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
```

### 3.4 Early Stopping

```python
# ============================================================
# Early Stopping：验证损失不再下降时停止训练
# Early Stopping: Stop when validation loss stops improving
# ============================================================
class EarlyStopping:
    """Early Stopping 工具 / Early stopping utility.
    
    Args:
        patience: 容忍的不改善 epoch 数 / Epochs to wait.
        min_delta: 最小改善量 / Minimum improvement.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# 使用 / Usage
early_stopping = EarlyStopping(patience=5)
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么需要 `optimizer.zero_grad()`？

```
PyTorch 的梯度默认是累积的（不会自动清零）：

没有 zero_grad()：
  Step 1: grad = g₁
  Step 2: grad = g₁ + g₂    ← 累积！错误！
  Step 3: grad = g₁ + g₂ + g₃

有 zero_grad()：
  Step 1: grad = 0, 然后 grad = g₁  ← 正确
  Step 2: grad = 0, 然后 grad = g₂  ← 正确

为什么设计成累积？
  因为"梯度累积"是大模型训练的重要技巧：
  当 batch_size=8 但 GPU 只够 batch_size=2 时，
  累积 4 次梯度再 step()，等价于 batch_size=8

梯度累积示例：
  accumulation_steps = 4
  for i, (data, target) in enumerate(loader):
      loss = criterion(model(data), target) / accumulation_steps
      loss.backward()          # 梯度累积
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()     # 每 4 步更新一次
          optimizer.zero_grad()
```

### 4.2 DataLoader 的 num_workers 选择

```
num_workers 原则：
  0:  主进程加载（最慢，但最简单，调试用）
  1:  一个子进程
  2-4: 通常最优（取决于 CPU 核心数和 I/O 速度）
  >4: 可能增加 IPC 开销，反而变慢
  
  经验法则: num_workers = min(CPU 核心数, 4)

  ⚠️ Windows 上 num_workers > 0 需要在 if __name__ == '__main__' 中调用
  ⚠️ 太多 workers 会增加内存占用（每个 worker 复制一份数据集）
```

## 5. 例题（Worked Examples）

### 例题：带梯度累积的训练循环

```python
def train_with_gradient_accumulation(
    model, dataloader, criterion, optimizer, device,
    accumulation_steps: int = 4,
):
    """支持梯度累积的训练 / Training with gradient accumulation.
    
    等效 batch_size = 实际 batch_size × accumulation_steps
    Effective batch_size = actual batch_size × accumulation_steps
    """
    model.train()
    optimizer.zero_grad()
    
    for step, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target) / accumulation_steps  # 缩放损失
        loss.backward()  # 梯度累积
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现完整的训练循环，训练一个 MLP 在 MNIST 数据集上达到 95%+ 准确率。

**练习 2：** 解释 `model.train()` 和 `model.eval()` 的区别，以及为什么评估时还需要 `torch.no_grad()`。

### 进阶题

**练习 3：** 实现一个自定义 `Dataset`，从 CSV 文件读取数据，支持动态数据增强。

**练习 4：** 实现带 `tqdm` 进度条的训练循环，显示实时 loss、学习率和 ETA。

> **参考答案：**
> ```python
> from tqdm import tqdm
> 
> for epoch in range(num_epochs):
>     pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
>     for data, target in pbar:
>         # ... 训练步骤 ...
>         pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
> ```
