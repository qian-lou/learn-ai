# 模型保存与加载 / Model Save and Load

## 1. 背景（Background）

> **为什么要学这个？**
>
> 训练好的模型需要被持久化保存，以便后续推理、微调或分享。对于 Java 工程师来说，模型保存就像是**对象序列化（Serialization）**——将内存中的模型参数持久化到磁盘文件。不同的是，PyTorch 提供了多种保存粒度：只保存参数（`state_dict`）、保存完整训练状态（checkpoint）、或导出为跨平台格式（ONNX/TorchScript）。
>
> 在大模型领域，模型保存还涉及**分片保存**（一个模型分为多个文件）、**安全格式**（safetensors）、**Hugging Face Hub** 上传等话题。
>
> **在整个体系中的位置：** 模型保存与加载是训练和推理之间的桥梁。也是模型共享、迁移学习、增量训练的基础。

## 2. 知识点（Key Concepts）

| 保存方式 | 内容 | 文件大小 | 典型用途 |
|----------|------|----------|----------|
| `state_dict` | 仅模型参数 | 较小 | 推理部署 ✅ |
| Checkpoint | 参数 + 优化器 + epoch + loss | 较大 | 恢复训练 ✅ |
| `torch.save(model)` | 整个模型对象 | 较大 | 不推荐 ❌ |
| ONNX | 跨平台格式 | 中等 | 跨框架部署 |
| safetensors | 安全的张量格式 | 同 state_dict | HuggingFace 标准 ✅ |

**核心要点：**
- **始终保存 `state_dict()`**，而非整个模型对象（避免序列化问题）
- 恢复训练需要保存 optimizer state + scheduler state + epoch
- 大模型通常使用 **safetensors** 格式（更安全、加载更快）
- 加载时注意 `map_location` 参数（GPU↔CPU 设备映射）

## 3. 内容（Content）

### 3.1 保存和加载 state_dict（推荐方式）

```python
import torch
import torch.nn as nn

# ============================================================
# 方式 1: 保存/加载 state_dict（推荐！）
# Method 1: Save/load state_dict (recommended!)
# ============================================================

class MyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MyModel(784, 256, 10)

# 保存参数 / Save parameters
torch.save(model.state_dict(), 'model_weights.pth')

# 加载参数 / Load parameters
# 必须先创建模型实例（结构必须一致！）
loaded_model = MyModel(784, 256, 10)
loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.eval()  # 切换到推理模式

# 查看 state_dict 内容 / Inspect state_dict
for key, value in model.state_dict().items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
# 输出:
# fc1.weight: shape=torch.Size([256, 784]), dtype=torch.float32
# fc1.bias:   shape=torch.Size([256]),      dtype=torch.float32
# fc2.weight: shape=torch.Size([10, 256]),  dtype=torch.float32
# fc2.bias:   shape=torch.Size([10]),       dtype=torch.float32
```

### 3.2 保存完整 Checkpoint（恢复训练用）

```python
import torch

# ============================================================
# 保存 Checkpoint：训练中断后恢复
# Save Checkpoint: Resume after training interruption
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    filepath: str,
):
    """保存训练检查点 / Save training checkpoint.
    
    Args:
        model: 模型 / Model.
        optimizer: 优化器 / Optimizer.
        scheduler: 学习率调度器 / LR scheduler.
        epoch: 当前 epoch / Current epoch.
        loss: 当前损失 / Current loss.
        filepath: 保存路径 / Save path.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'torch_rng_state': torch.get_rng_state(),        # 随机状态
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint 已保存: {filepath} (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = 'cpu',
) -> int:
    """加载训练检查点 / Load training checkpoint.
    
    Args:
        filepath: 文件路径 / File path.
        model: 模型 / Model.
        optimizer: 优化器 / Optimizer (optional for inference).
        scheduler: 调度器 / Scheduler (optional).
        device: 目标设备 / Target device.
    
    Returns:
        起始 epoch / Starting epoch.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Checkpoint 已加载: epoch={epoch}, loss={loss:.4f}")
    
    return epoch + 1  # 返回下一个 epoch


# ============================================================
# 使用示例：带自动恢复的训练循环
# Usage: Training loop with auto-resume
# ============================================================
import os

model = MyModel(784, 256, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

checkpoint_path = 'checkpoint.pth'
start_epoch = 0

# 尝试恢复 / Try to resume
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

for epoch in range(start_epoch, 50):
    # train_loss = train_one_epoch(...)
    train_loss = 0.0  # 占位
    scheduler.step()
    
    # 每 5 个 epoch 保存一次 / Save every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)
```

### 3.3 跨设备加载

```python
import torch

# ============================================================
# 跨设备加载（GPU 保存，CPU 加载，或反之）
# Cross-device loading (GPU saved, CPU loaded, or vice versa)
# ============================================================

# GPU 训练的模型加载到 CPU
# Load GPU-trained model on CPU
model = MyModel(784, 256, 10)
state_dict = torch.load('model_weights.pth', map_location='cpu')
model.load_state_dict(state_dict)

# 加载到特定 GPU
# Load to specific GPU
state_dict = torch.load('model_weights.pth', map_location='cuda:0')

# 自动选择设备
# Auto-select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('model_weights.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# ============================================================
# weights_only=True（安全加载，PyTorch 2.0+ 推荐）
# Safe loading (PyTorch 2.0+ recommended)
# ============================================================
# torch.load 默认使用 pickle，可能执行恶意代码！
# torch.load uses pickle by default, can execute malicious code!
state_dict = torch.load('model_weights.pth', weights_only=True)
# weights_only=True 只加载张量数据，不执行任何 Python 代码
```

### 3.4 Safetensors 格式（Hugging Face 推荐）

```python
# ============================================================
# safetensors: 更安全、更快的模型保存格式
# safetensors: Safer and faster model saving format
# pip install safetensors
# ============================================================
from safetensors.torch import save_file, load_file

model = MyModel(784, 256, 10)

# 保存 / Save
save_file(model.state_dict(), 'model.safetensors')

# 加载 / Load
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)

# safetensors 优势：
# 1. 安全：不使用 pickle，无代码执行风险
# 2. 快速：零拷贝加载，比 torch.load 快 2-5x
# 3. 标准：Hugging Face Hub 的默认格式
# 4. 支持内存映射（mmap）：加载 70B+ 模型不需要额外内存
```

### 3.5 大模型的分片保存

```python
# ============================================================
# 大模型分片保存（模型太大，拆成多个文件）
# Sharded saving for large models
# ============================================================

# Hugging Face Transformers 自动处理分片
from transformers import AutoModelForCausalLM

# 保存时自动分片（每片最大 5GB）
# Auto-shard on save (max 5GB per shard)
# model.save_pretrained('output/', max_shard_size='5GB')

# 文件结构：
# output/
# ├── model-00001-of-00003.safetensors  (4.9 GB)
# ├── model-00002-of-00003.safetensors  (4.9 GB)
# ├── model-00003-of-00003.safetensors  (3.2 GB)
# └── model.safetensors.index.json      (索引文件)

# 加载时自动合并 / Auto-merge on load
# model = AutoModelForCausalLM.from_pretrained('output/')
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么不推荐 `torch.save(model)`？

```
torch.save(model)（保存整个对象）的问题：

1. 使用 pickle 序列化 → 安全风险
   - pickle 可以执行任意 Python 代码
   - 从不信任的源加载模型可能被攻击

2. 依赖模型类的源代码
   - 如果模型类改了名字或移了包 → 加载失败
   - 如果 PyTorch 版本变了 → 可能不兼容

3. 无法灵活加载
   - 不能部分加载参数
   - 不能跨平台使用

推荐: torch.save(model.state_dict(), path)
  - state_dict 只是 {名字: Tensor} 的字典
  - 与模型类的具体实现解耦
  - 可以部分加载、跨设备迁移
```

### 4.2 state_dict 的部分加载

```python
# ============================================================
# 部分加载（迁移学习中很常用）
# Partial loading (very common in transfer learning)
# ============================================================

# 场景：预训练模型有 12 层，你的模型只用 6 层
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()

# 过滤掉不匹配的键 / Filter mismatched keys
matched_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

# 更新模型参数 / Update model parameters
model_dict.update(matched_dict)
model.load_state_dict(model_dict)

print(f"加载了 {len(matched_dict)}/{len(pretrained_dict)} 个参数")

# strict=False: 忽略不匹配的键（不推荐，可能漏掉问题）
# model.load_state_dict(pretrained_dict, strict=False)
```

## 5. 例题（Worked Examples）

### 例题：实现带自动恢复的完整训练系统

```python
import torch
import torch.nn as nn
import os

class TrainingManager:
    """训练管理器：封装保存、加载、恢复逻辑 / Training manager."""
    
    def __init__(self, model, optimizer, scheduler, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点 / Save checkpoint."""
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        path = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(state, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
    
    def load_latest(self) -> int:
        """加载最新的检查点 / Load latest checkpoint."""
        checkpoints = sorted([
            f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_')
        ])
        if not checkpoints:
            return 0
        
        latest = os.path.join(self.save_dir, checkpoints[-1])
        state = torch.load(latest, weights_only=False)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.best_loss = state['best_loss']
        return state['epoch'] + 1
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 训练一个 MLP 模型 10 个 epoch，每 2 个 epoch 保存一次 checkpoint，训练结束后保存最终的 `state_dict`。

**练习 2：** 解释 `map_location` 参数的作用。如果你在 GPU 0 上训练模型，保存后想在 GPU 1 上加载，应该怎么写？

> **答案：** `torch.load('model.pth', map_location='cuda:1')`

### 进阶题

**练习 3：** 实现一个 `save_top_k` 函数：只保留验证 loss 最低的 k 个 checkpoint，自动删除旧的。

> **参考答案：**
> ```python
> import heapq, os
> 
> class TopKCheckpointer:
>     def __init__(self, save_dir: str, k: int = 3):
>         self.save_dir = save_dir
>         self.k = k
>         self.heap = []  # (loss, path) 最大堆
>
>     def save_if_better(self, model, loss: float, epoch: int):
>         path = f'{self.save_dir}/ckpt_e{epoch}_l{loss:.4f}.pth'
>         if len(self.heap) < self.k:
>             torch.save(model.state_dict(), path)
>             heapq.heappush(self.heap, (-loss, path))
>         elif loss < -self.heap[0][0]:
>             _, old_path = heapq.heapreplace(self.heap, (-loss, path))
>             os.remove(old_path)
>             torch.save(model.state_dict(), path)
> ```

**练习 4:** 对比 `.pth` 和 `.safetensors` 格式的加载速度差异。用 `time.time()` 计时，加载一个 ~100MB 的模型文件。
