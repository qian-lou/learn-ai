# GPU 加速与混合精度 / GPU Acceleration and Mixed Precision

## 1. 背景（Background）

> **为什么要学这个？**
>
> GPU（图形处理器）是深度学习的"发动机"。一块现代 GPU 的矩阵运算能力是 CPU 的 **50-100 倍**。大模型训练和推理离开 GPU 几乎不可能。对于 Java 工程师来说，GPU 加速就像是将单机 Tomcat 升级为分布式集群——同样的代码，但底层硬件的并行度完全不同。
>
> **混合精度训练**则是大模型时代的"标配技术"——通过在计算中同时使用 FP16/BF16（半精度）和 FP32（单精度），可以**减少一半显存占用**并**加速 2-3 倍**，同时保持与 FP32 几乎相同的训练效果。
>
> **在整个体系中的位置：** GPU 加速和混合精度是大模型训练的基础设施。理解它们，才能理解分布式训练（DDP/FSDP）、量化推理、显存优化等高级主题。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | 关键命令 |
|------|------|----------|
| CUDA | NVIDIA GPU 的并行计算平台 | `torch.cuda.is_available()` |
| 设备管理 | CPU ↔ GPU 数据迁移 | `tensor.to('cuda')` |
| FP32 | 单精度浮点数，32 位 | 默认精度 |
| FP16 | 半精度浮点数，16 位 | 范围小，可能溢出 |
| BF16 | 脑浮点数，16 位 | 范围大（推荐） |
| AMP | 自动混合精度 | `torch.amp.autocast` |
| GradScaler | 梯度缩放器 | 防止 FP16 梯度下溢 |

**核心要点：**
- 模型和数据必须在**同一设备**上才能计算
- `pin_memory=True` + `non_blocking=True` 加速 CPU→GPU 传输
- **BF16 > FP16**（BF16 范围与 FP32 相同，不需要 GradScaler）
- 混合精度不是"所有计算都用 FP16"，而是自动选择最优精度

## 3. 内容（Content）

### 3.1 GPU 基础操作

```python
import torch

# ============================================================
# 1. 检查 GPU 环境 / Check GPU environment
# ============================================================
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 设备选择模式（推荐通用写法）
# Device selection (recommended universal pattern)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 2. 数据和模型迁移到 GPU / Move data and model to GPU
# ============================================================
# Tensor 迁移 / Tensor migration
x = torch.randn(3, 4)
x_gpu = x.to(device)          # 方式 1（推荐）
x_gpu = x.cuda()              # 方式 2（仅 GPU 可用时）
x_cpu = x_gpu.cpu()           # 迁回 CPU

# 模型迁移 / Model migration
model = torch.nn.Linear(784, 10)
model = model.to(device)       # 所有参数和缓冲区迁移到 GPU

# ⚠️ 输入数据也必须在 GPU 上！
# ⚠️ Input data must also be on GPU!
x_input = torch.randn(32, 784).to(device)
output = model(x_input)  # OK ✅
# model(torch.randn(32, 784))  # Error! 数据在 CPU，模型在 GPU

# ============================================================
# 3. 高效数据传输 / Efficient data transfer
# ============================================================
from torch.utils.data import DataLoader, TensorDataset

dataloader = DataLoader(
    TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,))),
    batch_size=32,
    pin_memory=True,   # 锁页内存：加速 CPU→GPU 传输
    num_workers=4,      # 多进程加载
)

for data, target in dataloader:
    # non_blocking=True：异步传输，不等待完成
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    # ... 训练 ...
```

### 3.2 显存管理

```python
import torch

# ============================================================
# GPU 显存监控和管理
# GPU memory monitoring and management
# ============================================================

if torch.cuda.is_available():
    # 查看显存使用 / Check memory usage
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"缓存显存:   {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    print(f"最大分配:   {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    
    # 清理显存缓存 / Clear memory cache
    torch.cuda.empty_cache()
    
    # 重置最大显存统计 / Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

# ============================================================
# 显存占用估算
# Memory estimation
# ============================================================
# 7B 参数模型的显存需求：
# Memory requirements for a 7B parameter model:
#
# 模型参数:
#   FP32: 7B × 4B = 28 GB
#   FP16: 7B × 2B = 14 GB
#   INT8: 7B × 1B = 7 GB
#
# 训练时还需要:
#   梯度 (FP32): 7B × 4B = 28 GB
#   优化器状态 (Adam, FP32): 7B × 4B × 2 = 56 GB
#   激活值: 取决于 batch_size 和序列长度
#
# total 训练 (FP32): 28 + 28 + 56 = 112 GB（至少！）
# 混合精度训练可以减半参数和梯度部分
```

### 3.3 混合精度训练（AMP）

```python
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# ============================================================
# 混合精度训练完整模板
# Mixed precision training template
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(
    nn.Linear(784, 512), nn.GELU(),
    nn.Linear(512, 256), nn.GELU(),
    nn.Linear(256, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# GradScaler 防止 FP16 梯度下溢
# GradScaler prevents FP16 gradient underflow
scaler = GradScaler('cuda')

for epoch in range(10):
    model.train()
    for data, target in dataloader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # autocast: 自动选择 FP16/FP32 计算
        # autocast: Automatically choose FP16/FP32
        with autocast('cuda'):
            output = model(data)          # FP16 计算
            loss = criterion(output, target)  # FP32 计算（损失函数）
        
        # 缩放损失 + 反向传播
        # Scale loss + backward
        scaler.scale(loss).backward()
        
        # 反缩放梯度 + 裁剪 + 更新参数
        # Unscale + clip + step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


# ============================================================
# BF16 模式（更简单，推荐 A100/H100）
# BF16 mode (simpler, recommended for A100/H100)
# ============================================================
# BF16 的指数范围与 FP32 相同，不需要 GradScaler！
# BF16 has the same exponent range as FP32, no GradScaler needed!

# 检查 BF16 支持
if torch.cuda.is_bf16_supported():
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast('cuda', dtype=torch.bfloat16):  # 指定 BF16
            output = model(data)
            loss = criterion(output, target)
        
        loss.backward()  # 不需要 scaler！
        optimizer.step()
```

### 3.4 FP32 / FP16 / BF16 对比

```
数据类型对比：

         │ 符号位 │ 指数位 │ 尾数位 │ 总位数 │ 范围          │ 精度
─────────┼────────┼────────┼────────┼────────┼───────────────┼──────
FP32     │ 1      │ 8      │ 23     │ 32     │ ±3.4×10^38    │ 高
FP16     │ 1      │ 5      │ 10     │ 16     │ ±65504        │ 低
BF16     │ 1      │ 8      │ 7      │ 16     │ ±3.4×10^38    │ 中

为什么 BF16 更适合深度学习？
  FP16 的问题：
  - 范围太小（最大 65504），大梯度会溢出（overflow）
  - 小梯度会下溢（underflow）→ 需要 GradScaler 缩放
  
  BF16 的优势：
  - 范围与 FP32 相同 → 不需要 GradScaler
  - 精度虽低于 FP16，但深度学习对精度要求不高
  - A100/H100 GPU 原生支持 BF16，性能与 FP16 相当

性能对比（A100 GPU，4096×4096 矩阵乘法）：
  FP32: 19.5 TFLOPS
  TF32: 156 TFLOPS (Tensor Core)   ← torch.set_float32_matmul_precision('high')
  FP16: 312 TFLOPS (Tensor Core)
  BF16: 312 TFLOPS (Tensor Core)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 GPU 比 CPU 快？

```
CPU vs GPU 架构对比：

CPU（少核高频）：
  ┌─────────────────┐
  │  Core 1 (强大)   │ ← 复杂控制逻辑
  │  Core 2 (强大)   │ ← 大缓存
  │  Core 3 (强大)   │ ← 分支预测
  │  ...             │
  │  Core 8-16       │
  └─────────────────┘
  适合：顺序执行、复杂逻辑、I/O 密集

GPU（多核低频）：
  ┌─────────────────────────────────┐
  │ ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○ │
  │ ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○ │ ← 数千个小核
  │ ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○ │ ← 简单逻辑
  │ ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○ │ ← 同时执行
  └─────────────────────────────────┘
  适合：大规模并行计算（矩阵运算！）
  
  A100: 6912 CUDA cores + 432 Tensor Cores
  H100: 16896 CUDA cores + 528 Tensor Cores
```

### 4.2 GradScaler 的工作原理

```
FP16 梯度下溢问题：
  正常梯度可能很小：1e-7
  FP16 最小正数：~6e-8
  梯度被截断为 0！→ 参数不再更新

GradScaler 解决方案：
  1. 将 loss 放大 scale 倍（如 65536）→ 梯度也放大
  2. 放大后的梯度不会下溢
  3. 更新参数前，将梯度缩小 1/scale
  4. 如果检测到 inf/nan，跳过这步，减小 scale
  5. 如果连续多步正常，增大 scale

  Forward:  loss = model(x)
  Scale:    scaled_loss = loss * scale (65536)
  Backward: grads = scaled_loss.backward()  → 梯度也放大了
  Unscale:  grads = grads / scale           → 恢复原始大小
  Step:     optimizer.step(grads)
```

## 5. 例题（Worked Examples）

### 例题 1：对比不同精度的训练速度和显存

```python
import torch
import torch.nn as nn
import time

def benchmark(dtype_name, use_amp=False, amp_dtype=None):
    """对比不同精度的训练性能 / Benchmark different precisions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.GELU(),
        nn.Linear(4096, 4096), nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(64, 1024, device=device)
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    for _ in range(100):
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = model(x).sum()
            loss.backward()
        else:
            loss = model(x).sum()
            loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"{dtype_name:10s}: {elapsed:.2f}s, Peak Memory: {peak_mem:.0f} MB")

if torch.cuda.is_available():
    benchmark("FP32")
    benchmark("FP16 AMP", use_amp=True, amp_dtype=torch.float16)
    if torch.cuda.is_bf16_supported():
        benchmark("BF16 AMP", use_amp=True, amp_dtype=torch.bfloat16)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 编写代码检查当前系统的 GPU 信息（名称、显存、CUDA 版本），如果没有 GPU 则给出友好提示。

**练习 2：** 解释为什么以下代码会报错：
```python
model = nn.Linear(10, 5).cuda()
x = torch.randn(3, 10)  # CPU 上的数据
output = model(x)        # Error!
```

### 进阶题

**练习 3：** 对比 `pin_memory=True` 和 `pin_memory=False` 在 DataLoader 中的数据传输速度差异。

**练习 4：** 实现一个显存监控装饰器，在每次前向传播前后打印显存变化。

> **参考答案：**
> ```python
> def memory_monitor(func):
>     def wrapper(*args, **kwargs):
>         before = torch.cuda.memory_allocated() / 1e6
>         result = func(*args, **kwargs)
>         after = torch.cuda.memory_allocated() / 1e6
>         print(f"显存变化: {before:.1f} MB → {after:.1f} MB (Δ{after-before:+.1f} MB)")
>         return result
>     return wrapper
> ```
