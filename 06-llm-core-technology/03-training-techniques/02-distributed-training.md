# 分布式训练 / Distributed Training

## 1. 背景（Background）

> **为什么要学这个？**
>
> GPT-3 有 175B 参数，占用约 **350GB 显存**（FP16），而单张 A100 只有 80GB 显存——根本放不下！分布式训练是训练大模型的**必备技术**。
>
> 对于 Java 工程师来说，分布式训练就像是**分布式系统**——数据分片（Sharding）、负载均衡、节点通信、一致性保证，概念完全相通。
>
> **在整个体系中的位置：** 分布式训练让 Scaling Laws 成为可能——没有分布式，就不可能训练超过单卡容量的大模型。

## 2. 知识点（Key Concepts）

| 并行策略 | 拆分维度 | 通信量 | 适用规模 |
|----------|---------|--------|---------|
| 数据并行（DDP） | 数据 | 梯度同步 | 多卡 |
| 模型并行（TP） | 层内张量 | 中间结果 | 多卡 |
| 流水线并行（PP） | 层 | 激活值 | 多节点 |
| FSDP/ZeRO | 参数+梯度+优化器 | 参数通信 | 大规模 |

## 3. 内容（Content）

### 3.1 数据并行（DDP）

```
DDP (Distributed Data Parallel):

原理：
  每张卡有完整模型副本，各自处理不同的数据
  前向传播: 各卡独立计算 → 得到各自的梯度
  反向传播: All-Reduce 同步所有卡的梯度
  参数更新: 各卡用同步后的梯度更新

  GPU 0: batch_0 → loss_0 → grad_0 ─┐
  GPU 1: batch_1 → loss_1 → grad_1 ─┤── All-Reduce ── avg_grad
  GPU 2: batch_2 → loss_2 → grad_2 ─┤
  GPU 3: batch_3 → loss_3 → grad_3 ─┘

优势: 实现简单，Scale 效率高
限制: 每张卡必须放下完整模型
```

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================
# PyTorch DDP 基本用法
# PyTorch DDP basic usage
# ============================================================

def train_ddp(rank, world_size):
    """DDP 训练函数 / DDP training function."""
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 模型
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据加载器（自动分片）
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    dist.destroy_process_group()

# 启动: torchrun --nproc_per_node=4 train.py
```

### 3.2 FSDP（Fully Sharded Data Parallel）

```
FSDP 的核心思想：

DDP 的浪费：
  4 张卡，每张都存完整模型 → 4 份冗余
  显存 = 4 × (参数 + 梯度 + 优化器状态)

FSDP 的优化：
  将参数/梯度/优化器状态切片（shard）到各卡
  只在需要时恢复完整参数（All-Gather）
  用后立即释放（Reduce-Scatter）

  显存 = 参数/4 + 梯度/4 + 优化器/4 + 激活值
  → 显存减少 ~75%！

ZeRO (DeepSpeed) 分级:
  ZeRO-1: 只切片优化器状态        → 显存减少 ~33%
  ZeRO-2: 切片优化器 + 梯度       → 显存减少 ~50%
  ZeRO-3: 切片参数 + 梯度 + 优化器 → 显存减少 ~75% (≈ FSDP)
```

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# FSDP 基本用法
model = MyModel()
model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)
```

### 3.3 模型并行（Tensor Parallel + Pipeline Parallel）

```
Tensor Parallel（张量并行）:
  将单层的权重矩阵切分到多卡
  例: Linear(4096, 4096) → 2 张卡各 Linear(4096, 2048)
  
  适合: 大维度的 Attention 和 FFN 层
  用于: Megatron-LM

Pipeline Parallel（流水线并行）:
  将模型按层切分到多卡
  GPU 0: Layer 0-7
  GPU 1: Layer 8-15
  GPU 2: Layer 16-23
  GPU 3: Layer 24-31
  
  问题: "气泡"时间（某些 GPU 在等待其他 GPU）
  优化: Micro-batching（将 batch 切为更小的 micro-batch）

实际训练大模型时，三种并行通常组合使用（3D 并行）:
  TP × PP × DP = 总GPU数
  例: 8 TP × 8 PP × 16 DP = 1024 GPUs
```

### 3.4 显存分析

```
训练 7B 模型的显存需求:

  参数 (FP16):       14 GB  (7B × 2 bytes)
  梯度 (FP16):       14 GB
  优化器 (FP32):     56 GB  (Adam: 参数 + 一阶动量 + 二阶动量 = 4×)
  激活值:           ~20 GB  (取决于 batch size 和序列长度)
  ─────────────────────────
  总计:             ~104 GB → 需要至少 2 张 A100-80GB

  用 FSDP (4 卡):
  参数: 14/4 = 3.5 GB + 梯度: 14/4 = 3.5 GB + 优化器: 56/4 = 14 GB
  每卡: ~21 GB + 激活值 → 单卡 A100 完全够用
```

## 4. 详细推理（Deep Dive）

### 4.1 通信瓶颈

```
分布式训练的核心挑战：GPU 间通信

DDP All-Reduce: 需要同步所有 GPU 的梯度
  梯度大小 = 参数量 × 2 bytes（FP16）
  7B 模型: 14 GB 梯度需要同步
  
  NVLink 带宽: 600 GB/s（同一节点，快）
  InfiniBand: 100-400 Gb/s（跨节点，慢）
  以太网: 10-100 Gb/s（更慢）

优化策略:
  1. 梯度压缩（Gradient Compression）
  2. 通信与计算重叠（Overlap）
  3. 梯度累积（减少通信频率）
```

## 5. 例题（Worked Examples）

### 例题：梯度累积模拟大 Batch

```python
# 梯度累积: 用小物理 batch 模拟大逻辑 batch
gradient_accumulation_steps = 4
for step, batch in enumerate(dataloader):
    loss = model(batch) / gradient_accumulation_steps
    loss.backward()
    
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
# 等价于 4 倍的 batch size，但显存不变
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 `torchrun` 在单机多卡上运行一个 DDP 训练。

**练习 2：** 计算训练 LLaMA-70B 模型需要多少张 A100-80GB（用 FSDP）。

### 进阶题

**练习 3：** 用 FSDP 训练一个 1B 参数的模型，对比 DDP 和 FSDP 的显存占用。

**练习 4：** 研究 DeepSpeed ZeRO-3 的配置，用 `deepspeed` 训练一个模型。
