# Tensor 基础与自动求导 / Tensor Basics and Autograd

## 1. 背景（Background）

> **为什么要学这个？**
>
> PyTorch Tensor 是深度学习的"原子数据结构"。它类似 NumPy 的 ndarray，但有两个核心超能力：**GPU 加速**和**自动微分（autograd）**。对于 Java 工程师来说，可以把 Tensor 类比为一个**增强版的多维数组**——就像 Java 的 `int[][]` 升级为一个可以在 GPU 上计算、并自动记录运算历史的智能对象。
>
> 在大模型开发中，所有的数据（文本 token、模型参数、梯度、注意力权重）都以 Tensor 的形式存储和计算。掌握 Tensor 的创建、形状操作、设备管理，是使用 PyTorch 的第一步。
>
> **在整个体系中的位置：** Tensor 是 PyTorch 的基础。`nn.Module`、训练循环、GPU 加速——一切都建立在 Tensor 之上。

## 2. 知识点（Key Concepts）

| 概念 | NumPy | PyTorch | Java 类比 |
|------|-------|---------|-----------|
| 基本容器 | ndarray | Tensor | 多维数组 |
| 创建方式 | `np.array()` | `torch.tensor()` | `new int[]{}` |
| 数据类型 | `dtype=np.float32` | `dtype=torch.float32` | `float` |
| GPU 支持 | ❌ | ✅ `tensor.to('cuda')` | - |
| 自动微分 | ❌ | ✅ `requires_grad=True` | - |
| 维度操作 | `reshape/transpose` | `view/permute` | - |

**核心要点：**
- Tensor 和 ndarray 可以**零拷贝互转**（共享内存时）
- `requires_grad=True` 开启梯度追踪，用于反向传播
- **Shape 注释**是大模型代码的必备习惯：`# Shape: [B, S, D]`
- 设备管理：CPU ↔ GPU 需要显式 `.to(device)` 转移

## 3. 内容（Content）

### 3.1 Tensor 创建

```python
import torch
import numpy as np

# ============================================================
# 1. 基本创建方式
# Basic creation methods
# ============================================================

# 从 Python 列表创建 / Create from Python list
x = torch.tensor([1, 2, 3])               # Shape: [3]
y = torch.tensor([[1, 2], [3, 4]])         # Shape: [2, 2]

# 指定数据类型 / Specify dtype
# 大模型常用 float32（训练）和 float16/bfloat16（推理）
x_float = torch.tensor([1.0, 2.0], dtype=torch.float32)  # FP32
x_half = torch.tensor([1.0, 2.0], dtype=torch.float16)   # FP16
x_bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)  # BF16（推荐）

# ============================================================
# 2. 常用创建函数
# Common creation functions
# ============================================================

zeros = torch.zeros(3, 4)           # Shape: [3, 4] 全零
ones = torch.ones(2, 3)             # Shape: [2, 3] 全一
rand = torch.randn(2, 3)            # Shape: [2, 3] 标准正态分布
arange = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

# 创建与已有 Tensor 相同形状的张量
# Create tensor with same shape as existing
like_zeros = torch.zeros_like(rand)  # 同形状、同设备、同dtype
like_ones = torch.ones_like(rand)

# 单位矩阵 / Identity matrix
eye = torch.eye(3)  # Shape: [3, 3]

# ============================================================
# 3. NumPy 互转（零拷贝！）
# NumPy interop (zero-copy!)
# ============================================================

# Tensor → NumPy（共享内存，修改会互相影响）
# Tensor → NumPy (shared memory, modifications affect each other)
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()             # 零拷贝 / Zero-copy
t[0] = 99.0
print(n[0])               # 99.0 ← 共享内存！

# NumPy → Tensor（同样共享内存）
# NumPy → Tensor (also shared memory)
n2 = np.array([4.0, 5.0, 6.0])
t2 = torch.from_numpy(n2)   # 零拷贝

# 注意：GPU 上的 Tensor 必须先 .cpu() 才能转 NumPy
# Note: GPU tensors must .cpu() first before converting to NumPy
# gpu_tensor.cpu().numpy()

# 有 grad 的 Tensor 必须先 .detach()
# Tensors with grad must .detach() first
grad_tensor = torch.tensor([1.0], requires_grad=True)
# grad_tensor.numpy()  # Error!
grad_tensor.detach().numpy()  # OK ✅
```

### 3.2 形状操作（Shape Operations）

```python
import torch

# ============================================================
# 形状操作是大模型开发最常用的操作
# Shape operations are the most common in LLM development
# ============================================================

a = torch.randn(2, 3, 4)  # Shape: [2, 3, 4]
print(f"形状: {a.shape}")   # torch.Size([2, 3, 4])
print(f"维度数: {a.dim()}")  # 3
print(f"元素数: {a.numel()}")  # 24

# ============================================================
# 1. view / reshape — 改变形状（不改变数据）
# view / reshape — Change shape (data unchanged)
# ============================================================
b = a.view(2, 12)           # Shape: [2, 12] — 要求内存连续
c = a.reshape(6, 4)         # Shape: [6, 4] — 自动处理非连续
d = a.view(-1)              # Shape: [24] — 展平，-1 自动推断

# 在大模型中常用：合并/拆分注意力头
# In LLMs: merge/split attention heads
# [B, S, n_heads, d_head] ↔ [B, S, d_model]
heads = torch.randn(8, 512, 12, 64)  # Shape: [B, S, H, D]
merged = heads.view(8, 512, -1)       # Shape: [B, S, 768] — 合并头

# ============================================================
# 2. permute / transpose — 交换维度
# permute / transpose — Swap dimensions
# ============================================================
e = a.permute(0, 2, 1)     # Shape: [2, 4, 3] — 任意维度排列
f = a.transpose(1, 2)      # Shape: [2, 4, 3] — 只交换两个维度

# 大模型中：将 [B, S, H, D] 转为 [B, H, S, D]（注意力计算需要）
# In LLMs: [B, S, H, D] → [B, H, S, D] (needed for attention)
attn_input = torch.randn(8, 512, 12, 64)       # [B, S, H, D]
attn_ready = attn_input.permute(0, 2, 1, 3)    # [B, H, S, D]

# ============================================================
# 3. unsqueeze / squeeze — 增删维度
# unsqueeze / squeeze — Add/remove dimensions
# ============================================================
g = torch.randn(3, 4)     # Shape: [3, 4]
h = g.unsqueeze(0)         # Shape: [1, 3, 4] — 在第 0 维增加
i = g.unsqueeze(-1)        # Shape: [3, 4, 1] — 在最后增加

j = torch.randn(1, 3, 1, 4)
k = j.squeeze()            # Shape: [3, 4] — 去掉所有 size=1 的维度
l = j.squeeze(0)           # Shape: [3, 1, 4] — 只去第 0 维

# ============================================================
# 4. cat / stack — 拼接张量
# cat / stack — Concatenate tensors
# ============================================================
t1 = torch.randn(2, 3)    # Shape: [2, 3]
t2 = torch.randn(2, 3)    # Shape: [2, 3]

# cat: 沿已有维度拼接 / Concatenate along existing dim
cat_0 = torch.cat([t1, t2], dim=0)  # Shape: [4, 3]
cat_1 = torch.cat([t1, t2], dim=1)  # Shape: [2, 6]

# stack: 创建新维度拼接 / Stack along new dim
stacked = torch.stack([t1, t2], dim=0)  # Shape: [2, 2, 3]
```

### 3.3 Tensor 运算

```python
import torch

# ============================================================
# 1. 逐元素运算 / Element-wise operations
# ============================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)       # [5, 7, 9]
print(a * b)       # [4, 10, 18]（逐元素乘，非矩阵乘！）
print(a ** 2)      # [1, 4, 9]
print(torch.exp(a))  # [e¹, e², e³]

# ============================================================
# 2. 矩阵乘法（大模型核心操作）
# Matrix multiplication (core LLM operation)
# ============================================================

# @ 运算符 = torch.matmul（推荐）
A = torch.randn(2, 3)   # Shape: [2, 3]
B = torch.randn(3, 4)   # Shape: [3, 4]
C = A @ B                # Shape: [2, 4] — 矩阵乘法

# 批量矩阵乘法（Attention 中的 QK^T）
# Batched matmul (QK^T in Attention)
Q = torch.randn(8, 12, 512, 64)  # Shape: [B, H, S, D]
K = torch.randn(8, 12, 512, 64)  # Shape: [B, H, S, D]
attn_scores = Q @ K.transpose(-2, -1)  # Shape: [B, H, S, S]
# 每个 batch、每个 head 独立做矩阵乘法

# ============================================================
# 3. 广播机制（Broadcasting）
# Broadcasting mechanism
# ============================================================
x = torch.randn(2, 3)    # Shape: [2, 3]
y = torch.tensor([1.0])  # Shape: [1] — 自动广播
print((x + y).shape)     # Shape: [2, 3]

# ============================================================
# 4. 聚合操作 / Aggregation
# ============================================================
t = torch.randn(3, 4)
print(t.sum())              # 全部求和
print(t.mean(dim=1))        # 按行求均值，Shape: [3]
print(t.max(dim=0).values)  # 按列求最大值，Shape: [4]
print(t.argmax(dim=1))      # 按行求最大值索引，Shape: [3]
```

### 3.4 自动求导（Autograd）基础

```python
import torch

# ============================================================
# autograd 是 PyTorch 的自动微分引擎
# autograd is PyTorch's automatic differentiation engine
# ============================================================

# 1. 开启梯度追踪 / Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1
y.backward()               # 反向传播
print(f"x = {x.item()}")
print(f"dy/dx = {x.grad.item()}")  # 2x+3 = 2*2+3 = 7.0

# 2. 多变量梯度 / Multi-variable gradient
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
z = a**2 * b + b**3
z.backward()
print(f"dz/da = {a.grad.item()}")  # 2ab = 2*1*2 = 4.0
print(f"dz/db = {b.grad.item()}")  # a² + 3b² = 1 + 12 = 13.0

# 3. 模型参数的梯度 / Gradient of model parameters
model = torch.nn.Linear(3, 1)
x = torch.randn(5, 3)
y = model(x).sum()
y.backward()
print(f"权重梯度形状: {model.weight.grad.shape}")  # [1, 3]
print(f"偏置梯度形状: {model.bias.grad.shape}")     # [1]
```

## 4. 详细推理（Deep Dive）

### 4.1 Tensor 的内存布局

```
Tensor 在内存中的布局：

逻辑视图（2D 矩阵 [2, 3]）：
  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │
  ├───┼───┼───┤
  │ 4 │ 5 │ 6 │
  └───┴───┴───┘

物理内存（行优先 / Row-major）：
  [1, 2, 3, 4, 5, 6]
  
  stride = (3, 1)
  - 行方向跳 3 步（从 1 到 4）
  - 列方向跳 1 步（从 1 到 2）

view() vs reshape() 的区别：
  view()   → 要求内存连续（contiguous），否则报错
  reshape()→ 自动处理非连续情况（可能拷贝数据）
  
  permute() 后内存不连续：
    a = torch.randn(2, 3)  # stride: (3, 1) — 连续
    b = a.t()                # stride: (1, 3) — 非连续！
    # b.view(-1)  ← Error!
    b.contiguous().view(-1)  # OK，先拷贝为连续内存
    b.reshape(-1)            # OK，内部自动处理
```

### 4.2 dtype 与显存管理

```
常见 dtype 及其显存占用（每参数）：

  dtype          │ 字节数 │ 范围      │ 用途
  ───────────────┼────────┼───────────┼──────────
  float32 (FP32) │ 4B     │ ±3.4e38   │ 训练（默认）
  float16 (FP16) │ 2B     │ ±65504    │ 混合精度训练
  bfloat16(BF16) │ 2B     │ ±3.4e38   │ 大模型训练（推荐）
  float64 (FP64) │ 8B     │ ±1.8e308  │ 科学计算（很少用）
  int8           │ 1B     │ -128~127  │ 量化推理
  int4           │ 0.5B   │ -8~7      │ 极限量化

显存估算（以 LLaMA-7B 为例，7B=70 亿参数）：
  FP32: 7B × 4B = 28 GB
  FP16: 7B × 2B = 14 GB
  INT8: 7B × 1B = 7 GB
  INT4: 7B × 0.5B = 3.5 GB
```

### 4.3 contiguous 与 stride

```python
import torch

# ============================================================
# 理解 stride 和 contiguous
# Understanding stride and contiguous
# ============================================================

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Shape: {a.shape}")           # [2, 3]
print(f"Stride: {a.stride()}")       # (3, 1)
print(f"Contiguous: {a.is_contiguous()}")  # True

# transpose 改变 stride 但不移动数据
# Transpose changes stride without moving data
b = a.t()
print(f"Shape: {b.shape}")           # [3, 2]
print(f"Stride: {b.stride()}")       # (1, 3)
print(f"Contiguous: {b.is_contiguous()}")  # False

# 需要 contiguous() 才能 view
# Need contiguous() before view
c = b.contiguous()
print(f"Contiguous after: {c.is_contiguous()}")  # True
d = c.view(-1)  # 现在可以 view 了
```

## 5. 例题（Worked Examples）

### 例题 1：实现注意力分数计算

```python
import torch
import torch.nn.functional as F

# ============================================================
# 模拟 Scaled Dot-Product Attention 的张量操作
# Simulate tensor ops in Scaled Dot-Product Attention
# ============================================================

B, H, S, D = 2, 4, 8, 64  # batch, heads, seq_len, head_dim

Q = torch.randn(B, H, S, D)  # Shape: [2, 4, 8, 64]
K = torch.randn(B, H, S, D)  # Shape: [2, 4, 8, 64]
V = torch.randn(B, H, S, D)  # Shape: [2, 4, 8, 64]

# 1. 计算注意力分数 QK^T / √d
# Compute attention scores QK^T / √d
scores = Q @ K.transpose(-2, -1) / (D ** 0.5)  # Shape: [2, 4, 8, 8]
print(f"注意力分数形状: {scores.shape}")

# 2. Softmax 归一化
attn_weights = F.softmax(scores, dim=-1)  # Shape: [2, 4, 8, 8]
print(f"每行之和: {attn_weights.sum(dim=-1)[0, 0]}")  # 全为 1.0

# 3. 加权求和得到输出
output = attn_weights @ V  # Shape: [2, 4, 8, 64]
print(f"输出形状: {output.shape}")

# 4. 合并多头
output_merged = output.permute(0, 2, 1, 3).contiguous()  # [B, S, H, D]
output_merged = output_merged.view(B, S, -1)               # [B, S, H*D]
print(f"合并后形状: {output_merged.shape}")  # [2, 8, 256]
```

### 例题 2: CPU vs GPU 性能对比

```python
import torch
import time

# ============================================================
# 矩阵乘法性能对比：CPU vs GPU
# Matrix multiplication benchmark: CPU vs GPU
# ============================================================

size = 4096

# CPU 测试 / CPU benchmark
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

# GPU 测试（如果可用）/ GPU benchmark (if available)
if torch.cuda.is_available():
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()  # 确保 GPU 就绪
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()  # 等待 GPU 完成
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.1f}x")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 创建以下张量，并打印它们的 shape、dtype 和 device：
- 一个 3×4 的全零张量（float32）
- 一个 2×3×4 的标准正态分布张量
- 从列表 `[[1,2],[3,4]]` 创建的整数张量

**练习 2：** 给定 `a = torch.randn(4, 3, 2)`，用 `view`、`permute`、`unsqueeze` 分别将其变为以下形状：
- `[4, 6]`
- `[4, 2, 3]`
- `[1, 4, 3, 2]`

### 进阶题

**练习 3：** 实现一个函数，模拟多头注意力中的"拆分头"和"合并头"操作：

```python
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """将 [B, S, D] 拆为 [B, H, S, D_head]"""
    # TODO: 实现
    pass

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """将 [B, H, S, D_head] 合并为 [B, S, D]"""
    # TODO: 实现
    pass
```

> **参考答案：**
> ```python
> def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
>     B, S, D = x.shape
>     return x.view(B, S, n_heads, D // n_heads).permute(0, 2, 1, 3)
>
> def merge_heads(x: torch.Tensor) -> torch.Tensor:
>     B, H, S, D = x.shape
>     return x.permute(0, 2, 1, 3).contiguous().view(B, S, H * D)
> ```

**练习 4：** 解释为什么 `a.t().view(-1)` 会报错，而 `a.t().reshape(-1)` 不会。从 stride 和 contiguous 的角度分析。
