# 推理加速 / Inference Acceleration

## 1. 背景（Background）

> **为什么要学这个？**
>
> 大模型推理延迟高、成本大。KV Cache 避免重复计算，FlashAttention 优化内存访问，Speculative Decoding 用小模型加速大模型——这些技术让推理速度提升 **2-10 倍**。
>
> 对于 Java 工程师来说，这些优化就像 **Redis 缓存 + 批处理 + 异步处理**——减少重复计算、优化 IO、并行化。

## 2. 知识点（Key Concepts）

| 技术 | 原理 | 加速效果 |
|------|------|---------|
| KV Cache | 缓存已计算的 K/V | 避免 O(N²)重复计算 |
| FlashAttention | IO 感知注意力 | 速度 2-4x |
| Continuous Batching | 动态组批 | 吞吐量 10x+ |
| Speculative Decoding | 小模型生成候选 | 速度 2-3x |
| PagedAttention | 分页管理 KV cache | 显存利用率 2-4x |

## 3. 内容（Content）

### 3.1 KV Cache

```
KV Cache 原理：

自回归生成时，每个新 token 需要与前面所有 token 交互：
  Step 1: [A]           → 计算 K₁, V₁
  Step 2: [A, B]        → 计算 K₁, V₁（重复！）+ K₂, V₂
  Step 3: [A, B, C]     → 计算 K₁, V₁（重复！）+ K₂, V₂（重复！）+ K₃, V₃

有 KV Cache:
  Step 1: 计算 K₁, V₁ → 存入 Cache
  Step 2: 从 Cache 取 K₁, V₁ + 计算 K₂, V₂ → 存入 Cache
  Step 3: 从 Cache 取 K₁, V₁, K₂, V₂ + 计算 K₃, V₃

→ 每步只需计算 1 个 token 的 K/V，而非重新计算所有

KV Cache 显存估算:
  大小 = 2 × n_layers × n_heads × d_head × seq_len × batch × dtype_size
  LLaMA-7B, seq=2048, batch=1:
  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes ≈ 1 GB
```

### 3.2 FlashAttention

```
FlashAttention 核心思想：

传统 Attention 的瓶颈不在计算，而在内存访问！

GPU 内存层次:
  SRAM (片上): 20 MB, 19 TB/s     ← 快但小
  HBM (显存):  80 GB, 2 TB/s      ← 大但慢

传统实现:
  Q, K → HBM 读取 → 计算 QK^T → HBM 写入 → 读取 → Softmax → HBM 写入 → 读取 → ×V
  每步都要反复读写 HBM！

FlashAttention:
  将 Q, K, V 分块（tiling）→ 每块完全在 SRAM 中计算
  → 减少 HBM 访问次数 → 速度提升 2-4x

使用方式:
  PyTorch 2.0+: torch.nn.functional.scaled_dot_product_attention()
  自动使用 FlashAttention（如果硬件支持）
```

### 3.3 Continuous Batching

```
传统 Static Batching:
  请求 1 (长度 100): [■■■■■■■■■■]
  请求 2 (长度 30):  [■■■_______]  ← 等待最长的完成
  请求 3 (长度 50):  [■■■■■_____]

  问题: 短请求被长请求拖累 → GPU 利用率低

Continuous Batching (vLLM 使用):
  请求 1: [■■■■■■■■■■]
  请求 2: [■■■][新请求4: ■■■■]  ← 2完成后立即插入新请求
  请求 3: [■■■■■][新请求5: ■■]

  → GPU 始终满载 → 吞吐量提升 10x+
```

## 4. 详细推理（Deep Dive）

### 4.1 Speculative Decoding

```
原理:
  Draft Model (小模型, 如 1B): 快速生成 K 个候选 token
  Target Model (大模型, 如 70B): 一次验证 K 个 token
  
  如果候选被接受 → 一次前向传播生成 K 个 token
  如果被拒绝 → 从拒绝位置重新生成

加速原因:
  大模型验证 K 个 token (1 次前向) << 大模型逐个生成 K 个 token (K 次前向)
```

## 5. 例题（Worked Examples）

```python
import torch

# FlashAttention via PyTorch 2.0
q = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)

# 自动选择最优实现（FlashAttention / Math / Efficient）
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 对比开启/关闭 KV Cache 的推理速度。

*参考答案*：

```python
import time, torch

def timed(use_cache):
    t = time.time()
    model.generate(**inputs, max_new_tokens=200, use_cache=use_cache)
    return time.time() - t

# use_cache=False 每步都重算全部历史 K/V → O(N²)
# use_cache=False recomputes all history K/V each step → O(N²)
print("cache on :", timed(True))   # 关闭后明显变慢
print("cache off:", timed(False))  # generation 长度越大差距越大
```

结论：开启 KV Cache 后单步只计算新 token 的 K/V，时间复杂度从约 O(N²) 降到 O(N)；序列越长，加速越明显（数倍到数量级）。代价是显存换时间。

**练习 2：** 计算 LLaMA-70B 在 seq_len=4096 时的 KV Cache 大小。

*参考答案*：

公式：`2 × n_layers × n_kv_heads × d_head × seq_len × batch × dtype_size`。

LLaMA-2-70B 用 GQA，`n_layers=80`，`n_kv_heads=8`（注意不是 64 个 query head），`d_head=128`，FP16=2 字节，batch=1：

```
2 × 80 × 8 × 128 × 4096 × 1 × 2 B
= 1,342,177,280 B ≈ 1.25 GB
```

要点：必须用 **KV head 数（GQA 下为 8）** 而非 query head 数，否则会高估 8 倍。若误按 MHA（64 头）算则约 10 GB——这正是 GQA 大幅节省 KV Cache 的原因。

### 进阶题

**练习 3：** 用 PyTorch 的 `scaled_dot_product_attention` 替换手写 attention。

*参考答案*：

```python
import torch.nn.functional as F

# 手写版（教学，慢且占显存）/ hand-written (slow, memory-heavy)
def naive_attn(q, k, v):
    w = (q @ k.transpose(-2, -1)) / q.size(-1) ** 0.5
    return F.softmax(w, dim=-1) @ v

# SDPA：自动选择 FlashAttention / 内存高效后端，内部完成缩放与 causal mask
# SDPA: auto-picks FlashAttention/mem-efficient backend; scaling + mask built-in
def sdpa_attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
# 形状均为 [B, n_heads, seq, d_head] / shapes: [B, H, S, D]
```

要点：SDPA 内部已做 1/√d 缩放，**不要**再手动除；`is_causal=True` 替代手写下三角 mask；fp16/bf16 + 满足对齐时自动走 FlashAttention，显存从 O(N²) 降到 O(N)。

**练习 4：** 研究 vLLM 的 PagedAttention 原理，理解虚拟内存管理思想。

*参考答案*：

核心类比——操作系统虚拟内存的"分页"：

- **痛点**：传统按 `max_len` 为每个请求预留连续 KV Cache，实际长度远小于上限，造成大量内部碎片，显存利用率常仅 ~30%。
- **做法**：把 KV Cache 切成固定大小的 **block（页）**，逻辑序列经 **block table（页表）** 映射到物理 block，**非连续**存放，按需分配，几乎无碎片。
- **收益**：(1) 利用率提升到 90%+，可容纳更大 batch；(2) 相同前缀的请求共享物理 block，即 **Prefix Caching**；(3) 与 continuous batching 协同，吞吐提升数倍。
- **延伸**：同源思想还支撑 beam search 的 copy-on-write，以及 2026 年常见的投机解码（speculative decoding）等优化。可读 vLLM 论文与源码 `block_manager` 验证理解。
