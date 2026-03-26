# 缩放定律 / Scaling Laws

## 1. 背景（Background）

> **为什么要学这个？**
>
> Scaling Laws（缩放定律）是大模型时代最重要的**理论发现**——它揭示了模型性能与参数量、数据量、计算量之间存在**可预测的幂律关系**。这意味着我们可以在训练之前就估算模型的最终性能和最优资源分配。
>
> 对于 Java 工程师来说，Scaling Laws 就像是**性能基准测试**——通过小规模实验预测大规模系统的表现，指导容量规划和资源分配。
>
> **在整个体系中的位置：** Scaling Laws 解释了"为什么大模型比小模型好"，也指导了"应该训练多大的模型、用多少数据"。

## 2. 知识点（Key Concepts）

| 定律 | 提出者 | 核心观点 |
|------|--------|---------|
| Kaplan Scaling | OpenAI (2020) | 模型越大越好，数据相对不那么重要 |
| Chinchilla Law | DeepMind (2022) | 模型和数据应等比例增长 |
| Emergent Abilities | Google (2022) | 某些能力只在超大规模才涌现 |

## 3. 内容（Content）

### 3.1 Kaplan Scaling Laws (2020)

```
核心发现（OpenAI, Kaplan et al.）：

模型性能（Loss）与三个因素的幂律关系：

  L(N) ∝ N^{-0.076}    — N: 参数量
  L(D) ∝ D^{-0.095}    — D: 数据量（token 数）
  L(C) ∝ C^{-0.050}    — C: 计算量（FLOPs）

关键推论：
  1. 参数量翻倍，Loss 降低约 5%
  2. 数据量翻倍，Loss 降低约 6.5%
  3. 计算量翻倍，Loss 降低约 3.4%

Kaplan 的建议：
  固定计算预算时，应该优先增大模型
  → 这导致了 GPT-3 (175B) 只用 300B tokens 训练
```

### 3.2 Chinchilla Laws (2022)

```
DeepMind 的修正（Hoffmann et al.）：

Kaplan 过度强调模型大小，忽略了数据量！

Chinchilla 最优比例：
  对于给定计算预算 C：
  最优参数量 N_opt ∝ C^{0.5}
  最优数据量 D_opt ∝ C^{0.5}
  
  → 参数量和数据量应该等比例增长！

经验法则：
  N 参数的模型，最优训练数据 ≈ 20N tokens
  
  7B 模型  → 140B tokens
  13B 模型 → 260B tokens
  70B 模型 → 1.4T tokens

Chinchilla 验证：
  70B 参数 + 1.4T tokens = Chinchilla
  性能超越 280B 参数的 Gopher（只用 300B tokens）
  → 用更少参数 + 更多数据 = 更好效果 + 更低推理成本
```

### 3.3 计算成本估算

```python
# ============================================================
# 训练成本估算 / Training cost estimation
# ============================================================

def estimate_training_cost(
    num_params_billion: float,
    num_tokens_billion: float,
    gpu_flops_tflops: float = 312,  # A100 BF16 理论峰值
    gpu_utilization: float = 0.4,   # 实际利用率
    gpu_cost_per_hour: float = 2.0, # 美元/GPU/小时
) -> dict:
    """估算训练成本.
    
    训练 FLOPs ≈ 6 × N × D (前向 + 反向)
    """
    N = num_params_billion * 1e9
    D = num_tokens_billion * 1e9
    
    # 总 FLOPs
    total_flops = 6 * N * D
    
    # GPU 小时
    effective_tflops = gpu_flops_tflops * gpu_utilization
    gpu_seconds = total_flops / (effective_tflops * 1e12)
    gpu_hours = gpu_seconds / 3600
    
    # 成本
    cost = gpu_hours * gpu_cost_per_hour
    
    return {
        "total_flops": f"{total_flops:.2e}",
        "gpu_hours": f"{gpu_hours:,.0f}",
        "cost_usd": f"${cost:,.0f}",
        "a100_days_1000gpu": f"{gpu_hours/1000/24:.1f} 天",
    }

# 估算不同规模模型的训练成本
for n, d in [(7, 140), (13, 260), (70, 1400)]:
    result = estimate_training_cost(n, d)
    print(f"{n}B params, {d}B tokens: {result}")
```

### 3.4 Scaling Laws 的实际影响

```
Scaling Laws 如何指导决策：

1. 模型大小选择:
   预算固定 → Chinchilla Law 算出最优 N 和 D
   
2. 训练数据量:
   LLaMA-1: 7B + 1T tokens (过拟合区间)
   LLaMA-2: 7B + 2T tokens（远超 Chinchilla 最优）
   LLaMA-3: 8B + 15T tokens（极度过训练，但推理更高效）
   
   趋势: 工业界倾向于过度训练小模型
   → 推理成本 >> 训练成本，所以宁可多训练也要小模型

3. 能力预测:
   小规模实验 → 拟合幂律曲线 → 预测大模型性能
   → 避免盲目砸钱
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么是幂律而非线性？

```
收益递减的数学解释：

Loss ∝ N^{-α} 意味着:
  从 1B → 2B (翻倍): Loss 降低 ~5%
  从 10B → 20B (翻倍): Loss 降低 ~5%（同样翻倍，同样收益）
  从 100B → 200B (翻倍): Loss 降低 ~5%

但从 1B → 100B (100倍): Loss 只降低 ~30%
→ 巨大投入，收益有限

这就是为什么 GPT-5 比 GPT-4 好的幅度
远小于 GPT-4 比 GPT-3 好的幅度
```

### 4.2 涌现能力与 Scaling

```
涌现 ≠ 连续改进

  大部分能力: Loss 平滑降低 → 性能平滑提升
  涌现能力: Loss 缓慢降低 → 到某个临界点突然获得能力

  例: Chain-of-Thought 推理
    10B: 完全不行 (随机猜)
    50B: 勉强可以
    100B+: 稳定有效

  可能解释: 涌现能力需要多种子能力同时达标
  → 每个子能力单独看是连续改进
  → 但需要"全部满足"才能表现为宏观能力
```

## 5. 例题（Worked Examples）

### 例题：拟合 Scaling Law 曲线

```python
import numpy as np

# 模拟不同规模模型的 Loss
params = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])  # 十亿参数
losses = np.array([3.2, 2.8, 2.6, 2.3, 2.15, 1.95])

# 拟合幂律: L = a * N^(-b)
log_params = np.log(params)
log_losses = np.log(losses)
b, log_a = np.polyfit(log_params, log_losses, 1)
a = np.exp(log_a)
print(f"L = {a:.2f} * N^({b:.3f})")
print(f"预测 100B 模型 Loss: {a * 100**b:.2f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 Chinchilla Law 计算：如果你有 1000 A100-天 的预算，应该训练多大的模型？

**练习 2：** 解释为什么 LLaMA-3 (8B) 用 15T tokens 远超 Chinchilla 最优数据量。

### 进阶题

**练习 3：** 收集 5 个不同大小模型在同一 benchmark 上的分数，拟合 Scaling Law 曲线，预测更大模型的性能。

**练习 4：** 分析"涌现能力"的经济含义：为什么公司愿意花数千万美元训练一个可能"涌现"新能力的模型？
