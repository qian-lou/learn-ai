# 数值精度与稳定性
# Numerical Precision and Stability

## 1. 背景（Background）

> **为什么要学这个？**
>
> 前面几篇教了用 `einsum` 算 Attention 的 QKᵀ、用 `np.exp/np.where` 做向量化——你已经完全有能力手写一个 softmax。但**照直写的 `np.exp(x) / np.exp(x).sum()` 在真实 logits 上会溢出成 `NaN`**，整个前向传播直接报废。这是大模型里最经典的数值坑：`float16` 只能表示到约 65504，`np.exp(1000)` 直接变 `inf`。本篇讲透浮点位宽、softmax 减最大值、logsumexp 三件事——它们是后续 Attention、交叉熵能跑起来的**地基**。

## 2. 知识点（Key Concepts）

| 概念 | 问题 | 稳定写法 |
|------|------|---------|
| 浮点位宽 | `float16` 上限 ~65504，易溢出 | 训练用 `bfloat16`（动态范围同 `float32`） |
| softmax | `exp(大数)` → `inf` → `nan` | 先减最大值 `x - x.max()` |
| logsumexp | `log(sum(exp(x)))` 中间 `exp` 溢出 | `m + log(sum(exp(x - m)))` |
| 成对加法 | 两数 `exp` 后相加溢出 | `np.logaddexp(a, b)` |

## 3. 内容（Content）

```python
import numpy as np

# ============================================================
# 1. 浮点位宽与动态范围 / Float width & dynamic range
# ============================================================
# float32：1 符号 + 8 指数 + 23 尾数，max ≈ 3.4e38，精度高
# float16：1 符号 + 5 指数 + 10 尾数，max ≈ 65504，动态范围窄
# bfloat16：1 符号 + 8 指数 + 7 尾数，max ≈ 3.4e38（指数位同 float32）
print(np.finfo(np.float32).max)  # 3.4028235e+38
print(np.finfo(np.float16).max)  # 65500.0（约 65504）

# ❌ float16 溢出反例：50000 * 2 超过上限 → inf
x = np.float16(50000)
print(x * np.float16(2))  # inf（RuntimeWarning: overflow）

# 关键结论：bfloat16 指数位和 float32 一样多（8 位），
# 所以「不易溢出」；代价是尾数只有 7 位，「精度更粗」。
# 训练时宁可精度粗一点，也不能中途 inf/nan——这就是
# 现代大模型训练首选 bfloat16 而非 float16 的原因。

# ============================================================
# 2. 朴素 softmax 的溢出 / Naive softmax overflow
# ============================================================
logits = np.array([1000.0, 1001.0, 1002.0])  # 真实 logits 常达几百上千

# ❌ 朴素写法：exp(1000) 在 float64 也溢出
naive = np.exp(logits) / np.exp(logits).sum()
print(naive)  # [nan nan nan]（exp 溢出 inf，inf/inf = nan）

# ✅ 减最大值：数学等价，数值稳定
# Time: O(N), Space: O(N)
shifted = logits - logits.max()          # [-2, -1, 0]，exp 不溢出
stable = np.exp(shifted) / np.exp(shifted).sum()
print(stable)  # [0.09003057 0.24472847 0.66524096]

# ============================================================
# 3. logsumexp 稳定实现 / Stable logsumexp
# ============================================================
# 需求：算 log(sum(exp(x)))，直接算中间会溢出
# ❌ 朴素：sum(exp(1000+)) = inf → log(inf) = inf
print(np.log(np.exp(logits).sum()))  # inf

# ✅ 减最大值再加回：LSE(x) = m + log(sum(exp(x - m)))，m = max(x)
# Time: O(N), Space: O(N)
def logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return m + np.log(np.exp(x - m).sum())

print(logsumexp(logits))  # 1002.4076059644444

# ✅ 两数版本用内建 np.logaddexp（内部已做减最大值）
print(np.logaddexp(1000.0, 1001.0))  # 1001.3132616875182
```

## 4. 详细推理（Deep Dive）

```
一、为什么减最大值不改变 softmax 结果（等价证明）
  softmax(x)_i = e^{x_i} / Σ_j e^{x_j}
  分子分母同乘常数 e^{-m}（m = max(x)）：
    = e^{x_i} · e^{-m} / (Σ_j e^{x_j} · e^{-m})
    = e^{x_i - m} / Σ_j e^{x_j - m}
  → 结果完全不变，但每个 e^{x_i - m} 的指数 ≤ 0，
    最大也只是 e^0 = 1，永不溢出。（下溢成 0 无害。）

二、logsumexp 恒等式
  log Σ e^{x_i}
    = log Σ e^{x_i - m + m}
    = log( e^m · Σ e^{x_i - m} )
    = m + log Σ e^{x_i - m}
  同样把「大指数」搬到 log 外面，括号内指数 ≤ 0，稳定。

三、float16 vs bfloat16（同为 16 位，分配不同）
  float16   [S|EEEEE|MMMMMMMMMM]  指数 5 位 → 范围窄，训练易 inf
  bfloat16  [S|EEEEEEEE|MMMMMMM]  指数 8 位 → 范围同 float32，训练首选
  记忆：bf16 = 「float32 砍掉后 16 位尾数」，范围不变、精度减半。
```

## 5. 例题（Worked Examples）

### 例题 1：从 nan 到稳定的 softmax / From NaN to a Stable Softmax

本例对比朴素 softmax 与减最大值版本，直观看到「同一批 logits，一个全是 `nan`，一个正常」，并验证减最大值前后结果一致。

```python
import numpy as np

def softmax_naive(x: np.ndarray) -> np.ndarray:
    """朴素 softmax（真实 logits 会溢出）/ Naive softmax (overflows)."""
    # Time: O(N), Space: O(N)
    e = np.exp(x)
    return e / e.sum()

def softmax_stable(x: np.ndarray) -> np.ndarray:
    """稳定 softmax：减最大值 / Stable softmax: subtract max."""
    # Time: O(N), Space: O(N)
    e = np.exp(x - x.max())
    return e / e.sum()

# 小 logits：两者一致；大 logits：只有稳定版活着
small = np.array([2.0, 1.0, 0.0])
large = np.array([1000.0, 1001.0, 1002.0])

print("小 logits 朴素:", softmax_naive(small))
# [0.66524096 0.24472847 0.09003057]
print("小 logits 稳定:", softmax_stable(small))
# [0.66524096 0.24472847 0.09003057]（与朴素一致，证明等价）

print("大 logits 朴素:", softmax_naive(large))  # [nan nan nan]
print("大 logits 稳定:", softmax_stable(large))
# [0.09003057 0.24472847 0.66524096]（正常）
```

### 例题 2：数值稳定的交叉熵 / Numerically-Stable Cross-Entropy

交叉熵 = `-log(softmax(logits)[label])`。若先算 softmax 再取 log，softmax 里的极小概率 log 后会得 `-inf`。正确做法是用 logsumexp 直接合并：`loss = logsumexp(logits) - logits[label]`。

```python
import numpy as np

def cross_entropy(logits: np.ndarray, label: int) -> float:
    """数值稳定的交叉熵 / Stable cross-entropy via logsumexp.

    利用 -log(softmax(x)[y]) = logsumexp(x) - x[y]，
    避开先 exp 再 log 的双重溢出/下溢。
    Time: O(N), Space: O(N)
    """
    m = logits.max()
    lse = m + np.log(np.exp(logits - m).sum())  # logsumexp
    return float(lse - logits[label])

logits = np.array([1000.0, 1001.0, 1002.0])
# 正确标签是第 2 类（对应稳定 softmax 概率 0.665）
print(cross_entropy(logits, label=2))  # 0.4076059644...
# 校验：-log(0.66524096) ≈ 0.4076，与上式一致
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：不使用任何 for 循环，判断给定 `float16` 数组里哪些元素在乘以 100 后会溢出为 `inf`，返回布尔掩码。
*参考答案*：
```python
import numpy as np
# Time: O(N), Space: O(N)
x = np.array([100, 500, 700, 1000], dtype=np.float16)
overflow = np.isinf(x.astype(np.float16) * np.float16(100))
print(overflow)  # [False False  True  True]（700*100=70000 > 65504）
```

### 进阶题
**练习 2**：实现一个**沿指定轴**的稳定 softmax（对 `[B, N]` 的每一行独立归一化），要求全向量化、无 Python 循环，并用 `keepdims=True` 让减最大值正确广播。
*参考答案*：
```python
import numpy as np

def softmax_axis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """沿 axis 的稳定 softmax / Row-wise stable softmax.

    Time: O(N), Space: O(N)  —— N 为元素总数
    """
    # keepdims 保留被压缩的轴，才能广播回原 shape 做减法
    x_max = x.max(axis=axis, keepdims=True)          # Shape: [B, 1]
    e = np.exp(x - x_max)                             # 每行最大值变 0，不溢出
    return e / e.sum(axis=axis, keepdims=True)        # Shape: [B, N]

logits = np.array([[1000.0, 1001.0, 1002.0],
                   [0.0, 0.0, 0.0]])                  # Shape: [2, 3]
probs = softmax_axis(logits, axis=1)
print(probs.sum(axis=1))  # [1. 1.]（每行归一化）
print(probs[0])           # [0.09003057 0.24472847 0.66524096]
```

---

> **一句话串联**：减最大值的 softmax、logsumexp 的交叉熵、`keepdims` 的行归一化，正是阶段四手写 Attention 的 scaled-dot-product、阶段五的分类损失、阶段六推理时 `bfloat16` 不溢出的**同一套地基**——这里踩过的 `nan`，就是那里省下的 debug 之夜。
