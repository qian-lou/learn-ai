# 概率与统计
# Probability and Statistics

## 1. 背景（Background）

> **为什么要学这个？**
>
> 概率论是机器学习的**理论根基**。分类器输出概率分布，损失函数基于最大似然估计，贝叶斯推断是不确定性量化的基础。Softmax、交叉熵、KL 散度——这些核心概念都来自概率论。

## 2. 知识点（Key Concepts）

| 概念 | ML 应用 |
|------|---------|
| 概率分布 | Softmax 输出 |
| 最大似然估计 (MLE) | 训练损失函数 |
| 贝叶斯定理 | 后验推断 |
| 信息熵 / 交叉熵 | 损失函数 |
| KL 散度 | 知识蒸馏, VAE |

## 3. 内容（Content）

### 3.1 概率分布

```python
import numpy as np

# ============================================================
# 常见分布 / Common distributions
# ============================================================
# 均匀分布
uniform = np.random.uniform(0, 1, size=1000)

# 正态分布（权重初始化常用）
normal = np.random.normal(mean=0, std=0.02, size=1000)

# 伯努利分布（Dropout）
dropout_mask = np.random.binomial(1, p=0.9, size=(768,))  # 90% 保留

# ============================================================
# Softmax（将 logits 转为概率分布）
# ============================================================
def softmax(logits):
    """Softmax 函数 / Softmax function.
    Time: O(N)  Space: O(N)
    """
    exp_logits = np.exp(logits - np.max(logits))  # 减 max 防溢出
    return exp_logits / exp_logits.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)  # [0.659, 0.242, 0.099] — 概率之和为 1
```

### 3.2 信息论

```python
# ============================================================
# 信息熵 / Entropy
# H(p) = -Σ p(x) log p(x)
# ============================================================
def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

# 均匀分布熵最大
print(entropy(np.array([0.25, 0.25, 0.25, 0.25])))  # 1.386
# 确定分布熵最小
print(entropy(np.array([1.0, 0.0, 0.0, 0.0])))       # 0.0

# ============================================================
# 交叉熵（分类损失函数的本质）
# H(p, q) = -Σ p(x) log q(x)
# ============================================================
def cross_entropy(true_probs, pred_probs):
    return -np.sum(true_probs * np.log(pred_probs + 1e-10))

# 分类任务: true = one-hot [0, 1, 0]
true = np.array([0, 1, 0])
pred_good = np.array([0.1, 0.8, 0.1])
pred_bad = np.array([0.3, 0.4, 0.3])
print(f"好预测 CE: {cross_entropy(true, pred_good):.4f}")  # 0.2231
print(f"差预测 CE: {cross_entropy(true, pred_bad):.4f}")    # 0.9163

# ============================================================
# KL 散度（知识蒸馏的损失函数）
# KL(p||q) = Σ p(x) log(p(x)/q(x))
# ============================================================
def kl_divergence(p, q):
    return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
```

## 4. 详细推理（Deep Dive）

```
为什么用交叉熵做分类损失？

CE = -log(p_correct)

预测正确类别概率 0.9 → CE = 0.105（小损失 ✅）
预测正确类别概率 0.1 → CE = 2.303（大损失 ❌）

本质：最小化交叉熵 = 最大化正确类别的概率 = 最大似然估计
```

## 5-6. 例题/习题

**练习 1：** 实现 Softmax + 交叉熵损失（注意数值稳定性）。

**练习 2：** 计算两个概率分布的 KL 散度，验证 KL(p||q) ≠ KL(q||p)。
