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
normal = np.random.normal(loc=0.0, scale=0.02, size=1000)  # loc=均值 mean, scale=标准差 std

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
    # 数值稳定：分子分母同加 epsilon 防止 log(0) 与除零
    # Numerical stability: add epsilon to both numerator & denominator
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
```

## 4. 详细推理（Deep Dive）

```
为什么用交叉熵做分类损失？

CE = -log(p_correct)

预测正确类别概率 0.9 → CE = 0.105（小损失 ✅）
预测正确类别概率 0.1 → CE = 2.303（大损失 ❌）

本质：最小化交叉熵 = 最大化正确类别的概率 = 最大似然估计
```

## 5. 例题（Worked Examples）

### 例题 1：朴素贝叶斯分类器的概率推导与计算 / Naive Bayes Probability Derivation

在垃圾邮件分类中，我们需要计算 $P(\text{Spam} \mid \text{Words}) \propto P(\text{Spam}) \prod P(\text{Word}_i \mid \text{Spam})$。本例题演示如何计算这个条件概率。

```python
import numpy as np

# 假设先验概率 / Prior probabilities
p_spam = 0.4
p_ham = 0.6

# 两个单词在垃圾/正常邮件中出现的似然概率 / Likelihoods of words
# Word 1: 'money', Word 2: 'meeting'
# Shape: [2, 2] -> [Spam/Ham, Word1/Word2]
p_word_given_class = np.array([
    [0.8, 0.1],  # Spam: P('money'|Spam)=0.8, P('meeting'|Spam)=0.1
    [0.05, 0.7]  # Ham:  P('money'|Ham)=0.05, P('meeting'|Ham)=0.7
])

# 测试样本含有 'money' 但不含 'meeting' / Test sample: contains 'money', no 'meeting'
# 计算非归一化后验概率 / Compute unnormalized posteriors
# Time: O(W), Space: O(1)
post_spam = p_spam * p_word_given_class[0, 0] * (1 - p_word_given_class[0, 1])
post_ham = p_ham * p_word_given_class[1, 0] * (1 - p_word_given_class[1, 1])

# 归一化 / Normalize
total = post_spam + post_ham
prob_spam = post_spam / total

print(f"该邮件属于垃圾邮件的概率 / Prob(Spam | money, not meeting): {prob_spam:.4f}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：编写代码，计算一组观测数据的样本均值、样本标准差及偏度（Skewness）。
*参考答案*：
```python
import numpy as np
from scipy.stats import skew
# Time: O(N), Space: O(1)
data = np.array([1, 2, 2, 3, 4, 5, 10])  # 右偏数据
print(f"均值: {np.mean(data)}")
print(f"标准差: {np.std(data)}")
print(f"偏度: {skew(data)}")
```

### 进阶题
**练习 2**：在大模型（如 GPT）生成文本采样中，我们需要理解 Softmax 与温度（Temperature）的影响。给定对数概率向量 `logits`，利用不同的 Temperature 参数对其进行缩放，并计算对应的概率分布。分析 $T \to 0$ 和 $T \to \infty$ 时的概率特征。
*参考答案*：
```python
import numpy as np

def softmax(logits: np.ndarray, temp: float) -> np.ndarray:
    """带温度的 Softmax / Softmax with temperature.
    
    Time: O(K), Space: O(K)
    """
    scaled = logits / temp
    # 防止溢出的指数变换 / Stabilize exponents.
    e_x = np.exp(scaled - np.max(scaled))
    return e_x / e_x.sum()

logits = np.array([2.0, 1.0, 0.1])
print(f"T=0.5 (确定性倾向): {softmax(logits, 0.5)}")
print(f"T=1.0 (常规采样):   {softmax(logits, 1.0)}")
print(f"T=2.0 (随机性倾向): {softmax(logits, 2.0)}")
# 结论：T 越小概率越向最大值集聚，T 趋近于 0 变为 one-hot；T 越大概率分布越平缓，趋近于均匀分布。
```