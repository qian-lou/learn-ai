# 概率与统计（分布/贝叶斯）
# Probability and Statistics

## 1. 背景（Background）

> 机器学习本质上是在做概率推断。语言模型就是在估计 P(next_word | context)。理解概率分布、贝叶斯定理、最大似然估计，才能理解模型的损失函数为什么这样定义。

## 2-3. 知识点与内容

```python
import numpy as np
from scipy import stats

# 常见分布 / Common distributions
# 正态分布（模型权重初始化）
samples = np.random.normal(mean=0, scale=1, size=10000)

# 伯努利分布（二分类）
# 分类分布（多分类 Softmax 输出）

# 贝叶斯定理 / Bayes' theorem
# P(A|B) = P(B|A) * P(A) / P(B)
# 在 NLP 中：P(label|text) ∝ P(text|label) * P(label)

# 信息论 / Information theory
# 交叉熵损失（分类任务的标准损失函数）
# Cross-entropy loss: H(p, q) = -Σ p(x) * log(q(x))
def cross_entropy(y_true, y_pred):
    """交叉熵损失 / Cross-entropy loss.
    
    Time: O(N)  Space: O(1)
    """
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# KL 散度（衡量两个分布的差异，RLHF 中使用）
# KL divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
```

## 4-6. 推理/例题/习题

**核心理解：** 语言模型训练 = 最大化 Σ log P(token_t | token_1...token_{t-1})

**练习：** 手动计算一个简单的交叉熵损失值，并用 PyTorch 的 `nn.CrossEntropyLoss` 验证。
