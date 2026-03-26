# 激活函数 / Activation Functions

## 1. 背景（Background）
> 激活函数引入非线性。GELU 已取代 ReLU 成为大模型的主流激活函数。

## 2-3. 知识点与内容
```python
import torch.nn.functional as F

x = torch.linspace(-5, 5, 100)

relu = F.relu(x)       # max(0, x) — 经典
gelu = F.gelu(x)       # x*Φ(x) — GPT/BERT 使用
silu = F.silu(x)       # x*sigmoid(x) — LLaMA 使用

# Softmax: 多分类输出，将 logits 转概率
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)  # [0.659, 0.243, 0.099]
```

## 4. 详细推理
- ReLU 解决了 Sigmoid 的梯度消失问题，但有"神经元死亡"问题
- GELU 更平滑，允许少量负值通过，大模型效果更好
- Softmax 的温度参数 T：`softmax(x/T)`，T 越大分布越均匀

## 5-6. 例题/习题
**练习：** 可视化 6 种激活函数及其导数。
