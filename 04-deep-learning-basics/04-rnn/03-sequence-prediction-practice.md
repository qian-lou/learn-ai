# 序列预测实战 / Sequence Prediction Practice

## 1. 背景（Background）
> 用 LSTM 做文本生成——预测下一个 token。这就是 GPT 的前身，GPT 只是用 Transformer 替换了 LSTM。

## 2-3. 知识点与内容
```python
# 语言模型的基本流程：
# 1. 文本 → token 序列（分词）
# 2. 输入序列 → LSTM/Transformer → 预测下一个 token
# 3. 采样策略：贪心(Greedy) / Top-K / Top-P(Nucleus)
# GPT = Transformer Decoder 替换了 LSTM，其他流程一致
```

## 4-6. 推理/例题/习题
**练习：** 用 LSTM 实现字符级语言模型，训练后生成文本。
