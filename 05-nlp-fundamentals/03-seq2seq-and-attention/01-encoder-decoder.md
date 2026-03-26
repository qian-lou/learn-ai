# 编码器-解码器架构 / Encoder-Decoder Architecture

## 1. 背景（Background）
> Encoder-Decoder 是 Transformer 的基础结构。Encoder 理解输入，Decoder 生成输出。

## 2-3. 知识点与内容
```python
# Encoder: 输入序列 → 上下文表示 (BERT是纯Encoder)
# Decoder: 上下文 → 输出序列 (GPT是纯Decoder)
# Encoder-Decoder: T5, BART (翻译/摘要)
# 问题：固定长度上下文向量 → 信息瓶颈 → 注意力机制解决
```

## 4-6. 推理/例题/习题
**练习：** 用 LSTM 实现简单的 Seq2Seq 模型。
