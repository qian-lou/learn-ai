# Transformer 完整架构 / Complete Transformer Architecture

## 1. 背景（Background）
> Transformer 是现代 AI 的基石核心。理解完整架构是理解所有大模型的关键。

## 2-3. 知识点与内容
```
Transformer 架构图：
┌─ Encoder ─────────────────┐  ┌─ Decoder ─────────────────┐
│  Input Embedding          │  │  Output Embedding         │
│  + Positional Encoding    │  │  + Positional Encoding    │
│  ┌───────────────────┐   │  │  ┌───────────────────┐   │
│  │ Multi-Head Attn   │×N │  │  │ Masked MH Attn    │×N │
│  │ Add & LayerNorm   │   │  │  │ Add & LayerNorm   │   │
│  │ Feed Forward      │   │  │  │ Cross Attention   │   │
│  │ Add & LayerNorm   │   │  │  │ Add & LayerNorm   │   │
│  └───────────────────┘   │  │  │ Feed Forward      │   │
└──────────────────────────┘  │  │ Add & LayerNorm   │   │
                               │  └───────────────────┘   │
                               │  Linear + Softmax        │
                               └──────────────────────────┘

BERT = 只用 Encoder（理解输入）
GPT  = 只用 Decoder（生成输出）
T5   = 完整 Encoder-Decoder
```

## 4-6. 推理/例题/习题
**练习：** 对比 BERT/GPT/T5 的架构差异和适用场景。
