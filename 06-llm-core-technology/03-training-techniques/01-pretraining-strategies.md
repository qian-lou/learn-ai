# 预训练策略 / Pretraining Strategies

## 1. 背景（Background）
> 预训练是大模型获取知识的阶段。MLM(BERT)、CLM(GPT)、Span Corruption(T5) 是三种主要策略。

## 2-3. 知识点与内容
```
三种预训练策略对比：
MLM (Masked Language Model - BERT):
  输入: "I [MASK] NLP"  →  预测: "love"
  特点: 双向上下文，适合理解任务

CLM (Causal Language Model - GPT):
  输入: "I love"  →  预测: "NLP"
  特点: 单向（自回归），适合生成任务

Span Corruption (T5):
  输入: "I <X> NLP is <Y>"  →  输出: "<X> love <Y> amazing"
  特点: 支持多种任务格式
```

## 4-6. 推理/例题/习题
**练习：** 理解 CLM 的 Causal Mask 为什么要遮住未来的 token。
