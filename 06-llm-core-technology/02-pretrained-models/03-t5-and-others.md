# T5 及其他模型 / T5 and Other Models

## 1. 背景（Background）
> T5 统一了 NLP 任务为"文本到文本"格式。还有 LLaMA、Qwen、ChatGLM 等开源大模型。

## 2-3. 知识点与内容
```
模型对比：
| 模型   | 架构        | 参数量    | 特点               |
|--------|------------|----------|-------------------|
| BERT   | Encoder    | 110M-340M | 理解任务（分类/NER） |
| GPT-3  | Decoder    | 175B      | 生成任务（对话/写作） |
| T5     | Enc-Dec    | 220M-11B  | 统一格式，通用性强    |
| LLaMA  | Decoder    | 7B-70B    | 开源标杆            |
| Qwen   | Decoder    | 0.5B-72B  | 中文优化            |
```

## 4-6. 推理/例题/习题
**练习：** 在 HuggingFace 上对比不同模型在同一任务上的表现。
