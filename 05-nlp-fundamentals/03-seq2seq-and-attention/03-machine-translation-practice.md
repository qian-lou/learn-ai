# 机器翻译实战 / Machine Translation Practice

## 1. 背景（Background）
> 机器翻译是 Transformer 诞生的背景。通过实战理解 Seq2Seq + Attention 完整流程。

## 2-3. 知识点与内容
```python
from transformers import pipeline
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
result = translator("Hello, how are you?")
```

## 4-6. 推理/例题/习题
**练习：** 使用预训练翻译模型，对比不同模型的翻译质量。
