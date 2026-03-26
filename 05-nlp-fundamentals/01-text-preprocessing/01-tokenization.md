# 分词 / Tokenization

## 1. 背景（Background）
> 分词是 NLP 第一步。大模型使用 BPE/WordPiece 子词分词器，理解分词对理解模型输入至关重要。

## 2-3. 知识点与内容
```python
# 子词分词（现代大模型标配）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokens = tokenizer.tokenize("自然语言处理很有趣")
ids = tokenizer.encode("自然语言处理很有趣")
decoded = tokenizer.decode(ids)

# BPE (GPT) / WordPiece (BERT) / SentencePiece (T5/LLaMA)
```

## 4-6. 推理/例题/习题
**练习：** 对比不同分词器的 vocab_size 和分词结果。
