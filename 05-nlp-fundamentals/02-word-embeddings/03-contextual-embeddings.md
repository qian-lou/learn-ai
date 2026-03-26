# 上下文嵌入 / Contextual Embeddings

## 1. 背景（Background）
> 传统词嵌入给每个词固定向量。上下文嵌入（ELMo → BERT）动态生成向量——"bank" 在不同语境有不同表示。这是通往大模型的关键转折点。

## 2-3. 知识点与内容
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("The bank of the river", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # Shape: [1, seq_len, 768]
```

## 4-6. 推理/例题/习题
**练习：** 提取 "bank" 在不同语境下的 BERT 向量，计算余弦相似度。
