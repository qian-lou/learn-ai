# Datasets 与数据处理 / HuggingFace Datasets

## 1. 背景（Background）
> HuggingFace Datasets 提供统一的数据集加载和处理接口，支持流式加载超大数据集。

## 2-3. 知识点与内容
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("imdb")
df = dataset["train"].to_pandas()

# 数据预处理
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# 流式加载大数据集（不会 OOM）
stream = load_dataset("c4", "en", split="train", streaming=True)
```

## 4-6. 推理/例题/习题
**练习：** 加载一个 NLP 数据集，完成分词预处理，构建 DataLoader。
