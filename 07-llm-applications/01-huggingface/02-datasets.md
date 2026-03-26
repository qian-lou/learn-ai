# Datasets 与数据处理 / HuggingFace Datasets

## 1. 背景（Background）

> **为什么要学这个？**
>
> HuggingFace Datasets 提供了统一的数据集加载和处理接口，支持从 10 万+ 公开数据集中一行代码加载数据。它的 Arrow 后端支持**零拷贝读取**和**流式处理超大数据集**，不会 OOM。
>
> 对于 Java 工程师来说，Datasets 就像 **Spring Data + JPA**——统一的数据访问层，屏蔽底层存储差异。

## 2. 知识点（Key Concepts）

| 功能 | API | 说明 |
|------|-----|------|
| 加载数据集 | `load_dataset` | 从 Hub 或本地加载 |
| 数据预处理 | `.map()` | 批量预处理（向量化） |
| 过滤 | `.filter()` | 条件过滤 |
| 流式加载 | `streaming=True` | 不下载全部数据 |
| 格式转换 | `.set_format("torch")` | 转为 PyTorch 格式 |

## 3. 内容（Content）

### 3.1 加载数据集

```python
from datasets import load_dataset

# ============================================================
# 从 Hub 加载 / Load from Hub
# ============================================================
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

print(dataset["train"][0])
# {'text': 'I love this movie...', 'label': 1}

# 加载特定分片
train = load_dataset("imdb", split="train[:1000]")  # 前 1000 条

# ============================================================
# 加载本地数据 / Load local data
# ============================================================
# JSON
dataset = load_dataset("json", data_files="data.jsonl")

# CSV
dataset = load_dataset("csv", data_files="data.csv")

# 文件夹（按子文件夹作为标签）
dataset = load_dataset("imagefolder", data_dir="./images")
```

### 3.2 数据预处理

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ============================================================
# 用 .map() 批量预处理 / Batch preprocessing with .map()
# ============================================================
def tokenize_function(examples):
    """批量分词 / Batch tokenization."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

# batched=True 开启批量处理（快 10-100x）
tokenized = dataset.map(tokenize_function, batched=True, num_proc=4)
# num_proc=4 开启多进程并行

# 删除不需要的列
tokenized = tokenized.remove_columns(["text"])
# 设置格式为 PyTorch tensor
tokenized.set_format("torch")
```

### 3.3 流式加载超大数据集

```python
# ============================================================
# 流式加载（不会 OOM）/ Streaming (no OOM)
# ============================================================
stream = load_dataset("c4", "en", split="train", streaming=True)

# 流式数据集是迭代器，按需加载
for i, example in enumerate(stream):
    print(example["text"][:100])
    if i >= 5:
        break

# 流式处理 + 预处理
processed = stream.map(tokenize_function, batched=True, batch_size=1000)
```

### 3.4 自定义数据集

```python
from datasets import Dataset

# ============================================================
# 从 Python 对象创建 / Create from Python objects
# ============================================================
data = {
    "text": ["Hello world", "AI is great", "Python rocks"],
    "label": [0, 1, 1],
}
dataset = Dataset.from_dict(data)

# 从 Pandas DataFrame 创建
import pandas as pd
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 Datasets 比 Pandas 快？

```
Datasets 基于 Apache Arrow 格式：
  - 列式存储 → 读取特定列极快
  - 零拷贝 → 内存映射文件，不需要加载到内存
  - 支持惰性加载 → 处理 TB 级数据集

Pandas: 1000 万行 × 768 维 → 需要 ~30GB RAM
Datasets: 同样数据 → 只需 ~1GB RAM（内存映射）
```

## 5. 例题（Worked Examples）

```python
# 完整流程：加载 → 预处理 → DataLoader
from torch.utils.data import DataLoader

dataset = load_dataset("imdb", split="train[:5000]")
tokenized = dataset.map(tokenize_function, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataloader = DataLoader(tokenized, batch_size=16, shuffle=True)

for batch in dataloader:
    print(batch["input_ids"].shape)  # [16, 256]
    break
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 加载 IMDB 数据集，完成分词预处理，构建 DataLoader。

**练习 2：** 用流式方式加载 C4 数据集，统计前 10000 条数据的平均长度。

### 进阶题

**练习 3：** 创建一个自定义的指令微调数据集（Alpaca 格式），上传到 HuggingFace Hub。

**练习 4：** 对比 `datasets.map(batched=True)` 和逐条处理的速度差异。
