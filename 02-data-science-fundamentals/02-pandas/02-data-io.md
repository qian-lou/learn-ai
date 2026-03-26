# 数据读写（CSV/JSON/SQL）
# Data I/O (CSV/JSON/SQL)

## 1. 背景（Background）

> 大模型训练数据通常以 CSV、JSON、Parquet 等格式存储。Pandas 提供统一的 I/O 接口。对 Java 工程师来说，这替代了 Jackson/Gson 的 JSON 处理和 JDBC 的数据库读取。

## 2-3. 知识点与内容

```python
import pandas as pd

# CSV 读写 / CSV I/O
df = pd.read_csv("data.csv", encoding="utf-8")
df.to_csv("output.csv", index=False)

# JSON 读写 / JSON I/O（NLP 数据集常用 JSONL 格式）
df = pd.read_json("data.json")
df = pd.read_json("data.jsonl", lines=True)  # JSONL：每行一个 JSON
df.to_json("output.jsonl", orient="records", lines=True)

# Parquet 读写（大数据集推荐格式，比 CSV 快 10x）
df = pd.read_parquet("data.parquet")
df.to_parquet("output.parquet")

# SQL 读取 / SQL I/O
# import sqlite3
# conn = sqlite3.connect("database.db")
# df = pd.read_sql("SELECT * FROM users WHERE age > 20", conn)

# HuggingFace 数据集 → Pandas
# from datasets import load_dataset
# ds = load_dataset("imdb")
# df = ds["train"].to_pandas()
```

## 4-6. 推理/例题/习题

**练习 1：** 将一个 CSV 文件转换为 Parquet 格式并对比文件大小。
**练习 2：** 读取一个 JSONL 格式的 NLP 数据集，统计各标签的分布。
