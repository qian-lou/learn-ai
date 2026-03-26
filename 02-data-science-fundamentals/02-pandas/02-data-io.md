# 数据读写
# Data I/O

## 1. 背景（Background）

> **为什么要学这个？**
>
> 数据读写是数据分析的第一步。Pandas 支持 CSV、JSON、Excel、Parquet、SQL 等格式的读写。大模型训练数据通常存储为 JSONL 或 Parquet 格式。

## 2. 知识点（Key Concepts）

| 格式 | 读取 | 写入 | 适用场景 |
|------|------|------|---------|
| CSV | `read_csv` | `to_csv` | 通用 |
| JSON/JSONL | `read_json` | `to_json` | API/LLM 数据 |
| Parquet | `read_parquet` | `to_parquet` | 大规模数据 |
| Excel | `read_excel` | `to_excel` | 报表 |
| SQL | `read_sql` | `to_sql` | 数据库 |

## 3. 内容（Content）

```python
import pandas as pd

# ============================================================
# CSV（最常用）
# ============================================================
df = pd.read_csv("data.csv", encoding="utf-8")
df = pd.read_csv("data.csv", usecols=["name", "age"])  # 只读指定列
df = pd.read_csv("large.csv", chunksize=10000)  # 分块读取大文件

df.to_csv("output.csv", index=False, encoding="utf-8")

# ============================================================
# JSON / JSONL（LLM 训练数据常用格式）
# ============================================================
# JSONL: 每行一个 JSON 对象
df = pd.read_json("data.jsonl", lines=True)
df.to_json("output.jsonl", lines=True, orient="records", force_ascii=False)

# ============================================================
# Parquet（列式存储，压缩比高，读取快）
# ============================================================
# pip install pyarrow
df = pd.read_parquet("data.parquet")
df.to_parquet("output.parquet", engine="pyarrow")

# Parquet vs CSV:
# 1GB CSV → ~200MB Parquet（压缩 5x）
# 读取速度: Parquet 快 10x

# ============================================================
# SQL 数据库
# ============================================================
# from sqlalchemy import create_engine
# engine = create_engine("mysql+pymysql://user:pass@host:3306/db")
# df = pd.read_sql("SELECT * FROM users LIMIT 1000", engine)
# df.to_sql("results", engine, if_exists="replace", index=False)
```

## 4. 详细推理（Deep Dive）

```
大模型数据格式选择:
  SFT 数据: JSONL（Alpaca/ShareGPT 格式）
  Embedding: Parquet（存储向量高效）
  评估结果: CSV（简单直观）
  大规模语料: Parquet + 分片（多文件）
```

## 5-6. 例题/习题

**练习 1：** 读取一个 CSV 文件，清洗后保存为 JSONL 格式。

**练习 2：** 用 `chunksize` 分块处理一个 5GB 的 CSV 文件，统计行数和缺失值。
