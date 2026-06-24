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

# ------------------------------------------------------------
# read_csv 的四个关键参数（决定正确性与内存/速度）
# Four key read_csv params (correctness + memory/speed)
# ------------------------------------------------------------

# ① usecols：只解析需要的列，跳过其余——少读列 = 少内存、少 CPU
#    usecols: parse only needed columns; fewer columns = less memory & CPU
df = pd.read_csv("wide.csv", usecols=["user_id", "label"])

# ② dtype：显式指定列类型，避免 pandas 逐列推断（推断既慢又可能误判）
#    dtype: pin column types; skips per-column inference (slow & error-prone)
df = pd.read_csv(
    "data.csv",
    dtype={
        "user_id": "int32",        # 默认 int64，int32 省一半内存
        "label": "category",       # 低基数字符串列用 category，内存骤降（见 §4.3）
    },
)
# 注意：含缺失值的整数列无法用 int（NaN 是浮点），用可空整型 "Int64"（大写 I）
# Note: integer columns with NaN need nullable "Int64" (capital I), not int

# ③ parse_dates：读取时直接把字符串列解析为 datetime64，省去事后 to_datetime
#    parse_dates: parse string columns into datetime64 at read time
df = pd.read_csv(
    "events.csv",
    parse_dates=["created_at"],    # 单列；多列传列表
    # date_format="%Y-%m-%d %H:%M:%S",  # 指定格式比自动嗅探快得多 / explicit format is faster
)

# ④ chunksize：返回一个迭代器，每次产出 N 行的 DataFrame，用于流式处理超大文件
#    chunksize: returns an iterator yielding N-row chunks; stream huge files
total = 0
for chunk in pd.read_csv("huge.csv", chunksize=100_000):   # 每块 10 万行
    total += chunk["amount"].sum()        # 逐块聚合，峰值内存只占一块 / O(chunk) memory
print(total)

# 其他高频参数 / Other common params:
# sep="\t"（TSV）, header=None, names=[...], na_values=["NA","-"],
# nrows=1000（只读前 N 行做探查）, skiprows=..., compression="gzip"（直读 .csv.gz）

# ============================================================
# JSON / JSONL（LLM 训练数据常用格式）
# ============================================================
# JSONL: 每行一个 JSON 对象
df = pd.read_json("data.jsonl", lines=True)
df.to_json("output.jsonl", lines=True, orient="records", force_ascii=False)

# ============================================================
# Parquet（列式存储，压缩比高，读取快）—— 2026 大数据/LLM 首选
# Parquet (columnar, high compression, fast) — the default for big-data/LLM
# ============================================================
# pip install pyarrow   （或 uv add pyarrow）
df = pd.read_parquet("data.parquet")
df.to_parquet("output.parquet", engine="pyarrow", compression="zstd")  # zstd 压缩比/速度俱佳

# 列裁剪：只读需要的列——列存格式下这是物理跳过，不读其余列的字节
# Column pruning: physically skips other columns' bytes (true columnar benefit)
df = pd.read_parquet("data.parquet", columns=["user_id", "label"])

# 谓词下推：把过滤条件下沉到读取层，跳过不满足的 row group，少读磁盘（见 §4.2）
# Predicate pushdown: filter at the read layer, skipping row groups (see §4.2)
df = pd.read_parquet(
    "events.parquet",
    filters=[("year", "==", 2026), ("label", "in", ["a", "b"])],
)

# ------------------------------------------------------------
# Feather / Arrow IPC：内存映射、零拷贝、极速 —— 适合"进程间/管道中转"
# Feather (Arrow IPC): mmap, zero-copy, fastest — for intermediate handoff
# ------------------------------------------------------------
df.to_feather("cache.feather")           # 写入近乎原始内存布局 / near-raw memory layout
df = pd.read_feather("cache.feather")     # 读取几乎零解析开销 / almost no parse cost

# 三种格式取舍 / Choosing among the three:
#   CSV     : 人类可读、跨工具通用，但无类型、无压缩、解析慢（见 §4.1）
#   Parquet : 列存 + 压缩 + 保留 dtype，适合"落盘归档/大规模训练语料"，跨语言
#   Feather : 速度最快、保留 dtype，但压缩弱、偏临时缓存，不适合长期归档

# ============================================================
# Excel（报表场景；底层走 openpyxl，按 cell 解析，最慢，勿用于大数据）
# ============================================================
df = pd.read_excel("report.xlsx", sheet_name="Sheet1", usecols="A:C")
df.to_excel("out.xlsx", index=False, sheet_name="结果")

# ============================================================
# SQL 数据库（pandas 直连关系库；底层 SQLAlchemy ≈ Java 的 JDBC）
# SQL databases (pandas over SQLAlchemy; the analog of Java JDBC)
# ============================================================
# from sqlalchemy import create_engine
# engine = create_engine("mysql+pymysql://user:pass@host:3306/db")  # ≈ DataSource/连接池
# df = pd.read_sql("SELECT * FROM users WHERE age > 18", engine)      # ≈ JdbcTemplate.query
# 大表分块读取：避免一次性把全表拉进内存 / chunk a huge table
# for chunk in pd.read_sql("SELECT * FROM big_table", engine, chunksize=50_000):
#     process(chunk)
# df.to_sql("results", engine, if_exists="append", index=False)       # ≈ 批量 INSERT
```

## 4. 详细推理（Deep Dive）

### 4.1 CSV 为什么慢？—— 行存 + 纯文本 + 运行时类型推断

CSV 是**面向行的纯文本**格式，读取一份 CSV 时，pandas 必须做三件昂贵的事：

```
CSV 读取的三道开销 / Three costs of reading CSV:
  ① 词法切分（tokenizing）：逐字节扫描，按分隔符/引号/转义切出字段
        —— 还要处理含逗号的引号字段、跨行字段、编码（UTF-8 变长字符）
  ② 文本→类型转换（parsing）："123" → int64、"3.14" → float64、"2026-01-01" → datetime
        —— 字符串解析为数字本身就慢，逐元素进行
  ③ 类型推断（inference）：未指定 dtype 时，pandas 要"先抽样/扫描整列"猜类型
        —— 这是默认路径下的隐藏成本，也是误判（如把邮编 00123 读成 int 丢前导零）的根源
```

本质矛盾：CSV 没有保存任何类型信息（一切都是文本），也没有列的边界元数据，所以**每次读取都要把整份文件重新解析一遍**。对比 Java：用 Jackson/OpenCSV 解析 CSV 同样要逐字段 tokenize + 反序列化，慢的根因一致——文本解析无法避免。

加速 CSV 读取的工程手段：① 传 `dtype` 跳过推断；② 传 `usecols` 少读列；③ 用 `engine="pyarrow"`（pandas 2.x 起，PyArrow 引擎多线程解析，比默认 C 引擎更快）；④ 直读压缩包 `compression="gzip"` 省 I/O。但**根治之道是换列式格式**。

### 4.2 Parquet 为什么快？—— 列式存储 + Row Group + 谓词下推

Parquet 是**面向列**的二进制格式，文件被组织成若干 **Row Group（行组）**，每个 Row Group 内按列连续存放，并为每列保存**统计元数据**（min/max/null 数）：

```
Parquet 文件物理布局 / Physical layout:
  ┌──────────────── Row Group 0 ────────────────┐
  │  Column A 的所有值（连续）  [统计: min/max]   │   ← 同列同类型，编码/压缩极高效
  │  Column B 的所有值（连续）  [统计: min/max]   │
  │  Column C 的所有值（连续）  [统计: min/max]   │
  └──────────────────────────────────────────────┘
  ┌──────────────── Row Group 1 ────────────────┐ ...
  ┌──────────────── Footer ─────────────────────┐
  │  Schema（列名 + 类型）+ 各 Row Group 的元数据  │   ← 读取时先读 footer 拿到布局
  └──────────────────────────────────────────────┘
```

三大加速来源：

- **列裁剪（column pruning）**：`columns=["a"]` 时，只读 Column A 的字节，物理跳过 B/C。行存的 CSV 做不到——它必须读完整行才能拿到某一列。
- **类型保留 + 高压缩**：同一列数据类型一致、取值相近，编码（字典编码、RLE、delta）+ 压缩（zstd/snappy）效果远好于混合文本；且 schema 写在 footer，**读回来类型无损**（CSV 读回全是文本要重新推断）。这就是"1GB CSV → ~200MB Parquet"的来由。
- **谓词下推（predicate pushdown）**：`filters=[("year","==",2026)]` 时，引擎先读 footer 里每个 Row Group 的 min/max，**若某 Row Group 的 year 范围根本不含 2026，就整块跳过、不读其字节**。这把"过滤"从读完再筛，提前到读取层就剪枝，省的是真实磁盘 I/O。

> Java/数据库类比：谓词下推正是列式数仓（如 ClickHouse、Iceberg + Parquet）和 Spark 的核心优化；min/max 统计 ≈ 数据库的"区域映射 / zone map"，作用类似索引的"先看摘要再决定是否读数据块"。

### 4.3 大文件内存优化清单（实战）

```python
# ① 缩小数值精度：int64→int32、float64→float32，内存直接减半
df["id"] = df["id"].astype("int32")
df["score"] = df["score"].astype("float32")

# ② 低基数字符串列转 category：N 行只存 N 个整型码 + 一张小字典
#    Low-cardinality strings → category: stores N int codes + a tiny dictionary
df["country"] = df["country"].astype("category")   # 上百万行的国家列可省 90%+ 内存

# ③ 用 PyArrow 后端的可空类型，避免 object 列的指针膨胀（pandas 2.x）
df = pd.read_csv("data.csv", dtype_backend="pyarrow")

# ④ 真的放不下：chunksize 流式处理，或直接换 Parquet + filters 只读需要的子集
```

### 4.4 与大模型数据集 `datasets` 的衔接

HuggingFace `datasets` 库是 LLM 数据流水线的事实标准，它**底层正是 Apache Arrow + Parquet**，因此与 pandas 无缝互转、且能内存映射超大语料（不必全量载入内存）。

```python
from datasets import Dataset, load_dataset

# pandas → datasets：把清洗好的 DataFrame 交给训练流水线
# Time: O(N) Space: O(N)
ds = Dataset.from_pandas(df)               # 零拷贝走 Arrow / zero-copy via Arrow

# datasets → pandas：把现成数据集拉回来做 EDA/可视化
df = load_dataset("imdb", split="train").to_pandas()

# 落盘归档：训练语料统一用 Parquet 分片存储，配合 filters 按需读子集
ds.to_parquet("corpus/train.parquet")      # 大规模语料 = Parquet + 多分片
```

```
大模型数据格式选择 / Format choice for LLM data:
  SFT 数据   : JSONL（Alpaca/ShareGPT 格式，人读友好、逐行流式）
  Embedding  : Parquet（高维向量列存 + 压缩，省空间、读取快）
  评估结果   : CSV（小、直观、便于 Excel/人工查看）
  大规模语料 : Parquet + 分片（列裁剪 + 谓词下推 + 内存映射，datasets 原生支持）
```

> Java 对比小结：pandas 的 I/O 层在生态位上对应 Java 的 **JDBC**（关系库读写，`read_sql`/`to_sql` ≈ `JdbcTemplate`）+ **Jackson**（JSON 序列化，`read_json`/`to_json` ≈ `ObjectMapper`）。而 Parquet/Arrow 这套列式格式是大数据/AI 领域的跨语言标准，Java 侧亦有 `parquet-mr`、Arrow Java 实现，二者读写的是同一份文件——这正是"语言无关数据层"的价值所在。

## 5. 例题（Worked Examples）

### 例题 1：从字符串中读取 CSV 数据、清洗缺失值并导出为 JSON 格式 / Reading CSV, cleaning, and exporting to JSON

在机器学习流水线中，数据入库与出库是常见场景。本例展示如何读取结构化文本，进行预处理并导出。

```python
import pandas as pd
import io

# 模拟输入流 CSV 数据 / Simulate incoming CSV data string
csv_data = """id,user,age,city
101,Alice,25,Beijing
102,Bob,,Shanghai
103,Charlie,30,
"""

# 1. 加载数据 / Load data from file-like string
# Time: O(N), Space: O(N)
df = pd.read_csv(io.StringIO(csv_data))

# 2. 缺失值填充 / Handle missing values
# Time: O(N), Space: O(1)
df['age'] = df['age'].fillna(df['age'].mean())
df['city'] = df['city'].fillna('Unknown')

# 3. 导出为 JSON 格式 / Export to JSON
# Time: O(N), Space: O(N)
json_output = df.to_json(orient='records', force_ascii=False)
print("导出的 JSON 数据 / Exported JSON:")
print(json_output)
```

### 例题 2：把大 CSV 分块、压缩内存后转存为 Parquet（ETL 实战 / ETL pipeline）

需求：上游给一份"宽且大"的 CSV 日志，需要只保留 3 列、修正类型以压缩内存、并落盘为 Parquet 供后续训练读取。这把 §3 的 `usecols`/`dtype`/`chunksize` 与 §4 的内存优化、列存落盘串了起来。

```python
import pandas as pd

SRC = "events.csv"        # 假设上游大文件 / large upstream CSV
DST = "events.parquet"    # 列存落盘目标 / columnar output
KEEP = ["user_id", "event_type", "created_at"]   # 只要这三列 / project 3 columns

# 思路：分块读 → 每块就地压缩类型 → 流式追加写 Parquet，峰值内存只占一块
# Plan: chunked read → shrink dtypes per chunk → stream-append to Parquet
# Time: O(N) 单遍扫描 / single pass; Space: O(chunk) 峰值仅一块 / one chunk at a time
import pyarrow as pa
import pyarrow.parquet as pq

writer = None
reader = pd.read_csv(
    SRC,
    usecols=KEEP,                       # ① 列裁剪：少读列 / project early
    dtype={"user_id": "int32"},         # ② int32 省一半内存 / halve memory
    parse_dates=["created_at"],         # ③ 读时即转 datetime / parse dates at read
    chunksize=200_000,                  # ④ 每块 20 万行 / 200k rows per chunk
)
try:
    for chunk in reader:
        # 低基数字符串列转 category，内存骤降 / low-cardinality → category
        chunk["event_type"] = chunk["event_type"].astype("category")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:              # 首块用其 schema 初始化 writer / init on first chunk
            writer = pq.ParquetWriter(DST, table.schema, compression="zstd")
        writer.write_table(table)       # 流式追加写 / stream-append
finally:
    if writer is not None:
        writer.close()                  # 确保 footer 写入 / ensure footer flushed

# 验证：回读时只取 user_id 列 + 谓词下推，几乎不碰其余字节
# Verify: read back one column + pushdown, touching almost no other bytes
back = pd.read_parquet(DST, columns=["user_id"])
print(f"行数 / rows = {len(back)}")
```

> 工程要点：① CSV→Parquet 是数据工程里最常见的"一次转换、长期收益"动作——之后所有读取都享受列裁剪/压缩/谓词下推；② `ParquetWriter` 流式写让我们能处理远超内存的文件（呼应 §4.3）；③ `compression="zstd"` 在压缩比与速度间平衡较好。这套"分块 ETL"模式，等价于 Java 用 `try-with-resources` 管理流 + 批处理写出，避免 OOM。

## 6. 习题（Exercises）

### 基础题
**练习 1**：如何让 `pd.read_csv` 方法只加载文件中的前 100 行数据，以节省读取大文件时的内存占用？
*参考答案*：
在 `pd.read_csv` 方法中指定 `nrows=100` 参数。
```python
# df = pd.read_csv("large_dataset.csv", nrows=100)
```

### 进阶题
**练习 2**：从磁盘读取一个大型 Excel 文件时，其中某个“时间”列是自定义字符串格式。编写代码在读取时利用 `parse_dates` 自动将其转换为 datetime 对象，并以“年-月”作为索引。
*参考答案*：
```python
# df = pd.read_excel("data.xlsx", parse_dates=['timestamp_column'])
# df.set_index(df['timestamp_column'].dt.to_period('M'), inplace=True)
```\n