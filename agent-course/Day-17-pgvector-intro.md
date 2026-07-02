# Day 17 · 向量库入门：pgvector（复用 Postgres）

> **今日目标**：用 pgvector 给 Postgres 加上"向量列"，存进几条 embedding 再按相似度查出来，跑通你的第一次向量检索。
> **时长**：~2h ｜ **前置**：Day 16（embedding & 余弦）、装过 Docker、会基本 SQL
> **今日产出**：一个 `day17_pgvector.py`，建表→写入 3~4 条句子向量→给定 query 查出最相近的 top-k，并打印距离。

## 1. 为什么 & 是什么

有了向量，得有地方**存**和**查**。专用向量库（Chroma/Milvus/Qdrant）很多，但对有 Postgres 的团队，**pgvector** 是上手成本最低的选择——它只是 Postgres 的一个**扩展**，给你一个新列类型 `vector` 和几个距离运算符。数据、事务、备份、权限全复用现有 Postgres，不用再运维一套新中间件。

给 Java 工程师的贴切类比：

| pgvector 世界 | Java/后端世界类比 | 说明 |
|---|---|---|
| `CREATE EXTENSION vector` | 引一个 starter 依赖（如加 `spring-boot-starter-data-jpa`） | 给现有数据库"开个能力"，不另起服务 |
| `vector(1024)` 列类型 | 一个定长 `float[]` 字段 | 维度写死在 DDL 里，和 embedding 模型对齐 |
| `<=>` 余弦距离运算符 | 自定义 `Comparator` 的"距离函数" | `<=>` 余弦距离、`<->` L2、`<#>` 负内积 |
| `ORDER BY emb <=> :q LIMIT 5` | `ORDER BY ... LIMIT`（语义版） | 这就是 top-k 检索，本质是一句 SQL |
| HNSW / IVFFlat 索引 | B-Tree 索引的"近似最近邻"版 | 百万级以上必建，否则全表扫 |

两个要点：

- **距离 vs 相似度**：pgvector 的 `<=>` 返回的是**余弦距离** = `1 - 余弦相似度`，所以**越小越相似**，`ORDER BY ... ASC` 取最近。别和昨天的"分数越大越好"搞反。
- **索引可以后置**：几千条数据**不建索引也很快**（顺序扫一遍），开发期先别折腾索引。等数据上百万、查询变慢，再加 HNSW（`CREATE INDEX ... USING hnsw (emb vector_cosine_ops)`）。这是典型的"先跑通、再优化"。

> **2026 提示**：pgvector 现已支持 `halfvec`（半精度，省一半存储）和并行索引构建。本节用基础 `vector` 类型即可。要在云上，Supabase、Neon、阿里云 RDS 都内置了 pgvector。

## 2. 跟着做（Hands-on）

**Step 1 — 用 Docker 起一个带 pgvector 的 Postgres**

```bash
# 官方镜像已内置 pgvector 扩展 / official image ships with pgvector
docker run -d --name pgvec \
  -e POSTGRES_PASSWORD=pass \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

**Step 2 — 装 Python 客户端**

```bash
pip install "psycopg[binary]>=3.2" pgvector "sentence-transformers>=3.0"
```

**Step 3 — 建表、写入、检索**

```python
"""Day 17: pgvector 建表/写入/检索 / create, insert, search with pgvector."""

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:pass@localhost:5432/postgres"
model = SentenceTransformer("BAAI/bge-m3")  # 1024 维 / 1024-dim
DIM = 1024


def setup(conn: psycopg.Connection) -> None:
    """开启扩展并重建文档表 / enable extension and (re)create the docs table."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS docs")
    # emb 列类型 vector(1024) 与模型维度对齐 / column dim must match the model
    conn.execute(
        f"CREATE TABLE docs (id bigserial PRIMARY KEY, content text, emb vector({DIM}))"
    )
    conn.commit()


def insert(conn: psycopg.Connection, texts: list[str]) -> None:
    """批量嵌入并写入 / embed in batch and insert."""
    # 一次性 encode 整批，避免逐条调用 / encode the whole batch at once
    vecs = model.encode(texts, normalize_embeddings=True)  # shape: (n, 1024)
    with conn.cursor() as cur:
        for text, vec in zip(texts, vecs):
            cur.execute("INSERT INTO docs (content, emb) VALUES (%s, %s)", (text, vec))
    conn.commit()


def search(conn: psycopg.Connection, query: str, k: int = 3) -> list[tuple]:
    """检索与 query 最相近的 top-k / retrieve top-k nearest to the query.

    Returns:
        列表 [(content, 余弦距离)]，距离越小越相近 / smaller distance = closer.
    """
    qvec = model.encode([query], normalize_embeddings=True)[0]  # shape: (1024,)
    # <=> 余弦距离运算符；ASC 取最近 / cosine-distance operator, ASC = nearest first
    rows = conn.execute(
        "SELECT content, emb <=> %s AS dist FROM docs ORDER BY dist ASC LIMIT %s",
        (qvec, k),
    ).fetchall()
    return rows


def main() -> None:
    """端到端：建表→写入→检索 / end-to-end demo."""
    with psycopg.connect(DSN) as conn:
        register_vector(conn)  # 让 psycopg 认识 vector 类型 / teach psycopg the type
        setup(conn)
        insert(conn, [
            "Python 是一门解释型动态语言",
            "Postgres 是开源关系型数据库",
            "向量数据库用于语义检索",
            "今天天气晴朗适合郊游",
        ])
        for content, dist in search(conn, "什么数据库适合做语义搜索？", k=3):
            print(f"dist={dist:.3f}  | {content}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day17_pgvector.py
```

预期：排第一的应是"向量数据库用于语义检索"或"Postgres 是开源关系型数据库"（距离最小），"今天天气"那条排最后。**一句 SQL 就完成了语义检索**——这就是 pgvector 的魅力。

## 3. 今日任务

1. 跑通 `day17_pgvector.py`，确认 query 能召回最相关的那条，且距离排序正确（小→大）。
2. **换 query 验证**：把 query 改成 `"哪种语言适合写脚本？"`，确认"Python..."那条排到第一——同一张表，换问法就召回不同内容。
3. **理解距离方向**：把 `ORDER BY dist ASC` 改成 `DESC` 跑一次，观察召回变成"最不相关"的——亲手验证"余弦距离越小越相似"。

**验收标准**：表能建出来、4 条数据写入成功；两次不同 query 都能召回语义最贴切的条目；你能解释 `<=>` 返回的是距离（越小越好）而非相似度。

## 4. 自测清单

- [ ] 我能说清 pgvector 为什么是"复用 Postgres"而非新中间件。
- [ ] 我知道 `vector(1024)` 的维度必须和 embedding 模型对齐。
- [ ] 我能区分 `<=>`（余弦距离）和昨天的余弦相似度（差一个 `1 - x`）。
- [ ] 我知道开发期几千条可以不建索引，百万级才需要 HNSW。
- [ ] 我跑通了"建表→写入→top-k 检索"全流程。

## 5. 延伸 & 关联

- 想看 pgvector 的索引：百万级数据上 `CREATE INDEX ON docs USING hnsw (emb vector_cosine_ops)`，再对比建索引前后的查询延迟。
- 其他向量库横向对比（Chroma/Milvus/Qdrant/Weaviate 等）：完整表格见 [../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)（即下方第一个链接）。
- 本仓库已有的相关章节：
  - 向量数据库总览（含 pgvector 在内的横评）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
  - RAG 基础概念：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
