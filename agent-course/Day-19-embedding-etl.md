# Day 19 · 嵌入 ETL Pipeline：读入→切分→嵌入→入库

> **今日目标**：把前三天的零件串成一条流水线——一个目录里的若干文档，自动加载、切分、批量嵌入、写进 pgvector，跑通完整 ETL。
> **时长**：~2h ｜ **前置**：Day 16~18（embedding / pgvector / chunking）
> **今日产出**：一个 `day19_etl.py`，对 `./docs` 目录下的 `.md`/`.txt` 文件一键灌库，打印各阶段计数（文件数→块数→入库数）。

## 1. 为什么 & 是什么

前三天你分别学了嵌入、向量库、切分；今天把它们**装配成一条管道**。这就是 RAG 的"建库"阶段——离线、批量、可重跑。它和后端的 **ETL（Extract-Transform-Load）** 是同一套思路：从源头抽数据（Extract=加载文档）、做转换（Transform=切分+嵌入）、装载到目标库（Load=写 pgvector）。

给 Java 工程师的贴切类比：

| 嵌入 ETL 世界 | Java/数据工程世界类比 | 说明 |
|---|---|---|
| Document Loader | 数据源 Reader（JDBC reader / 文件 reader） | 从 PDF/MD/HTML 抽出纯文本 |
| TextSplitter | Transform 阶段的"清洗+分片" | 把长文切成可嵌入的块 |
| 批量 `encode` | 批处理（batch insert / Spring Batch chunk） | 一次嵌一批，远快于逐条 |
| 写入 pgvector | Load 到目标库（`saveAll`） | 落库，附带元数据（来源、序号） |
| 幂等重跑 | `MERGE` / upsert，避免重复灌 | 重跑不该把同一文档灌两遍 |

三个工程要点：

- **批处理（batching）是性能命门**：embedding 模型按批推理效率远高于逐条。1000 个 chunk 逐条 `encode` 调 1000 次，按 batch（如 64）只需十几次。这是典型的 **O(N) 但常数差几十倍**——别在循环里逐条嵌入。
- **元数据（metadata）一起存**：每个 chunk 入库时带上**来源文件名、块序号**等。没有它，将来答案就**无法溯源**（Day 22 的引用全靠这个），也没法做元数据过滤（Day 23）。建表时多留几列。
- **幂等可重跑**：ETL 要能反复跑而不污染库。最简单：每次重灌前按来源 `DELETE` 旧块再插（小库够用）；大库用 `content_hash` 去重或 upsert。**这是和"只会跑一次 demo"拉开差距的地方。**

> **2026 实践**：langchain 的 loader/splitter 生态成熟，直接复用；嵌入仍用 `bge-m3`。真实数据脏（空文件、超长行、编码问题），ETL 里**该跳过的跳过、该记日志的记日志**，别让一个坏文件搞崩整批。

## 2. 跟着做（Hands-on）

**Step 1 — 装包 + 备好数据 + 起库**

```bash
pip install langchain-community langchain-text-splitters \
            "psycopg[binary]>=3.2" pgvector "sentence-transformers>=3.0"
# 复用 Day 17 的 Docker 库；准备几个文档 / reuse Day 17's container; add some docs
mkdir -p docs && printf "RAG 是检索增强生成。\npgvector 让 Postgres 存向量。\n" > docs/a.md
printf "Embedding 把文本变成向量。\n余弦相似度衡量语义距离。\n" > docs/b.txt
```

**Step 2 — ETL 主程序**

```python
"""Day 19: 嵌入 ETL —— 加载/切分/嵌入/入库 / load, split, embed, load to db."""

from pathlib import Path

import psycopg
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:pass@localhost:5432/postgres"
DIM = 1024
model = SentenceTransformer("BAAI/bge-m3")


def setup(conn: psycopg.Connection) -> None:
    """建表：content + 来源 source + 块序号 chunk_no + 向量。"""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        f"CREATE TABLE chunks (id bigserial PRIMARY KEY, content text, "
        f"source text, chunk_no int, emb vector({DIM}))"
    )
    conn.commit()


def load_and_split(dir_path: str) -> list[tuple[str, str, int]]:
    """加载目录文档并切块 / load docs from a dir and split into chunks.

    Returns:
        列表 [(content, source, chunk_no)] / list of (text, source, index).
    """
    # 加载 .md 与 .txt（glob 支持列表，别用字符类硬凑）/ load .md and .txt
    loader = DirectoryLoader(
        dir_path, glob=["**/*.md", "**/*.txt"], loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=40, separators=["\n\n", "。", "\n", ""],
    )
    out: list[tuple[str, str, int]] = []
    for d in docs:
        src = Path(d.metadata["source"]).name  # 文件名作来源 / file name as source
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            out.append((piece, src, i))
    return out


def embed_and_load(conn: psycopg.Connection, rows: list[tuple[str, str, int]]) -> int:
    """批量嵌入并写库 / batch-embed and insert. Returns 入库条数 / rows inserted."""
    texts = [r[0] for r in rows]
    # 一次性批量嵌入（batch_size 由库内部管理）/ embed all chunks in batch
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64)  # (n, 1024)
    with conn.cursor() as cur:
        for (content, src, no), vec in zip(rows, vecs):
            cur.execute(
                "INSERT INTO chunks (content, source, chunk_no, emb) VALUES (%s,%s,%s,%s)",
                (content, src, no, vec),
            )
    conn.commit()
    return len(rows)


def main() -> None:
    """端到端 ETL / end-to-end ETL with stage counters."""
    rows = load_and_split("./docs")
    files = len({r[1] for r in rows})
    print(f"加载文件 / files = {files}，切出块 / chunks = {len(rows)}")
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        setup(conn)                      # 幂等：每次重建表 / idempotent rebuild
        n = embed_and_load(conn, rows)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"入库 / inserted = {n}，库内总数 / total in db = {total}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day19_etl.py
```

预期：打印类似 `加载文件=2，切出块=N` → `入库=N，库内总数=N`。两次运行 `库内总数` 应**保持不变**（因为每次重建表，幂等）。你现在有了一条**可重跑的灌库流水线**。

## 3. 今日任务

1. 跑通 `day19_etl.py`，确认三个计数（文件→块→入库）一致，且重跑后库内总数不翻倍。
2. **加文档再灌**：往 `./docs` 再丢一个 `.md`，重跑，确认文件数+1、块数与入库数同步增长——验证 pipeline 真的扫了新文件。
3. **验证元数据落库**：连库执行 `SELECT source, chunk_no, left(content,20) FROM chunks ORDER BY source, chunk_no;`，确认每块都带了正确的来源和序号——这是 Day 22 引用的前提。

**验收标准**：ETL 一键跑通且各阶段计数自洽；重跑不重复灌库；新增文档能被纳入；库里每条都有 `source` 和 `chunk_no`。

## 4. 自测清单

- [ ] 我能把 RAG 建库阶段对应到经典 ETL 的 Extract/Transform/Load。
- [ ] 我知道为什么要批量嵌入而非循环逐条（常数级性能差异）。
- [ ] 我知道为什么 chunk 入库必须带 source/chunk_no 元数据。
- [ ] 我理解"幂等可重跑"的意义，能说出一种去重/重灌做法。
- [ ] 我跑通了整条 pipeline，并验证了元数据正确落库。

## 5. 延伸 & 关联

- 真实场景文件格式杂：`PyPDFLoader`(PDF)、`UnstructuredWordDocumentLoader`(docx)、`WebBaseLoader`(网页) 等，按需替换 loader 即可，下游不变。
- 增量更新：生产里不会每次全量重灌，而是按 `content_hash` 只嵌入"变更/新增"的文档，省算力——这是把 demo 推向生产的一步。
- 本仓库已有的相关章节：
  - RAG 实战（完整建库+检索代码）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - LangChain 基础（loader/splitter/向量库链路）：[../07-llm-applications/05-langchain/01-langchain-basics.md](../07-llm-applications/05-langchain/01-langchain-basics.md)
