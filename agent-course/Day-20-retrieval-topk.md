# Day 20 · 检索：query 嵌入 → 相似度搜索 → top-k 调参

> **今日目标**：把"提问"接到向量库上——query 嵌入、相似度检索、取 top-k，并亲手调 k 值和相似度阈值看召回怎么变。
> **时长**：~2h ｜ **前置**：Day 19（库里已有数据）
> **今日产出**：一个 `day20_retrieve.py`，对一句 query 返回 top-k 召回结果（带分数和来源），并能用阈值过滤掉"不够相关"的块。

## 1. 为什么 & 是什么

库灌好了，今天专注**检索（retrieval）**这一环——它是 RAG 质量的**上游闸门**：检索召回的内容决定了模型"看得到什么资料"。检索召回错了，再强的模型也只能基于错料胡编。所以"召回质量"比"生成措辞"重要得多。

给 Java 工程师的贴切类比：

| 检索世界 | Java/搜索世界类比 | 说明 |
|---|---|---|
| query 嵌入 | 把查询条件编译成可执行的查询对象 | 问题也要用**同一个模型**变成向量 |
| top-k | `LIMIT k` | 取最相近的 k 条，k 是关键超参 |
| 相似度阈值 | `WHERE score > 阈值` | 砍掉"勉强沾边"的低分召回 |
| recall vs precision | 查全率 vs 查准率 | k 调大→查全↑查准↓，反之亦然 |
| 召回为生成铺路 | Service 先查 Repository 再算 | 检索是上游，生成是下游 |

两个必须建立的调参直觉：

- **k（top-k）的权衡**：k 是"召回几块塞给模型"。**k 太小**：可能漏掉含答案的块（查全率低，答案不全甚至答不出）。**k 太大**：把无关块也塞进上下文，**噪声淹没信号**，模型被带偏，还多花 token。常用 **k=3~5** 起步，按效果调。它和 chunk size 是一对联动参数——块小就多取几个，块大就少取。
- **相似度阈值（threshold）**：只取分数够高的。库里**没有**相关内容时（问了库外的事），不设阈值会强行返回几条最不相关的"垫底货"，模型基于它们就**幻觉**。设阈值能让系统**该空就空**——这正是 Day 22"不知道就说不知道"的技术前提。注意：阈值和 k 是"先过阈值、再取前 k"的关系。

> **2026 实践**：别迷信单一 k。生产里常用**两段式**——先粗召回较多（k=20~50），再用 reranker 精排取前几（Day 23）。今天先把"嵌入→检索→top-k→阈值"这条基本盘吃透，进阶留到 Day 23。

## 2. 跟着做（Hands-on）

**Step 1 — 复用 Day 19 的库（确保里面有数据）**

```bash
# 若库空了，先跑一遍 Day 19 的 ETL 灌数据 / re-run Day 19 ETL if the table is empty
python day19_etl.py
```

**Step 2 — 检索 + top-k + 阈值**

```python
"""Day 20: 向量检索 top-k 与阈值过滤 / vector retrieval with top-k and threshold."""

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:pass@localhost:5432/postgres"
model = SentenceTransformer("BAAI/bge-m3")


def retrieve(
    conn: psycopg.Connection, query: str, k: int = 5, min_sim: float = 0.0,
) -> list[tuple[str, str, float]]:
    """检索 top-k 并按相似度阈值过滤 / top-k retrieval filtered by similarity.

    Args:
        conn: 数据库连接 / db connection.
        query: 用户问题 / the user query.
        k: 取前 k 条 / number of chunks to return.
        min_sim: 余弦相似度下限，低于则丢弃 / drop results below this cosine sim.

    Returns:
        列表 [(content, source, 相似度)]，按相似度降序 / sorted by sim desc.
    """
    qvec = model.encode([query], normalize_embeddings=True)[0]  # shape: (1024,)
    # 1 - 余弦距离 = 余弦相似度，便于和阈值比较 / convert distance to similarity
    rows = conn.execute(
        "SELECT content, source, 1 - (emb <=> %s) AS sim "
        "FROM chunks ORDER BY emb <=> %s ASC LIMIT %s",
        (qvec, qvec, k),
    ).fetchall()
    # 阈值过滤：砍掉勉强沾边的 / drop weak matches below the threshold
    return [(c, s, sim) for c, s, sim in rows if sim >= min_sim]


def main() -> None:
    """对比不同 k 与阈值下的召回 / compare recall under different k & threshold."""
    with psycopg.connect(DSN) as conn:
        register_vector(conn)

        q = "怎么让 Postgres 支持向量检索？"
        print(f"Query: {q}\n")

        print("--- k=3, 无阈值 / no threshold ---")
        for c, s, sim in retrieve(conn, q, k=3, min_sim=0.0):
            print(f"sim={sim:.3f} [{s}] {c[:30]}")

        print("\n--- k=3, 阈值 0.5 / threshold 0.5 ---")
        hits = retrieve(conn, q, k=3, min_sim=0.5)
        for c, s, sim in hits:
            print(f"sim={sim:.3f} [{s}] {c[:30]}")
        if not hits:
            print("（无满足阈值的结果，应回答'不知道' / nothing passed → say 'I don't know')")

        # 问一个库里根本没有的事，看阈值如何拦住幻觉 / out-of-KB question
        print("\n--- 库外问题 + 阈值 0.5 / out-of-KB + threshold ---")
        oob = retrieve(conn, "如何用 Kubernetes 部署微服务？", k=3, min_sim=0.5)
        print(f"召回 {len(oob)} 条 / hits（理想为 0）")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day20_retrieve.py
```

预期：相关 query 在低阈值下召回 pgvector/Postgres 相关块；加 0.5 阈值后仍保留高分块；而**库外问题**（K8s）在阈值过滤后**召回 0 条**——系统学会了"没料就别硬答"。

## 3. 今日任务

1. 跑通 `day20_retrieve.py`，确认三组对比：无阈值召回、有阈值召回、库外问题被阈值拦住。
2. **扫 k 看查全/查准**：把 k 从 1 调到 10，观察召回列表变长、靠后的块相似度越来越低——理解"k 越大噪声越多"。
3. **找阈值拐点**：打印某个 query 全部块的相似度，挑一个能"留下相关、滤掉无关"的阈值，记下你这套数据的合理区间（通常 0.4~0.6）。

**验收标准**：能对同一 query 跑出不同 k 的召回并解释差异；阈值能成功拦住库外问题（召回 0）；你能给出"这套数据 k 取几、阈值取多少"的经验值并说明理由。

## 4. 自测清单

- [ ] 我理解"检索是 RAG 的上游闸门，召回错则全错"。
- [ ] 我能说清 k 太大/太小对查全率与查准率的影响。
- [ ] 我知道相似度阈值如何拦住"库外问题"，避免硬答。
- [ ] 我知道 query 必须用和建库相同的 embedding 模型。
- [ ] 我跑通了 top-k + 阈值检索，并标定了这套数据的合理参数。

## 5. 延伸 & 关联

- 阈值不是绝对的：不同 embedding 模型的分数分布不同，bge-m3 的 0.5 和别的模型的 0.5 不等价，要按你的数据实测标定。
- 召回评估：可以人工标注"这个 query 的正确块是哪几个"，算 Recall@k / MRR，量化不同 k 的效果——Day 25 项目会用到。
- 本仓库已有的相关章节：
  - RAG 基础概念（检索步骤与相似度）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
  - 向量数据库（ANN 索引如何影响召回与延迟）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
