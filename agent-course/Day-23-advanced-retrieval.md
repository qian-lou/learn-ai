# Day 23 · 进阶检索：metadata 过滤 + 混合检索 + rerank

> **今日目标**：把"裸向量检索"升级成生产级三件套——metadata 过滤缩小范围、混合检索（关键词+语义）补盲区、reranker 精排提质量。
> **时长**：~2h ｜ **前置**：Day 20~22（检索/RAG/引用）
> **今日产出**：一个 `day23_advanced.py`，演示三段式检索：先 metadata 过滤+混合召回较多候选，再用 `bge-reranker-v2-m3` 精排取前几，对比纯向量检索的排序差异。

## 1. 为什么 & 是什么

纯向量检索有三个软肋：① 召回**整个库**，无法限定"只在某产品/某时间段的文档里找"；② 对**精确关键词**（型号、人名、错误码）不敏感，纯语义可能漏；③ top-k 的**排序未必最优**，最该排第一的可能在第三。今天三招各破一个，这是把 RAG 从"demo 级"推到"能打"的关键一跳。

给 Java 工程师的贴切类比：

| 进阶检索世界 | Java/搜索世界类比 | 说明 |
|---|---|---|
| metadata 过滤 | `WHERE category=? AND date>?` | 向量检索前先按结构化字段缩小范围 |
| 关键词检索（BM25/全文） | ES 的 `match` / SQL `LIKE`、全文索引 | 抓精确词项，向量抓语义，互补 |
| 混合检索（hybrid） | 多路召回后合并打分 | 语义路 + 关键词路，结果融合（RRF） |
| reranker（重排） | 召回后再过一层精排打分器 | 用更强的交叉编码器对候选两两精算相关度 |
| 召回→精排两段式 | 粗排 + 精排（推荐系统经典架构） | 先多快好省地捞一批，再慢工出细活排前几 |

三招分别是什么：

- **metadata 过滤**：检索时附加结构化条件（`source`、`category`、`date`…），**先过滤再算相似度**。例如"只在 2025 年的财报里找"。Day 19 存的元数据此刻全派上用场。这是**最便宜见效最快**的一招，优先用。
- **混合检索（hybrid search）**：**语义检索 + 关键词检索**两路并行，再融合排序（常用 **RRF, Reciprocal Rank Fusion**——按各路名次倒数加权）。向量擅长"意思相近"，关键词擅长"精确命中"（型号 `bge-m3`、错误码 `ERR_500`），合起来召回更全。
- **rerank（重排序）**：向量检索是"query 和 doc 各自编码再比距离"（双塔，快但糙）；**reranker 是交叉编码器**——把 query 和每个候选**拼在一起**送进模型精算相关度（慢但准）。流程：先向量粗召回 20~50 条，再用 reranker 精排取前 3~5。**精度提升通常最显著**，代价是每个候选都要过一次模型。

> **2026 选型**：重排模型用 **`BAAI/bge-reranker-v2-m3`**——多语言、轻量、社区主流。混合检索的关键词路，pgvector 用户可直接用 Postgres 自带**全文检索（`tsvector`/`ts_rank`）**，无需另起 ES。

## 2. 跟着做（Hands-on）

**Step 1 — 装包**

```bash
pip install "sentence-transformers>=3.0" FlagEmbedding numpy
# FlagEmbedding 提供 bge-reranker；首次跑会下载权重 / ships the reranker, downloads on first run
```

**Step 2 — metadata 过滤 + 混合召回 + rerank**

```python
"""Day 23: 进阶检索 —— 过滤 + 混合 + 重排 / filtering, hybrid, rerank."""

import numpy as np
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-m3")
# 交叉编码重排器；use_fp16 省显存 / cross-encoder reranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# 简化的"库"：每条带文本与元数据 / a tiny in-memory KB with metadata
DOCS = [
    {"text": "bge-m3 是多语言嵌入模型，输出 1024 维向量。", "cat": "model"},
    {"text": "bge-reranker-v2-m3 用交叉编码做重排序。", "cat": "model"},
    {"text": "Postgres 支持 tsvector 全文检索。", "cat": "db"},
    {"text": "向量检索用余弦距离排序。", "cat": "db"},
]


def keyword_hits(query: str, pool: list[dict]) -> list[int]:
    """极简关键词召回：命中任一词项即算 / naive keyword recall by term overlap."""
    terms = set(query.lower().replace("？", "").split())
    return [i for i, d in enumerate(pool) if any(t in d["text"].lower() for t in terms)]


def vector_topk(query: str, pool: list[dict], k: int) -> list[int]:
    """向量粗召回 top-k 的下标 / vector recall, returns indices of top-k."""
    qv = embedder.encode([query], normalize_embeddings=True)[0]  # (1024,)
    dv = embedder.encode([d["text"] for d in pool], normalize_embeddings=True)  # (n,1024)
    sims = dv @ qv  # 归一化后点积即余弦 / dot == cosine after normalize; (n,)
    return list(np.argsort(-sims)[:k])


def search(query: str, cat: str | None, k: int = 3) -> list[tuple[float, str]]:
    """三段式检索 / filter -> hybrid recall -> rerank.

    Args:
        query: 用户问题 / user query.
        cat: metadata 过滤的类别，None 表示不过滤 / category filter or None.
        k: 最终返回条数 / final number of results.

    Returns:
        列表 [(rerank 分数, 文本)]，按相关度降序 / sorted by rerank score desc.
    """
    # 1) metadata 过滤：先缩小候选池 / filter by metadata first
    pool_idx = [i for i, d in enumerate(DOCS) if cat is None or d["cat"] == cat]
    pool = [DOCS[i] for i in pool_idx]

    # 2) 混合召回：向量路 ∪ 关键词路，取并集做候选 / union of vector & keyword recall
    cand = set(vector_topk(query, pool, k=min(len(pool), 4))) | set(keyword_hits(query, pool))
    cand_texts = [pool[i]["text"] for i in cand]

    # 3) rerank：交叉编码器对 (query, doc) 精算分数 / cross-encoder rerank
    pairs = [[query, t] for t in cand_texts]
    scores = reranker.compute_score(pairs, normalize=True)  # 越大越相关 / higher = better
    ranked = sorted(zip(scores, cand_texts), key=lambda x: -x[0])
    return ranked[:k]


def main() -> None:
    """对比'有无 metadata 过滤'的检索结果 / compare with/without metadata filter."""
    q = "重排序用什么模型？"
    print(f"Query: {q}")
    print("\n--- 不过滤 / no filter ---")
    for s, t in search(q, cat=None):
        print(f"{s:.3f} | {t}")
    print("\n--- 过滤 cat=model / filtered ---")
    for s, t in search(q, cat="model"):
        print(f"{s:.3f} | {t}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day23_advanced.py
```

预期：rerank 后"bge-reranker-v2-m3 用交叉编码做重排序"以**明显更高的分数**排第一（交叉编码器把它和 query 的强相关性识别得很准）；加 `cat=model` 过滤后，db 类文档被提前**排除在候选之外**，召回更聚焦。

> 体会差异：纯向量检索里几条 model 文档分数可能咬得很近；reranker 会把"真正回答了问题"的那条**明显拉开**——这就是精排的价值。

## 3. 今日任务

1. 跑通 `day23_advanced.py`，确认 reranker 把最相关文档排到第一，且 metadata 过滤能缩小候选范围。
2. **对比向量 vs rerank 排序**：单独打印 `vector_topk` 的排序和 rerank 后的排序，找出至少一个"被 reranker 纠正了名次"的例子。
3. **测关键词路的价值**：构造一个含**精确词项但语义稍远**的 query（如直接问型号 `bge-reranker-v2-m3`），验证关键词路把向量路漏掉的精确命中补了回来。

**验收标准**：三段式检索跑通；能展示 reranker 改善了排序、metadata 过滤缩小了范围、关键词路补上了向量漏召的精确命中；你能讲清"召回（粗、全、快）+ 精排（细、准、慢）"两段式的分工。

## 4. 自测清单

- [ ] 我能说出纯向量检索的三个软肋，以及三招分别破哪个。
- [ ] 我理解 metadata 过滤为什么"最便宜见效最快"，要优先用。
- [ ] 我能解释混合检索为什么比单路更全（语义 vs 关键词互补）。
- [ ] 我能讲清双塔检索与交叉编码 reranker 的精度/速度权衡。
- [ ] 我跑通了三段式检索，并各举一例证明每招的价值。

## 5. 延伸 & 关联

- RRF 融合：真实混合检索不是简单取并集，而是用 `1/(k+rank)` 把多路名次融合成统一分数，比硬合并更稳，可作为下一步加固。
- 别为了用而用：很多场景"metadata 过滤 + 向量检索"就够好，reranker 是有明确召回质量瓶颈时才上的"重武器"，它会增加延迟。
- 本仓库已有的相关章节：
  - RAG 实战（Hybrid + Re-ranking 进阶方案）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - 向量数据库（payload filter / 混合检索能力对比）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
