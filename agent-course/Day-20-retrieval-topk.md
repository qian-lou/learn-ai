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

## 3. 检索评估实战（Recall@k / MRR / 命中率）

上面调 k、调阈值都靠"眼看召回列表顺不顺"——这是**主观**的。要判断"k=3 好还是 k=5 好""加了 Day 23 的 rerank 到底涨没涨"，得有**可量化**的指标。方法工业界很成熟：人工标注一小批 query 的**正确块（gold）**，再算三个指标。

给 Java 工程师的类比：这就是给检索器写**单元测试 + 断言**——gold 是期望值，指标是"通过率"，没有它你只能靠感觉说"好像变好了"。

**三个指标（都基于 chunk 唯一标识 `source#chunk_no`，复用 Day 19 存的元数据）：**

| 指标 | 定义 | 直觉 |
|---|---|---|
| **Recall@k** | 前 k 条命中的 gold 数 ÷ gold 总数 | "该找回的找回了几成"（查全率） |
| **Hit@k（命中率）** | 前 k 条里"有没有至少一个 gold"，0/1 | Day 25 说的"命中率"就是它 |
| **MRR** | 每条 query 取 `1/首个gold的名次`，再对全体求平均 | "正确块排得够不够靠前" |

**Step 1 — 标注一小批 query 的 gold 块**

复用 Day 19 的库，先看有哪些块可选（把它们的 `source#chunk_no` 抄下来当标注词典）：

```bash
# 列出所有块的唯一标识，方便你挑 gold / list chunk ids for labeling
psql "postgresql://postgres:pass@localhost:5432/postgres" \
  -c "SELECT source||'#'||chunk_no AS cid, left(content,30) FROM chunks ORDER BY source, chunk_no;"
```

挑 5~10 条你数据里有明确答案的 query，人工写下每条的正确块（gold 可有多个）。

**Step 2 — 指标函数 + 跑标注集**

下面这段**可独立运行**（内置一套离线标注数据，不连库也能跑通指标逻辑）；真实使用时把 `retrieved` 换成 `retrieve()` 的返回、`gold` 换成你标注的。

```python
"""Day 20: 检索评估 —— Recall@k / MRR / Hit@k / retrieval eval metrics."""


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Recall@k = |前k条 ∩ gold| / |gold|，衡量"该找回的找回了多少"。
    时间 O(k) 空间 O(k)。"""
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Hit@k（命中率）= 前 k 条里"有没有至少一个 gold"，返回 0/1。
    定义：1 if 前k条 ∩ gold ≠ ∅ else 0——即 Day 25 说的"命中率"。
    时间 O(k) 空间 O(k)。"""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """单条 query 的 RR = 1 / 首个 gold 出现的名次；没命中记 0。
    MRR 是它在全体 query 上的平均。时间 O(len) 空间 O(1)。"""
    for rank, cid in enumerate(retrieved, start=1):  # rank 从 1 计
        if cid in gold:
            return 1.0 / rank
    return 0.0


def evaluate(runs: list[tuple[list[str], set[str]]], k: int) -> dict[str, float]:
    """对一批 (检索结果, gold) 汇总三指标 / aggregate over a labeled set。
    时间 O(Q·len) 空间 O(k)，Q 为 query 数。"""
    n = len(runs)
    return {
        f"Recall@{k}": sum(recall_at_k(r, g, k) for r, g in runs) / n,
        f"Hit@{k}": sum(hit_at_k(r, g, k) for r, g in runs) / n,   # 命中率
        "MRR": sum(reciprocal_rank(r, g) for r, g in runs) / n,
    }


# 人工标注集：gold 用 "source#chunk_no"（复用 Day 19 元数据）。
# 内置 before=纯向量、after=+metadata过滤/rerank 两套排序，便于离线对比。
# 真实用法：after 那一列换成"接了 Day 23 三招后 retrieve() 的返回"。
LABELED = [
    ("q1", {"a.md#0"},
     ["a.md#1", "a.md#0", "b.txt#0", "b.txt#1", "a.md#2"],   # before
     ["a.md#0", "a.md#1", "b.txt#0", "b.txt#1", "a.md#2"]),  # after：gold 提到第 1
    ("q2", {"b.txt#0"},
     ["a.md#2", "b.txt#1", "b.txt#0", "a.md#0", "a.md#1"],
     ["b.txt#0", "b.txt#1", "a.md#2", "a.md#0", "a.md#1"]),
    ("q3", {"a.md#1", "a.md#2"},
     ["a.md#1", "b.txt#0", "a.md#2", "b.txt#1", "a.md#0"],
     ["a.md#1", "a.md#2", "b.txt#0", "b.txt#1", "a.md#0"]),
    ("q4", {"b.txt#1"},
     ["a.md#0", "a.md#1", "a.md#2", "b.txt#1", "b.txt#0"],   # before：gold 在第 4，k=3 漏
     ["b.txt#1", "a.md#0", "a.md#1", "a.md#2", "b.txt#0"]),  # after：过滤后提到第 1
    ("q5", {"a.md#0"},
     ["b.txt#0", "b.txt#1", "a.md#1", "b.txt#1", "a.md#0"],
     ["b.txt#0", "a.md#1", "b.txt#1", "a.md#0", "a.md#2"]),  # 这条 after 仍漏（真实：不会满分）
    ("q6", {"b.txt#0"},
     ["b.txt#0", "a.md#0", "a.md#1", "a.md#2", "b.txt#1"],
     ["b.txt#0", "a.md#0", "a.md#1", "a.md#2", "b.txt#1"]),  # 本就排第 1，两套都对
]


def main() -> None:
    """对比优化前 vs 后的检索指标 / compare metrics before vs after."""
    k = 3
    before = evaluate([(b, g) for _, g, b, _ in LABELED], k)
    after = evaluate([(a, g) for _, g, _, a in LABELED], k)
    print(f"{'指标':<10}{'优化前':>8}{'优化后':>8}")
    for key in before:  # dict 保序：Recall@k → Hit@k → MRR
        print(f"{key:<10}{before[key]:>8.3f}{after[key]:>8.3f}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day20_eval.py
```

预期输出（k=3，这套内置标注实跑得到）：

```
指标             优化前     优化后
Recall@3     0.667   0.833
Hit@3        0.667   0.833
MRR          0.547   0.875
```

**Step 3 — 把 Day 23 三招接进来，让"命中率涨"变成自己跑出的数字**

上面的 `after` 是内置的对照排序，真实做法是：把 `after` 那一列换成"**接了 Day 23 三招后的检索结果**"，同一批 gold 再算一次。三招各自会怎样抬指标：

| Day 23 优化 | 主要抬哪个指标 | 为什么 |
|---|---|---|
| **metadata 过滤** | Recall@k / Hit@k↑ | 先滤掉无关类别，gold 更容易进前 k |
| **混合检索（关键词路）** | Recall@k↑ | 把纯向量漏掉的"精确词项 gold"补回候选 |
| **rerank（精排）** | MRR↑ 最明显 | 把已召回的 gold 从第 3、第 4 **提到第 1** |

于是你会得到一张自己的对比表（下面是本例的实跑值）：

| 指标 | 纯向量（before） | +三招（after） | 变化 |
|---|---|---|---|
| Recall@3 | 0.667 | 0.833 | 该找回的更全了 |
| **Hit@3（命中率）** | **0.667** | **0.833** | 就是 Day 25 那张卡的"命中率 X→Y" |
| MRR | 0.547 | 0.875 | 正确块被明显提前 |

看懂这张表，Day 25 量化亮点卡里的"命中率 62%→90%"就不再是背下来的面试话术，而是**你在自己数据上量出来、能解释每个数字怎么来的**结果。

## 4. 今日任务

1. 跑通 `day20_retrieve.py`，确认三组对比：无阈值召回、有阈值召回、库外问题被阈值拦住。
2. **扫 k 看查全/查准**：把 k 从 1 调到 10，观察召回列表变长、靠后的块相似度越来越低——理解"k 越大噪声越多"。
3. **找阈值拐点**：打印某个 query 全部块的相似度，挑一个能"留下相关、滤掉无关"的阈值，记下你这套数据的合理区间（通常 0.4~0.6）。
4. **跑出你自己的检索指标**：按第 3 节标注 5~10 条 query 的 gold（用 `source#chunk_no`），算出你这套数据的 **Recall@k / MRR / 命中率**，并用一句话解释每个指标衡量的是什么。

**验收标准**：能对同一 query 跑出不同 k 的召回并解释差异；阈值能成功拦住库外问题（召回 0）；你能给出"这套数据 k 取几、阈值取多少"的经验值并说明理由；能跑出自己的 Recall@k/MRR/命中率并说清各自含义。

## 5. 自测清单

- [ ] 我理解"检索是 RAG 的上游闸门，召回错则全错"。
- [ ] 我能说清 k 太大/太小对查全率与查准率的影响。
- [ ] 我知道相似度阈值如何拦住"库外问题"，避免硬答。
- [ ] 我知道 query 必须用和建库相同的 embedding 模型。
- [ ] 我跑通了 top-k + 阈值检索，并标定了这套数据的合理参数。
- [ ] 我能算出 Recall@k / MRR / 命中率，并说清它们各自衡量什么。

## 6. 延伸 & 关联

- 阈值不是绝对的：不同 embedding 模型的分数分布不同，bge-m3 的 0.5 和别的模型的 0.5 不等价，要按你的数据实测标定。
- 评估自动化：本节是手工标注小集打底；生产里可用 **RAGAS** 等框架自动评 context recall / faithfulness，把这套指标接进 CI（Day 49~50 会系统学评估）。
- 本仓库已有的相关章节：
  - RAG 基础概念（检索步骤与相似度）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
  - 向量数据库（ANN 索引如何影响召回与延迟）：[../07-llm-applications/03-rag/02-vector-databases.md](../07-llm-applications/03-rag/02-vector-databases.md)
