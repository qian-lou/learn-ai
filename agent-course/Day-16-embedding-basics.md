# Day 16 · Embedding 原理与余弦相似度

> **今日目标**：理解"文本 → 向量"是怎么回事，亲手算一次两段文本的余弦相似度，建立 RAG 的第一块基石。
> **时长**：~2h ｜ **前置**：Day 1（裸调用）、会一点 NumPy
> **今日产出**：一个 `day16_embed.py`，对一组句子取 embedding，打印两两余弦相似度矩阵，并验证"语义近 → 分数高"。

## 1. 为什么 & 是什么

RAG 的第一步是把文本变成**向量**（embedding）。你已经知道 LLM 把文本切成 token；embedding 模型则更进一步——把整段文本压缩成一个**定长浮点数组**（如 1024 维），让"语义"变成"空间里的位置"。语义相近的句子，向量也挨得近。

给 Java 工程师的贴切类比：

| Embedding 世界 | Java 世界类比 | 说明 |
|---|---|---|
| 文本 → 向量 | `hashCode()` 把对象映射成 int | 都是"降维成可比较的值"，但 embedding 保留**语义距离**，hashCode 不保留 |
| 1024 维 `float[]` | 一个定长 `float[1024]` 特征向量 | 维度固定，由模型决定；不同模型不可混用 |
| 余弦相似度 | `equals()` 的"模糊版"，返回 0~1 的相似度而非 true/false | 衡量两个向量"指向"是否一致 |
| 向量检索 | 不是 `WHERE id=?`，而是"按相似度排序取最近的几条" | 语义版的 `ORDER BY distance LIMIT k` |

两个必须建立的心智模型：

- **余弦相似度（cosine similarity）**：衡量两个向量**夹角**的余弦值，范围 [-1, 1]，实际文本通常落在 [0, 1]。公式 `cos = (A·B) / (|A|·|B|)`——点积除以两者模长。它只看**方向**不看长度，所以天然抗"文本长短不一"。1 = 完全同向（语义几乎一致），0 = 正交（无关）。
- **为什么不直接比字符串**：`"猫"` 和 `"小猫咪"` 字面不同但语义近；`"苹果手机"` 和 `"苹果派"` 字面都含"苹果"但语义远。关键词匹配（Java 里的 `String.contains`、ES 的 BM25）抓不住这层语义，embedding 可以。这就是 RAG 能"按意思找资料"的根。

> **2026 选型**：本系列默认嵌入模型用 **`BAAI/bge-m3`**——多语言、支持中英混合、1024 维、开源可本地跑，是当前社区 RAG 的主流之一。要走 API 路线也可用 OpenAI `text-embedding-3-small`（1536 维，便宜）。**同一个库里的所有向量必须用同一个模型生成**，换模型 = 整库重嵌。

## 2. 跟着做（Hands-on）

**Step 1 — 装包**

```bash
pip install "sentence-transformers>=3.0" numpy
# 首次运行会自动下载 bge-m3 权重（约 2GB），需要联网 / downloads weights on first run
```

**Step 2 — 取 embedding 并算余弦相似度**

```python
"""Day 16: 文本 embedding 与余弦相似度 / text embeddings & cosine similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer

# bge-m3：多语言嵌入模型，输出 1024 维向量
# bge-m3: multilingual embedding model, outputs 1024-dim vectors
model = SentenceTransformer("BAAI/bge-m3")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度 / cosine similarity of two vectors.

    Args:
        a: 向量 A，形状 (d,) / vector A of shape (d,).
        b: 向量 B，形状 (d,) / vector B of shape (d,).

    Returns:
        余弦相似度，范围 [-1, 1] / cosine similarity in [-1, 1].
    """
    # 时间 O(d) 空间 O(1)：点积除以模长之积
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def main() -> None:
    """对一组句子求 embedding，打印两两相似度 / embed sentences, print pairwise sim."""
    sentences = [
        "猫喜欢晒太阳",          # 0
        "小猫咪爱在阳光下打盹",    # 1 与 0 语义近 / close to 0
        "今天的股票大跌了",       # 2 无关 / unrelated
    ]
    # encode 返回 (n, 1024) 矩阵；normalize 后可直接点积当余弦
    # encode returns (n, 1024); normalized so dot product == cosine
    vecs = model.encode(sentences, normalize_embeddings=True)  # shape: (3, 1024)
    print(f"向量维度 / dim = {vecs.shape[1]}")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine(vecs[i], vecs[j])
            print(f"[{i}] vs [{j}]  cos = {sim:.3f}  | {sentences[i]} <-> {sentences[j]}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day16_embed.py
```

预期：`[0] vs [1]` 的相似度明显高（约 0.6~0.8），而 `[0] vs [2]`、`[1] vs [2]` 明显低（约 0.2~0.4）。你会**亲眼看到语义距离变成了数字**。

> 小贴士：`normalize_embeddings=True` 后向量模长为 1，余弦相似度 == 点积，省一步除法。生产里几乎都开归一化。

**Step 3 — 两条嵌入路线：本地 vs API（生产选型）**

上面的 `bge-m3` 是**本地路线**：首跑要下载约 2GB 权重、跑得快但吃 GPU/内存。生产上还有一条**API 路线**——不占本地资源，一次 HTTP 拿向量：

```python
from openai import OpenAI

# ============================================================
# API 嵌入：OpenAI text-embedding-3，批量 + 降维
# 需 OPENAI_API_KEY / needs API key
# ============================================================
client = OpenAI()


def embed_api(texts: list[str], dim: int = 512) -> list[list[float]]:
    """批量取 API 嵌入。时间 O(1) 次请求，空间 O(n·dim)。

    Args:
        texts: 待嵌入文本列表（单请求上限约 2048 条）/ up to ~2048 per call.
        dim: 目标维度，MRL 特性支持把 1536 维截短到 512 省存储 / Matryoshka dims.

    Returns:
        与输入顺序对齐的向量列表 / vectors aligned to input order.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,          # 批量：一次传多条，省往返 / batch to cut round-trips
        dimensions=dim,       # 降维：1536→512，检索更快、存储减半
    )
    return [d.embedding for d in resp.data]   # data 顺序保证与 input 一致


vecs = embed_api(["猫喜欢晒太阳", "今天的股票大跌了"])
print(len(vecs), len(vecs[0]))   # 2 512
```

两条路线怎么选：

| 维度 | 本地 `bge-m3` | API `text-embedding-3` |
|------|--------------|------------------------|
| 向量维度 | 1024（固定） | 1536，可 `dimensions` 降到 512/256 |
| 起步成本 | 首跑下 ~2GB 权重 | 无，直接调 |
| 是否需 GPU | 建议有（CPU 慢） | 不需要 |
| 单位成本 | 0 元（自备算力） | 按 token 计费（量级 $0.02/1M tokens） |
| 延迟 | 本地快、无网络往返 | 有网络往返 + 限流 |
| 数据出境 | 不出本机（合规友好） | 文本发往第三方 |

> **选型结论**：数据敏感 / 已有 GPU / 高频海量 → **本地**；无 GPU、快速起步、量不大 → **API**。切换成本极低——**只需把 `model.encode(...)` 换成 `embed_api(...)` 一行**，检索逻辑完全不动；换 API 时注意批大小上限与限流重试（见 Day-19 ETL）。

## 3. 今日任务

1. 跑通 `day16_embed.py`，确认"语义近的句子分数高、无关的分数低"。
2. **加一句中英混合**：往 `sentences` 里加 `"The cat is napping in the sun"`，看它和句子 `[0]`/`[1]` 的相似度——验证 bge-m3 的跨语言能力（中英同义句应当也很近）。
3. **造一个"陷阱对"**：构造两句**字面重叠但语义相反**的话（如 `"我很喜欢这家餐厅"` vs `"我一点都不喜欢这家餐厅"`），观察余弦相似度——体会 embedding 不是简单看词面。

**验收标准**：终端打印出维度（1024）和相似度矩阵；语义近的 > 0.55、无关的 < 0.45；中英同义句相似度也较高；你能解释"陷阱对"为什么分数没你想的那么低（共享大量上下文词）。

## 4. 自测清单

- [ ] 我能用一句话说清 embedding 是什么、为什么 RAG 需要它。
- [ ] 我能手写余弦相似度公式，并说清它"只看方向不看长度"的好处。
- [ ] 我知道"同一个库必须用同一个 embedding 模型"，换模型要重嵌。
- [ ] 我能区分"关键词匹配"和"语义匹配"各自抓不住什么。
- [ ] 我跑通了代码，验证了语义近→分数高、跨语言也成立。

## 5. 延伸 & 关联

- 余弦相似度只是距离度量之一，还有 L2（欧氏）、内积（IP）。归一化后三者关系密切，Day 17 入向量库时会再遇到。
- 想直观感受高维向量：搜索 "embedding projector"（TensorFlow 出品的可视化），把词向量投到 3D 看聚类。
- 本仓库已有的相关章节：
  - RAG 基础概念（五步流程总览）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
  - 预训练模型 BERT（embedding 的源头之一）：[../06-llm-core-technology/02-pretrained-models/01-bert.md](../06-llm-core-technology/02-pretrained-models/01-bert.md)
