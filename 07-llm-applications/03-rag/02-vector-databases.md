# 向量数据库 / Vector Databases

## 1. 背景（Background）

> **为什么要学这个？**
>
> 向量数据库是 RAG 的**核心基础设施**——存储和检索高维向量，支持毫秒级相似度搜索。当你有百万级文档时，暴力搜索不可行，需要向量数据库的 ANN（近似最近邻）索引。
>
> 对于 Java 工程师来说，向量数据库 就像 **Elasticsearch 的语义版本**——ES 做关键词匹配，向量库做语义匹配。

## 2. 知识点（Key Concepts）

| 向量库 | 类型 | 特点 | 适用规模 |
|--------|------|------|---------|
| Faiss | 库 | Facebook 出品，极致性能 | 亿级 |
| Chroma | 内嵌型 | 轻量，开发友好 | 百万级 |
| Milvus | 分布式 | 云原生，生产级 | 十亿+ |
| Pinecone | 云服务 | 全托管 | 任意 |
| pgvector | 扩展 | PostgreSQL 扩展 | 百万级 |
| Qdrant | 分布式 | Rust 实现，高性能、过滤（payload filter）能力强 / Rust-based, high perf, strong filtering | 千万-十亿 |
| Weaviate | 分布式 | 内置混合检索（向量+BM25）、模块化（可插拔 embedder） / built-in hybrid search, modular | 千万+ |
| LanceDB | 内嵌型 | 嵌入式列存（Lance 格式），多模态友好、零运维 / embedded columnar store, multimodal-friendly | 百万-千万 |

> **2024-2025 增长最快的三家：** Qdrant、Weaviate、LanceDB。Qdrant 以 Rust 性能 + 强过滤见长；Weaviate 主打开箱即用的混合检索与模块化；LanceDB 走嵌入式列存路线，适合多模态与本地/边缘场景。

## 3. 内容（Content）

### 3.1 Faiss（高性能向量检索）

```python
import faiss
import numpy as np

# ============================================================
# Faiss: Facebook 的高性能向量检索库
# ============================================================

d = 768  # 向量维度（BERT/BGE 输出维度）
n = 100000  # 向量数量

# 生成模拟数据
vectors = np.random.randn(n, d).astype('float32')
query = np.random.randn(1, d).astype('float32')

# 1. 精确搜索（暴力）
index_flat = faiss.IndexFlatL2(d)
index_flat.add(vectors)
D, I = index_flat.search(query, k=5)
print(f"精确搜索 Top-5 索引: {I[0]}")

# 2. IVF 索引（加速）
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = 10  # 检索时搜索的聚类数
D, I = index_ivf.search(query, k=5)

# 3. HNSW 索引（推荐，速度快+精度高）
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # M=32
index_hnsw.add(vectors)
D, I = index_hnsw.search(query, k=5)
```

### 3.2 Chroma（开发友好）

```python
import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# Chroma: 最简单的向量数据库
# ============================================================

client = chromadb.PersistentClient(path="./chroma_db")

# 本地 SentenceTransformer Embedding（HuggingFaceEmbeddingFunction 走 HF API 需 key）
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-zh-v1.5"
)

# 创建集合
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=ef,
)

# 添加文档（自动向量化）
collection.add(
    documents=["Python 是一种编程语言", "Java 是企业级编程语言", "深度学习使用神经网络"],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[{"source": "wiki"}, {"source": "wiki"}, {"source": "textbook"}],
)

# 查询
results = collection.query(query_texts=["编程语言推荐"], n_results=2)
print(results["documents"])
```

### 3.3 索引类型对比

```
ANN（近似最近邻）索引对比：

Flat（暴力搜索）:
  精度: 100%（精确）
  速度: O(N) — 慢
  适用: <10 万条

IVF（倒排文件）:
  精度: ~95%
  速度: O(N/nlist) — 快
  适用: 百万级

HNSW（分层导航小世界图）:
  精度: ~99%
  速度: O(log N) — 很快
  内存: 较高
  适用: 百万-千万级（推荐！）

PQ（乘积量化）:
  精度: ~90%
  速度: 极快
  内存: 极低（压缩 32x）
  适用: 亿级（牺牲精度换空间）
```

## 4. 详细推理（Deep Dive）

### 4.1 相似度度量

```
余弦相似度 (Cosine Similarity):
  sim(a, b) = a·b / (|a|·|b|)
  范围: [-1, 1]，1 表示完全相同
  适用: 文本语义相似度（推荐）

L2 距离 (欧氏距离):
  dist(a, b) = √Σ(aᵢ-bᵢ)²
  范围: [0, ∞)，0 表示完全相同
  适用: 已归一化的向量

内积 (Inner Product):
  score(a, b) = a·b
  适用: 已归一化时等价于余弦相似度
```

## 5. 例题（Worked Examples）

```python
# Faiss GPU 加速
# gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
# → 搜索速度提升 10-100x
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 Chroma 构建一个文档检索系统，支持语义搜索。

*参考答案*：

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
# 本地 SentenceTransformer，免 API key / local embedding, no API key
ef = embedding_functions.SentenceTransformerEmbeddingFunction("BAAI/bge-base-zh-v1.5")
col = client.get_or_create_collection("docs", embedding_function=ef)

col.add(documents=["Python 是编程语言", "深度学习使用神经网络", "向量数据库支持语义搜索"],
        ids=["d1", "d2", "d3"])
# 语义搜索：按含义匹配而非字面 / semantic search by meaning, not keywords
print(col.query(query_texts=["哪种语言适合编程"], n_results=2)["documents"])
```

**练习 2：** 对比 Faiss 的 Flat、IVF、HNSW 索引在 10 万条数据上的速度和精度。

*参考答案*：以 Flat 精确结果为 ground truth，统计 IVF/HNSW 的 Recall@k 与查询耗时。Flat 100% 但慢，HNSW 近 100% 且最快，IVF 介于两者。

```python
import faiss, numpy as np, time

d, n = 768, 100_000
xb = np.random.randn(n, d).astype("float32")
xq = np.random.randn(100, d).astype("float32")

flat = faiss.IndexFlatL2(d); flat.add(xb)
_, gt = flat.search(xq, 5)                    # ground truth

ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
ivf.train(xb); ivf.add(xb); ivf.nprobe = 10
hnsw = faiss.IndexHNSWFlat(d, 32); hnsw.add(xb)

for name, idx in [("Flat", flat), ("IVF", ivf), ("HNSW", hnsw)]:
    t = time.perf_counter(); _, I = idx.search(xq, 5); dt = time.perf_counter() - t
    # Recall@5 = 近似结果与精确结果的重合率 / overlap with exact top-5
    recall = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(I, gt)])
    print(f"{name}: {dt * 1000:.1f}ms, Recall@5={recall:.3f}")
```

### 进阶题

**练习 3：** 用 Faiss 构建百万级向量索引，测量不同 nprobe 值的精度-速度权衡。

*参考答案*：nprobe 是 IVF 检索时搜索的聚类数，越大越精确但越慢——典型的精度-速度权衡。

```python
import faiss, numpy as np, time

d, n = 768, 1_000_000
xb = np.random.randn(n, d).astype("float32")
xq = np.random.randn(100, d).astype("float32")

flat = faiss.IndexFlatL2(d); flat.add(xb)
_, gt = flat.search(xq, 10)

ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1000)
ivf.train(xb); ivf.add(xb)
for nprobe in (1, 10, 50, 100):
    ivf.nprobe = nprobe                       # 搜索的聚类数 / clusters probed
    t = time.perf_counter(); _, I = ivf.search(xq, 10); dt = time.perf_counter() - t
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(I, gt)])
    print(f"nprobe={nprobe}: {dt * 1000:.1f}ms, Recall@10={recall:.3f}")
```

**练习 4：** 实现 Hybrid Search：BM25 关键词搜索 + 向量语义搜索的加权融合。

*参考答案*：两路检索分数各自归一化到 [0,1] 后加权求和重排，兼顾精确词命中与语义相似。

```python
import numpy as np
from rank_bm25 import BM25Okapi

corpus = ["Python 是编程语言", "深度学习使用神经网络", "向量数据库支持语义搜索"]
bm25 = BM25Okapi([doc.split() for doc in corpus])

def hybrid_search(query, query_vec, doc_vecs, alpha=0.5, k=2):
    # 归一化后加权融合 / min-max normalize then weighted sum
    lex = np.array(bm25.get_scores(query.split()))
    sem = doc_vecs @ query_vec                # 已归一化向量的内积=余弦 / cosine via dot
    norm = lambda s: (s - s.min()) / (np.ptp(s) + 1e-9)  # NumPy 2.x 移除 ndarray.ptp，用 np.ptp
    score = alpha * norm(lex) + (1 - alpha) * norm(sem)
    return np.argsort(score)[::-1][:k]        # Top-k 索引 / top-k indices
```
