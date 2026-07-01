# 03-rag — 检索增强生成（RAG）

> **所属阶段**：阶段七 · 大模型应用实战
> **学习目标**：掌握"先检索、后生成"的 RAG 架构，用外部知识库解决大模型知识过时与幻觉问题
> **预估时长**：5-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [rag-basics](./01-rag-basics.md) | RAG 基础原理 | 加载→切分→嵌入→检索→生成五步流程、Chunking 策略、RAG vs 微调、现代 embedding/reranker、Contextual Retrieval 与 agentic RAG |
| 02 | [vector-databases](./02-vector-databases.md) | 向量数据库 | Faiss/Chroma/Milvus/Qdrant/Weaviate/LanceDB、Flat/IVF/HNSW/PQ 四类索引、余弦/L2/内积三种相似度度量 |
| 03 | [rag-practice](./03-rag-practice.md) | RAG 实战项目 | 端到端 LCEL 管线、Hybrid Search + Re-ranking、RAGAS 四维评估、检索失败/幻觉排障清单 |

---

## 🔑 知识点详解

### 01 · RAG 基础原理

- **核心概念**：RAG = 先用向量相似度从知识库检索相关片段，再把片段拼进 prompt 让模型基于真实资料作答；解决了大模型"知识有截止日期"和"编造事实"两大痛点。
- **关键流程/公式**：五步 = 加载 → 切分（chunk_size 300-800 字符 + overlap 50-100）→ 嵌入（文本→768/1024 维向量）→ 检索（Top-K 相似度）→ 生成。生成 prompt 模板核心是"根据参考资料回答，没有就说不知道"。
- **易错点**：
  - chunk 太小→片段零碎缺上下文；太大→噪音多、模型抓不住重点。
  - 检索得到不等于模型会用——需在 prompt 里强调"仅基于参考资料"并适当降 Top-K。
- **Java 视角**：RAG ≈ Elasticsearch（语义版）+ 业务逻辑层——检索器 ≈ 按相似度查的 `Repository`，LLM ≈ 把上下文组装成答案的 `Service`。
- **前置**：模块 02（生成 prompt）、02-向量数据库（存检索底座）。

### 02 · 向量数据库

- **核心概念**：存储高维向量并做毫秒级 ANN（近似最近邻）检索的基础设施；百万级以上文档暴力搜索不可行，靠索引换速度。
- **关键选型/度量**：
  - 索引：Flat（100% 精确但 O(N) 慢，<10 万）→ IVF（倒排，百万级）→ **HNSW（O(log N)，~99% 精度，百万-千万级，推荐）** → PQ（乘积量化，压缩 32×，亿级换空间）。
  - 相似度：**余弦相似度** `sim=a·b/(|a||b|)`（文本语义首选）、L2 欧氏距离、内积（向量归一化后≈余弦）。
  - 库选型：Chroma（轻量开发）、Faiss（极致性能库）、Milvus（分布式生产）；2024-2025 增长最快为 Qdrant（Rust+强过滤）、Weaviate（内置混合检索）、LanceDB（嵌入式列存/多模态）。
- **易错点**：
  - 用内积度量却没归一化向量，结果失真——要么归一化，要么明确用余弦。
  - IVF 的 `nprobe` 是精度-速度旋钮：太小召回不足，太大接近暴力搜索。
- **Java 视角**：向量库 ≈ Elasticsearch 的"语义版"——ES 做关键词倒排，向量库做语义 ANN；HNSW ≈ 跳表思想的高维推广。
- **前置**：01（知道要存/检索什么）。

### 03 · RAG 实战与优化

- **核心概念**：把基础 RAG 升级到生产级——在切分、检索、生成、评估四个维度分别优化，并能定位"检索不到/用不上/幻觉"三类典型问题。
- **关键手段**：
  - **Hybrid Search**：`EnsembleRetriever` 加权融合 BM25（关键词精确）+ 向量（语义），典型权重 0.3/0.7。
  - **Re-ranking**：先粗召回 Top-20，再用 Cross-Encoder（如 `bge-reranker-v2-m3`）对 (query, doc) 精排取 Top-3——同时编码两句，精度远高于双塔召回。
  - **RAGAS 四指标**：Faithfulness（忠实度/是否编造）、Answer Relevancy（回答相关性）、Context Precision（检索精确率）、Context Recall（检索召回率）。
- **易错点**：
  - Top-K 一味调大反而降质——无关片段既是噪音又挤占上下文窗口。
  - 用已废弃的 `RetrievalQA`；现代写法是 `create_retrieval_chain` + `create_stuff_documents_chain`（LCEL），返回 `{"answer", "context"}`。
- **Java 视角**：Hybrid Search ≈ 组合"精确匹配 + 模糊语义"两路查询再融合排序；RAGAS ≈ 给检索/生成各环节写的自动化单元测试。
- **前置**：01、02。

---

## 🎯 学习要点

- **检索质量是 RAG 的天花板**：切分、嵌入模型、Top-K、reranker 任何一环拖后腿，生成再强也救不回——优化优先级永远是"先检索、后生成"。
- **先跑通最小闭环，再逐项加料**：PDF→切分→Chroma→LCEL 检索链先端到端跑起来，再依次引入 Hybrid、Re-ranking、Contextual Retrieval。
- **中文场景选对嵌入模型**：`BAAI/bge-*` 系列/`bge-m3`（支持 dense+sparse 混合）是中文 RAG 常用起点；嵌入时开 `normalize_embeddings=True` 配余弦。
- **索引按规模选**：<10 万用 Flat 图省事，百万-千万上 HNSW，亿级且显存吃紧才考虑 PQ；先用 `nprobe`/精度-速度权衡跑个基准。
- **一定要做评估**：接入 RAGAS 或至少人工抽检四个维度，别凭"看起来对"上线；检索命中率和忠实度要能量化。
- **进阶范式了解即可**：Contextual Retrieval（入库前给 chunk 拼文档级上下文）、GraphRAG/RAPTOR（层级摘要答全局问题）、agentic RAG（Agent 自主决定检索几轮）是当前前沿，按需引入。

---

## 🔗 关联

- **上一模块**：[02-prompt-engineering](../02-prompt-engineering/) — RAG 生成环节就是一段精心设计的 prompt。
- **下一模块**：[04-fine-tuning](../04-fine-tuning/) — RAG 供事实、微调供风格，两者互补；对比二者取舍见 01 章。
- **本阶段总览**：[阶段七 README](../README.md)
- **相关 Day**：[Day 16 Embedding 与余弦相似度](../../agent-course/Day-16-embedding-basics.md) · [Day 18 文档切分](../../agent-course/Day-18-chunking.md) · [Day 21 完整 RAG 链路](../../agent-course/Day-21-full-rag-chain.md) · [Day 23 进阶检索（混合+rerank）](../../agent-course/Day-23-advanced-retrieval.md) · [Day 25 文档问答 Capstone](../../agent-course/Day-25-rag-capstone.md)。
