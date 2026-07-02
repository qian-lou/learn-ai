# RAG 基础概念 / RAG Basics

## 1. 背景（Background）

> **为什么要学这个？**
>
> RAG（Retrieval-Augmented Generation，检索增强生成）是**企业落地大模型最常用的方案**。它解决了大模型的两个核心问题：知识过时（训练数据有截止日期）和幻觉（编造事实）。通过检索外部知识库，让模型基于真实数据回答问题。
>
> 对于 Java 工程师来说，RAG 就像是 **Elasticsearch + 业务逻辑层**——先检索相关信息，再基于检索结果生成回答。

## 2. 知识点（Key Concepts）

| 步骤 | 功能 | 工具 |
|------|------|------|
| 文档分割 | 切分长文档为段落 | RecursiveCharacterTextSplitter |
| 向量化 | 文本 → 向量 | Sentence-BERT, BGE |
| 向量存储 | 存储和检索向量 | Faiss, Chroma, Milvus |
| 检索 | 相似度搜索 Top-K | 余弦相似度 / L2 距离 |
| 生成 | 基于检索结果回答 | LLM (GPT/Qwen) |

## 3. 内容（Content）

### 3.1 RAG 工作流程

```
RAG 五步流程：

1. 文档加载 (Loading)
   PDF/Word/HTML/Markdown → 纯文本

2. 文档分割 (Chunking)
   长文档 → 500-1000 字符的段落（chunk）
   保留上下文：overlap=100 字符

3. 向量化 (Embedding)
   每个 chunk → 768/1024 维向量
   模型：BAAI/bge-base-zh, text-embedding-ada-002

4. 检索 (Retrieval)
   用户问题 → embedding → 在向量库中搜索 Top-K 相关 chunk

5. 生成 (Generation)
   将检索到的 chunk 拼入 prompt → LLM 生成回答

Prompt 模板：
  "根据以下参考资料回答用户问题。
   如果参考资料中没有相关信息，请说'我不确定'。
   
   参考资料：{retrieved_chunks}
   
   问题：{user_question}"
```

### 3.2 基础 RAG 实现

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 独立包，替代 community
from langchain_chroma import Chroma                       # 独立包，替代 community

# ============================================================
# Step 1-2: 加载并分割文档
# ============================================================
loader = PyPDFLoader("document.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个 chunk 最大字符数
    chunk_overlap=50,    # 重叠字符数（保留上下文）
    separators=["\n\n", "\n", "。", ".", " "],
)
chunks = splitter.split_documents(docs)
print(f"文档切分为 {len(chunks)} 个 chunks")

# ============================================================
# Step 3: 向量化并存储
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# ============================================================
# Step 4: 检索
# ============================================================
query = "这份文档的主要内容是什么？"
results = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:200])
```

### 3.3 RAG vs 微调

```
RAG:
  ✅ 知识实时更新（修改文档即可）
  ✅ 不需要训练（零计算成本）
  ✅ 可追溯来源（引用文档）
  ✅ 适合企业知识库、文档问答
  ❌ 受限于检索质量
  ❌ 上下文窗口有限

微调:
  ✅ 知识内化到模型（不需要检索）
  ✅ 适合特定领域风格/格式
  ❌ 更新知识需要重新训练
  ❌ 需要高质量标注数据
  ❌ 有训练成本

最佳实践: RAG + 微调结合
  微调: 教模型领域术语和回答风格
  RAG:  提供最新的事实性知识
```

### 3.4 2024-2025 RAG 进展：现代 embedding · reranker · agentic RAG

```python
# ============================================================
# (A) 现代 embedding：bge-base-zh-v1.5 已被取代
#     Modern embeddings (bge-base-zh-v1.5 is superseded)
#   开源: BAAI/bge-m3（多语言/多粒度长文本/支持 dense+sparse+多向量混合检索）、
#         Qwen3-Embedding（多语言、可指令化）。
#   商用 API: voyage-3 / voyage-3-large、OpenAI text-embedding-3-large。
# ============================================================
from langchain_huggingface import HuggingFaceEmbeddings
# bge-m3 一模型即可同时做语义(dense)+词面(sparse)混合检索
# bge-m3: one model for hybrid dense + sparse retrieval
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# ============================================================
# (B) 二阶段精排 reranker：先粗召回，再用 cross-encoder 精排
#     Two-stage rerank: coarse recall → cross-encoder fine ranking
#   bge-reranker-v2-m3 同时编码 (query, doc)，精度远高于双塔召回。
# ============================================================
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # v2-m3 取代旧 reranker-base
query = "项目的技术架构是什么？"
candidates = vectorstore.similarity_search(query, k=20)        # 粗召回 / recall
scores = reranker.predict([(query, d.page_content) for d in candidates])
top3 = [d for _, d in sorted(zip(scores, candidates),
                             key=lambda x: x[0], reverse=True)][:3]  # 精排取 3

# ============================================================
# (C) 更前沿的范式 / Frontier paradigms
# ------------------------------------------------------------
# 1) Contextual Retrieval（Anthropic, 2024）：入库前给每个 chunk 拼接
#    "该 chunk 在整篇文档中的位置/作用"上下文，再 embedding，显著降低召回失败率。
#    Prepend doc-level context to each chunk before embedding.
def contextualize(doc_summary: str, chunk: str) -> str:
    """给 chunk 注入文档级上下文 / inject document-level context."""
    return f"文档背景：{doc_summary}\n该片段内容：{chunk}"

# 2) GraphRAG / RAPTOR：先把语料聚类成"层级摘要树/知识图谱"，
#    回答全局型问题（"总结整本书主题"）时检索高层摘要而非零散 chunk。
# 3) Agentic / multi-hop RAG：用 LangGraph 让 Agent 自主决定
#    "要不要检索、检索什么、是否需要再检索一轮"，而非固定一次 retrieve。
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def retrieve(q: str) -> str:
    """按需检索知识库 / retrieve from KB on demand."""
    docs = vectorstore.similarity_search(q, k=3)
    return "\n".join(d.page_content for d in docs)

# Agent 自行多跳检索，适合需要分解的复杂问题 / autonomous multi-hop retrieval
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[retrieve])
```

## 4. 详细推理（Deep Dive）

### 4.1 Chunking 策略

```
Chunk 大小的权衡：

太小 (100字): 缺少上下文，检索结果零碎
太大 (2000字): 噪音多，LLM 难以聚焦关键信息
推荐 (300-800字): 信息完整，噪音适中

高级分割策略：
  - 按语义分割（而非固定长度）
  - 按标题/章节分割
  - 使用 LLM 辅助分割
```

## 5. 例题（Worked Examples）

```python
# 完整 RAG Pipeline（LCEL，替代已废弃的 RetrievalQA）
# Modern RAG via LCEL (replaces deprecated RetrievalQA)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{input}"
)
combine_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(
    vectorstore.as_retriever(search_kwargs={"k": 3}), combine_chain
)

result = qa_chain.invoke({"input": "项目的技术架构是什么？"})
print(f"回答: {result['answer']}")
print(f"来源: {result['context'][0].metadata}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 构建一个基于 PDF 文档的 RAG 问答系统。

*参考答案*：组合 3.2 的检索与 5 节的 LCEL 生成链，得到端到端问答。

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 独立包 / standalone pkg
from langchain_chroma import Chroma                       # 独立包 / standalone pkg
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\
    .split_documents(PyPDFLoader("document.pdf").load())
vs = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5"))

prompt = ChatPromptTemplate.from_template("根据上下文回答。\n\n上下文：{context}\n\n问题：{input}")
# LCEL 检索链，替代已废弃的 RetrievalQA / modern chain replacing deprecated RetrievalQA
chain = create_retrieval_chain(
    vs.as_retriever(search_kwargs={"k": 3}),
    create_stuff_documents_chain(ChatOpenAI(model="gpt-4o-mini"), prompt))
print(chain.invoke({"input": "文档的主要内容是什么？"})["answer"])
```

**练习 2：** 对比 chunk_size=200 和 chunk_size=1000 的检索效果。

*参考答案*：用同一文档、同一 query 切两套向量库观察召回片段。chunk 小则片段零碎、上下文不足；chunk 大则单片信息全但易混入噪音。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
query = "项目的技术架构是什么？"
for size in (200, 1000):
    chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)\
        .split_documents(docs)  # docs 为已加载文档 / preloaded docs
    # 每套独立 collection，避免互相污染 / separate collections to avoid cross-contamination
    vs = Chroma.from_documents(chunks, emb, collection_name=f"c{size}")
    top = vs.similarity_search(query, k=3)
    print(f"[chunk_size={size}] 命中片段长度 {[len(d.page_content) for d in top]}")
```

### 进阶题

**练习 3：** 实现 Hybrid Search（关键词搜索 + 向量搜索结合）。

*参考答案*：用 `EnsembleRetriever` 加权融合 BM25（关键词）与向量检索，兼顾精确匹配与语义召回。

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks)   # 关键词 / lexical
bm25.k = 5
vector = vs.as_retriever(search_kwargs={"k": 5})  # 语义 / semantic
# 加权融合：关键词 30% + 语义 70% / weighted fusion
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.3, 0.7])
docs = hybrid.invoke("项目的技术架构是什么？")
```

**练习 4：** 添加 Re-ranking（用 Cross-Encoder 对检索结果重排序）。

*参考答案*：先粗召回 Top-20，再用 Cross-Encoder 对 (query, doc) 打分精排取 Top-3。Cross-Encoder 同时编码两句话，精度高于双塔召回。

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # v2-m3 取代旧 reranker-base，与 3.4 一致
query = "项目的技术架构是什么？"
candidates = vs.similarity_search(query, k=20)            # 粗召回 / coarse recall

# 对每个 (query, doc) 打相关性分 / score each (query, doc) pair
scores = reranker.predict([(query, d.page_content) for d in candidates])
ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
top3 = [doc for _, doc in ranked[:3]]                     # 精排取前 3 / keep top-3
```
