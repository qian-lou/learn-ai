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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
# 完整 RAG Pipeline
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "项目的技术架构是什么？"})
print(f"回答: {result['result']}")
print(f"来源: {result['source_documents'][0].metadata}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 构建一个基于 PDF 文档的 RAG 问答系统。

**练习 2：** 对比 chunk_size=200 和 chunk_size=1000 的检索效果。

### 进阶题

**练习 3：** 实现 Hybrid Search（关键词搜索 + 向量搜索结合）。

**练习 4：** 添加 Re-ranking（用 Cross-Encoder 对检索结果重排序）。
