# RAG 实战 / RAG Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 本节将构建一个**完整的生产级 RAG 应用**——从文档解析到检索增强生成，覆盖 Chunking 策略优化、Embedding 模型选择、检索质量评估和生成质量控制。

## 2. 知识点（Key Concepts）

| 优化维度 | 基础方案 | 进阶方案 |
|----------|---------|---------|
| 分割 | 固定长度 | 语义分割 |
| 检索 | 向量搜索 | Hybrid + Re-ranking |
| 生成 | 简单拼接 | Prompt 工程 + 引用 |
| 评估 | 人工评判 | RAGAS 自动评估 |

## 3. 内容（Content）

### 3.1 完整 RAG 应用

```python
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============================================================
# Step 1: 加载多种格式文档
# ============================================================
# loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
# docs = loader.load()

# ============================================================
# Step 2: 智能分割
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", ".", "！", "？", " "],
    length_function=len,
)
# chunks = splitter.split_documents(docs)

# ============================================================
# Step 3: 向量化（推荐中文模型）
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# ============================================================
# Step 4: 自定义 RAG Prompt
# ============================================================
rag_prompt = PromptTemplate.from_template("""
你是一个专业的文档问答助手。根据以下参考资料回答问题。

规则：
1. 只基于参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，明确说"根据提供的文档，我无法找到相关信息"
3. 在回答中标注信息来源

参考资料：
{context}

问题：{question}

回答：""")

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#     chain_type_kwargs={"prompt": rag_prompt},
#     return_source_documents=True,
# )
```

### 3.2 检索优化策略

```python
# ============================================================
# 1. Hybrid Search（关键词 + 语义混合搜索）
# ============================================================
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# bm25 = BM25Retriever.from_documents(chunks)
# vector_retriever = vectorstore.as_retriever()
# hybrid = EnsembleRetriever(
#     retrievers=[bm25, vector_retriever],
#     weights=[0.3, 0.7],  # 关键词 30% + 语义 70%
# )


# ============================================================
# 2. Re-ranking（重排序）
# ============================================================
# from sentence_transformers import CrossEncoder
# reranker = CrossEncoder("BAAI/bge-reranker-base")
# 
# 检索 Top-20 → Re-rank → 取 Top-3
# scores = reranker.predict([(query, doc) for doc in candidates])
```

### 3.3 RAG 评估

```
RAG 评估四个维度 (RAGAS 框架):

1. Faithfulness (忠实度):
   回答是否基于检索到的内容？有没有编造？

2. Answer Relevancy (回答相关性):
   回答是否与问题相关？

3. Context Precision (上下文精确率):
   检索到的内容是否包含答案？

4. Context Recall (上下文召回率):
   所有相关信息是否都被检索到了？
```

## 4. 详细推理（Deep Dive）

### 4.1 RAG 常见问题与优化

```
问题 1: 检索不到相关内容
  → 优化 Embedding 模型（换 BGE/E5）
  → 调整 chunk_size
  → 添加元数据过滤

问题 2: 检索到但 LLM 没用上
  → 优化 Prompt（强调"基于参考资料"）
  → 减少 Top-K（避免噪音）
  → 使用 Re-ranking

问题 3: LLM 幻觉（编造内容）
  → 添加"如果不知道就说不知道"
  → 要求引用来源
  → 降低 temperature
```

## 5. 例题（Worked Examples）

```python
# 用 Gradio 构建 RAG 聊天界面
import gradio as gr

def rag_chat(message, history):
    result = qa_chain.invoke({"query": message})
    sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
    return f"{result['result']}\n\n📎 来源: {', '.join(set(sources))}"

# demo = gr.ChatInterface(rag_chat, title="文档问答助手")
# demo.launch()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 构建一个基于公司内部文档的 RAG 问答系统。

**练习 2：** 对比 Top-3、Top-5、Top-10 不同 K 值对回答质量的影响。

### 进阶题

**练习 3：** 实现 Hybrid Search + Re-ranking 的高级检索管线。

**练习 4：** 用 RAGAS 框架自动评估你的 RAG 系统质量。
