# RAG 实战 / RAG Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 本节将构建一个**完整的生产级 RAG 应用**——从文档解析到检索增强生成，覆盖 Chunking 策略优化、Embedding 模型选择、检索质量评估和生成质量控制。
>
> 对于 Java 工程师来说，RAG 就是一次"先查后算"：检索器 ≈ 语义版 `Repository`（按相似度而非主键查），LLM ≈ 把查到的上下文组装成答案的 `Service`。

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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # 独立包，替代 community
from langchain_chroma import Chroma                        # 独立包，替代 community
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

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

问题：{input}

回答：""")

# LCEL 组合（替代已废弃的 RetrievalQA）/ Modern LCEL composition
# 注意：prompt 需用 {context} 与 {input} 两个变量
# combine_chain = create_stuff_documents_chain(llm, rag_prompt)
# qa_chain = create_retrieval_chain(
#     vectorstore.as_retriever(search_kwargs={"k": 3}), combine_chain
# )
# result = qa_chain.invoke({"input": "..."})  # -> {"answer": ..., "context": [...]}
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
# reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # v2-m3 取代旧 reranker-base
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
    # 与 3.1 的 create_retrieval_chain 契约一致：输入键 input，输出 answer/context
    result = qa_chain.invoke({"input": message})
    sources = [doc.metadata.get("source", "unknown") for doc in result["context"]]
    return f"{result['answer']}\n\n📎 来源: {', '.join(set(sources))}"

# demo = gr.ChatInterface(rag_chat, title="文档问答助手")
# demo.launch()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 构建一个基于公司内部文档的 RAG 问答系统。

*参考答案*：用 `DirectoryLoader` 批量加载内部文档目录，其余沿用 3.1 节 LCEL 管线。

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 递归加载内部文档目录 / recursively load the internal docs folder
docs = DirectoryLoader("./company_docs", glob="**/*.txt", loader_cls=TextLoader).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vs = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5"),
                           persist_directory="./company_db")

prompt = ChatPromptTemplate.from_template(
    "只依据上下文回答，缺信息就说找不到。\n\n上下文：{context}\n\n问题：{input}")
chain = create_retrieval_chain(
    vs.as_retriever(search_kwargs={"k": 4}),
    create_stuff_documents_chain(ChatOpenAI(model="gpt-4o-mini"), prompt))
print(chain.invoke({"input": "报销流程是什么？"})["answer"])
```

**练习 2：** 对比 Top-3、Top-5、Top-10 不同 K 值对回答质量的影响。

*参考答案*：K 越大召回越全但噪音越多、prompt 越长。固定问题切换 retriever 的 k 观察答案。

```python
question = "报销流程是什么？"
for k in (3, 5, 10):
    chain = create_retrieval_chain(
        vs.as_retriever(search_kwargs={"k": k}),  # 只改 k / vary only k
        create_stuff_documents_chain(ChatOpenAI(model="gpt-4o-mini"), prompt))
    out = chain.invoke({"input": question})
    print(f"[k={k}] 引用 {len(out['context'])} 段 → {out['answer'][:80]}")
# 经验：k 太大引入无关片段反而降质 / too-large k hurts via irrelevant context
```

### 进阶题

**练习 3：** 实现 Hybrid Search + Re-ranking 的高级检索管线。

*参考答案*：先 Hybrid 粗召回，再用 `ContextualCompressionRetriever` 套 Cross-Encoder 精排。

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1) Hybrid 粗召回 / hybrid coarse recall
bm25 = BM25Retriever.from_documents(chunks); bm25.k = 10
hybrid = EnsembleRetriever(
    retrievers=[bm25, vs.as_retriever(search_kwargs={"k": 10})], weights=[0.3, 0.7])

# 2) Cross-Encoder 精排取 Top-3 / rerank to top-3
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"), top_n=3)
pipeline = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=hybrid)
docs = pipeline.invoke("报销流程是什么？")
```

**练习 4：** 用 RAGAS 框架自动评估你的 RAG 系统质量。

*参考答案*：RAGAS 需要 question / answer / contexts /（可选 ground_truth），用 LLM 自动打分四大指标。

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy,
                           context_precision, context_recall)

# 收集每条样本的问题、RAG 答案、检索片段、参考答案
# Collect question / generated answer / retrieved contexts / ground truth
data = Dataset.from_dict({
    "question": ["报销流程是什么？"],
    "answer": ["先在 OA 提交申请，主管审批后财务打款。"],
    "contexts": [["报销需在 OA 系统提交……主管审批……财务打款"]],
    "ground_truth": ["OA 提交→主管审批→财务打款"],
})
# 四指标：忠实度/相关性/上下文精确率/召回率 / four RAGAS metrics
result = evaluate(data, metrics=[faithfulness, answer_relevancy,
                                 context_precision, context_recall])
print(result)
```
