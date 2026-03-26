# RAG 实战 / RAG Practice

## 1. 背景（Background）
> 构建完整的 RAG 应用：文档解析 → Chunking → Embedding → 检索 → 生成。

## 2-3. 知识点与内容
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 文档分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存储
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 检索 + 生成
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
answer = qa.run("你的问题")
```

## 4-6. 推理/例题/习题
**练习：** 搭建一个基于公司文档的智能问答系统。
