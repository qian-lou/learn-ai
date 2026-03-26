# 向量数据库 / Vector Databases

## 1. 背景（Background）
> RAG 的核心组件。存储和检索高维向量，支持相似度搜索。主流：Faiss/Chroma/Milvus/Pinecone。

## 2-3. 知识点与内容
```python
# Chroma（轻量级向量数据库）
import chromadb
client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(documents=["Hello world", "AI is amazing"], ids=["1", "2"])
results = collection.query(query_texts=["Hi"], n_results=1)

# Faiss（Facebook 高性能向量检索）
import faiss
import numpy as np
d = 768
index = faiss.IndexFlatL2(d)
vectors = np.random.randn(1000, d).astype('float32')
index.add(vectors)
D, I = index.search(query_vector, k=5)  # Top-5 检索
```

## 4-6. 推理/例题/习题
**练习：** 用 Faiss 构建百万级向量索引，对比 IVF 和 HNSW 的检索速度。
