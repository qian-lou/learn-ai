# RAG 基础概念 / RAG Basics

## 1. 背景（Background）
> RAG (检索增强生成) 结合检索和生成，让大模型能访问外部知识，是企业落地大模型最常用的方案。

## 2-3. 知识点与内容
```
RAG 工作流程：
1. 文档分割(Chunking) → 将文档切分为段落
2. 向量化(Embedding) → 用 Embedding 模型编码为向量
3. 向量存储(Vector Store) → 存入 Faiss/Chroma/Milvus
4. 检索(Retrieval) → 用户问题 → 向量检索 Top-K 相关段落
5. 生成(Generation) → 将检索到的段落 + 问题 → LLM 生成回答

RAG vs 微调：
RAG: 知识可实时更新，不需要训练，成本低
微调: 知识嵌入模型参数，适合特定领域风格
最佳实践: RAG + 微调 结合使用
```

## 4-6. 推理/例题/习题
**练习：** 用 LangChain 实现一个基于本地文档的问答系统。
