# 02-word-embeddings — 词嵌入

> **所属阶段**：阶段五 · NLP 基础
> **学习目标**：理解词向量的训练原理和语义捕获能力

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [word2vec](./01-word2vec.md) | Word2Vec | CBOW 与 Skip-gram、负采样、词类比 |
| 02 | [glove-and-fasttext](./02-glove-and-fasttext.md) | GloVe 与 FastText | 共现矩阵分解、子词嵌入、OOV 处理 |
| 03 | [contextual-embeddings](./03-contextual-embeddings.md) | 上下文嵌入 | ELMo 双向 LSTM、动态词向量 |

---

## 🎯 学习要点

- Word2Vec 的分布式假设：相似上下文的词有相似向量
- FastText 的子词方法解决了 OOV（未登录词）问题
- 上下文嵌入是通往 BERT/GPT 的桥梁
