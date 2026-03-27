# 阶段五：自然语言处理基础

> **预估周期**：2-3 周
> **核心目标**：NLP 基础：文本处理、词嵌入、Seq2Seq

---

## 📋 模块大纲

### [01-text-preprocessing](./01-text-preprocessing/) — 文本预处理

NLP 的第一步：将原始文本转化为可计算的数值表示。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [tokenization](./01-text-preprocessing/01-tokenization.md) | 分词（中英文对比） |
| 02 | [text-representation](./01-text-preprocessing/02-text-representation.md) | 文本表示（Bag-of-Words/TF-IDF） |
| 03 | [text-cleaning-pipeline](./01-text-preprocessing/03-text-cleaning-pipeline.md) | 文本清洗流水线 |

---

### [02-word-embeddings](./02-word-embeddings/) — 词嵌入

将离散词汇映射到连续向量空间，捕获语义关系。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [word2vec](./02-word-embeddings/01-word2vec.md) | Word2Vec（CBOW/Skip-gram） |
| 02 | [glove-and-fasttext](./02-word-embeddings/02-glove-and-fasttext.md) | GloVe 与 FastText |
| 03 | [contextual-embeddings](./02-word-embeddings/03-contextual-embeddings.md) | 上下文嵌入（ELMo 概述） |

---

### [03-seq2seq-and-attention](./03-seq2seq-and-attention/) — Seq2Seq 与注意力

序列到序列模型，注意力机制是 Transformer 的前身。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [encoder-decoder](./03-seq2seq-and-attention/01-encoder-decoder.md) | 编码器-解码器架构 |
| 02 | [attention-mechanism](./03-seq2seq-and-attention/02-attention-mechanism.md) | 注意力机制原理 |
| 03 | [machine-translation-practice](./03-seq2seq-and-attention/03-machine-translation-practice.md) | 机器翻译实战 |
