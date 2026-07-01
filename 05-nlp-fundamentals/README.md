# 阶段五：自然语言处理基础

> **预估周期**：2-3 周
> **核心目标**：NLP 基础：文本处理、词嵌入、Seq2Seq 与注意力
> **主线**：稀疏表示 → 稠密静态嵌入 → 上下文嵌入 → 注意力（Transformer 前身）

---

## 🗺️ 本阶段主线

一句话串联：**把文本变成向量，再让向量学会看上下文，最后让模型学会"回头看"。**

```
文本预处理          词嵌入                 Seq2Seq + 注意力
(01)               (02)                   (03)
分词/BoW/TF-IDF  → Word2Vec/GloVe/FastText → Encoder-Decoder → Attention
稀疏、离散          稠密、静态              → 上下文动态          → Transformer 心脏
```

- **01 → 02**：TF-IDF 的稀疏高维暴露了「无语义」的缺陷，催生了稠密词向量。
- **02 → 03**：静态词向量无法区分多义词（`bank` 河岸/银行），催生上下文嵌入与注意力。
- **03 → 阶段六**：注意力机制是 Transformer 的直接前身，学完本阶段即可无缝进入 [阶段六：大模型核心技术](../06-llm-core-technology/)。

---

## 📋 模块大纲

### [01-text-preprocessing](./01-text-preprocessing/) — 文本预处理

NLP 的第一步：将原始文本转化为可计算的数值表示。分词决定模型「词表」，TF-IDF 是理解 Embedding 的统计过渡，清洗质量直接决定下游效果。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [tokenization](./01-text-preprocessing/01-tokenization.md) | 分词（词级/字符级/BPE/WordPiece/SentencePiece） |
| 02 | [text-representation](./01-text-preprocessing/02-text-representation.md) | 文本表示（One-Hot/BoW/TF-IDF/N-gram） |
| 03 | [text-cleaning-pipeline](./01-text-preprocessing/03-text-cleaning-pipeline.md) | 文本清洗流水线（正则/去重/质量过滤） |

---

### [02-word-embeddings](./02-word-embeddings/) — 词嵌入

将离散词汇映射到连续向量空间，捕获语义关系。核心是分布式假设：相似上下文的词有相似向量。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [word2vec](./02-word-embeddings/01-word2vec.md) | Word2Vec（CBOW/Skip-gram/负采样/词类比） |
| 02 | [glove-and-fasttext](./02-word-embeddings/02-glove-and-fasttext.md) | GloVe 全局共现 与 FastText 子词/OOV |
| 03 | [contextual-embeddings](./02-word-embeddings/03-contextual-embeddings.md) | 上下文嵌入（ELMo/BERT、多义词区分） |

---

### [03-seq2seq-and-attention](./03-seq2seq-and-attention/) — Seq2Seq 与注意力

序列到序列模型，注意力机制是 Transformer 的前身。这是本阶段通往大模型的「最后一公里」。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [encoder-decoder](./03-seq2seq-and-attention/01-encoder-decoder.md) | 编码器-解码器架构、Teacher Forcing、信息瓶颈 |
| 02 | [attention-mechanism](./03-seq2seq-and-attention/02-attention-mechanism.md) | Scaled Dot-Product/Multi-Head、QKV、掩码 |
| 03 | [machine-translation-practice](./03-seq2seq-and-attention/03-machine-translation-practice.md) | 中英翻译项目、Beam Search、BLEU 评估 |

---

## 🎯 阶段学习目标

- **能手推**：BPE 合并规则、TF-IDF = TF × IDF、注意力 `softmax(QKᵀ/√dₖ)·V` 三条核心公式。
- **能解释**：为什么静态词向量搞不定多义词，注意力如何解决 Seq2Seq 的信息瓶颈。
- **能落地**：用 Hugging Face `AutoTokenizer`、Gensim `Word2Vec`、`AutoModelForSeq2SeqLM` 跑通分词、词向量、翻译三条链路。
- **能对比**：TF-IDF+LR baseline vs BERT、Greedy vs Beam Search、CBOW vs Skip-gram 的取舍。

---

## 🔗 关联

- **上一阶段**：[阶段四：深度学习基础](../04-deep-learning-basics/) — RNN/LSTM 是本阶段 Seq2Seq 的底座。
- **下一阶段**：[阶段六：大模型核心技术](../06-llm-core-technology/) — 本阶段的注意力直接进化为 Transformer。
- **实战课程**：[agent-course Day-16 Embedding 基础](../agent-course/Day-16-embedding-basics.md)、[Day-18 Chunking](../agent-course/Day-18-chunking.md)、[Day-19 Embedding ETL](../agent-course/Day-19-embedding-etl.md) — 把本阶段的分词/嵌入知识用到 RAG 工程中。
