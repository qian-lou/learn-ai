# 01-text-preprocessing — 文本预处理

> **所属阶段**：阶段五 · NLP 基础
> **学习目标**：掌握文本数据的清洗、分词和数值化表示，理解为什么现代大模型都用子词分词
> **预估时长**：4-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [tokenization](./01-tokenization.md) | 分词 | 词级/字符级/子词分词演进；BPE 训练与推理、WordPiece（BERT）、SentencePiece（LLaMA/T5）；特殊 Token 与 OOV 处理 |
| 02 | [text-representation](./02-text-representation.md) | 文本表示 | One-Hot → BoW → TF-IDF → 稠密向量的演进；N-gram 保留局部词序；TF-IDF+LR baseline |
| 03 | [text-cleaning-pipeline](./03-text-cleaning-pipeline.md) | 文本清洗流水线 | 正则清洗、Unicode 标准化、去重（MinHash/LSH）、质量启发式过滤；传统清洗 vs 大模型清洗 |

---

## 🔑 知识点详解

### 01 · 分词 Tokenization

- **核心概念**：把原始文本切成模型可处理的 Token 序列；分词器决定词表大小、输入粒度和 OOV 处理方式。
- **关键 API**：`AutoTokenizer.from_pretrained(name)` → `tokenizer(texts, padding=True, truncation=True, return_tensors="pt")` 一步得到 `input_ids` + `attention_mask`。
- **BPE 一句话**：反复合并语料中最高频的相邻字符对，直到词表达到目标大小；因保留字节级基础字符，**OOV 零失败率**。
- **易错点**：① 批量输入不加 `padding=True` 会因长度不齐无法组 batch；② WordPiece 用 `##` 前缀表示词内延续（`token`→`##ization`），SentencePiece 用 `▁` 表示空格——二者对空格/缩进的可逆性不同，代码生成场景务必用 SentencePiece 这类无损分词。
- **Java 视角**：分词 ≈ 编译器的**词法分析（Lexical Analysis）**，把字符流切成有意义的 Token；子词词表 ≈ 一张学出来的合并规则表。
- **前置**：无（本阶段起点）。

### 02 · 文本表示 Text Representation

- **核心概念**：计算机只能算数值，必须把文本变成向量；这条演进线（One-Hot→BoW→TF-IDF→Word2Vec→BERT）就是 NLP 对「语义」理解不断加深的历史。
- **关键公式**：`TF-IDF(t,d) = TF(t,d) × log(N / df(t))` —— 在本文频繁、在全局稀有的词得分最高。
- **关键 API**：`CountVectorizer` / `TfidfVectorizer(ngram_range=(1,2), max_features=..., sublinear_tf=True)` → `.fit_transform` / `.transform`。
- **易错点**：① 纯 BoW/unigram **完全丢词序**（"not good" ≡ "good not"），需靠 bigram 找回局部顺序；② 查询/测试集必须用训练集 `fit` 出的同一 vectorizer 做 `transform`，否则词项空间不一致。
- **Java 视角**：文本表示 ≈ **对象序列化**；不同方法 ≈ 不同序列化协议（One-Hot 像超大稀疏枚举，TF-IDF 像带权重的字段，稠密向量像压缩后的二进制）。
- **前置**：01 分词（词表决定向量每一维对应哪个词）。

### 03 · 文本清洗流水线 Text Cleaning Pipeline

- **核心概念**：Garbage in, garbage out —— 数据质量直接决定模型质量；大模型时代数据工程的重要性甚至超过模型架构。
- **关键 API**：`re.sub` 系列 + `unicodedata.normalize('NFKC', text)`（全角转半角/兼容字符归一）；去重用 MinHash 签名估计 Jaccard 相似度，大规模再配 LSH 分桶把 O(N²) 降到近似 O(N)。
- **易错点**：① **传统 NLP 清洗 ≠ 大模型清洗**——大模型**不去停用词、不做词形还原、保留标点**，因为模型要完整语法；② 清洗不是越狠越好，删掉否定词/标点可能抹掉判别信号，应以验证集指标做消融。
- **Java 视角**：清洗 ≈ ETL 里的 **Transform** 阶段，做标准化、去噪、过滤后再 Load 进下游。
- **前置**：01、02（清洗结果最终服务于分词与表示）。

---

## 🎯 学习要点

- **动手一遍 BPE**：手推 3.2 节的 `learn_bpe`，观察高频对如何逐步合并——理解「为什么现代大模型都用子词」，这是本模块最该带走的一条。
- **背下 TF-IDF 公式**：`TF × IDF` 是从「词频统计」通往「语义嵌入」的过渡桥梁，也是解释「词重要性」的最直观模型。
- **跑通 baseline**：`TfidfVectorizer + LogisticRegression` 在 20 Newsgroups/IMDB 上做一遍分类，记住「任何文本任务先跑这个 baseline」。
- **理解 `attention_mask` 的来历**：padding 补齐后必须靠 mask 告诉模型哪些是填充位——这个概念会一路用到 Transformer。
- **辨析可逆分词**：亲手对比 BERT（WordPiece，`##`）与 LLaMA（SentencePiece，`▁`）对同一段带缩进代码的 encode→decode，体会代码 LLM 为何偏爱无损分词。
- **建一条清洗流水线**：把正则清洗 + 长度/词数过滤 + MD5 去重 + 质量启发式串成一个 `TextCleaningPipeline`，并统计过滤前后留存率。

---

## 🔗 关联

- **本阶段下一模块**：[02-word-embeddings](../02-word-embeddings/) — 把稀疏的 TF-IDF 升级为稠密的语义向量。
- **上游依赖**：[阶段四 · RNN](../../04-deep-learning-basics/04-rnn/) — 序列建模的底座。
- **本阶段总览**：[阶段五 README](../README.md)
- **实战延伸**：[agent-course Day-18 Chunking](../../agent-course/Day-18-chunking.md) — 分词/切分思想在 RAG 文档切块中的工程应用。
