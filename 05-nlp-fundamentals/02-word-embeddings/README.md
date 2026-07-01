# 02-word-embeddings — 词嵌入

> **所属阶段**：阶段五 · NLP 基础
> **学习目标**：理解词向量的训练原理和语义捕获能力，看清从静态到上下文嵌入的演进
> **预估时长**：4-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [word2vec](./01-word2vec.md) | Word2Vec | 分布式假设；CBOW（上下文→中心词）vs Skip-gram（中心词→上下文）；负采样加速；词类比 `king-man+woman≈queen` |
| 02 | [glove-and-fasttext](./02-glove-and-fasttext.md) | GloVe 与 FastText | GloVe 用全局共现矩阵；FastText 用字符 n-gram 子词解决 OOV 与形态学；静态嵌入的多义词局限 |
| 03 | [contextual-embeddings](./03-contextual-embeddings.md) | 上下文嵌入 | ELMo（BiLSTM 拼接）→ BERT（深层双向自注意力）；同词不同上下文不同向量；Sentence-BERT 语义搜索 |

---

## 🔑 知识点详解

### 01 · Word2Vec（CBOW / Skip-gram）

- **核心概念**：分布式假设——「一个词的含义由它的上下文决定」；用浅层神经网络把词学成稠密向量，语义相近的词在空间中也相近。
- **关键公式/API**：Skip-gram 最大化 `P(context|center)=softmax(v_ctx·v_center)`；Gensim `Word2Vec(sentences, vector_size, window, sg=1, min_count)`，`sg=1` 为 Skip-gram、`sg=0` 为 CBOW；类比查询 `wv.most_similar(positive=[...], negative=[...])`。
- **负采样一句话**：把 V 类 softmax 降为 (1+K) 个 sigmoid 二分类，复杂度从 O(V) 降到 O(K)，提速约百倍；负样本按词频的 **3/4 次方**分布采样。
- **易错点**：① **中文必须先分词**（jieba），词向量的「词」取决于分词结果；② 玩具语料学不出词类比，需足够大且上下文丰富的语料；③ CBOW 快、对高频词好，Skip-gram 慢、对低频词更好，别记反。
- **Java 视角**：嵌入表 ≈ `HashMap<String, float[]>`，按词查稠密向量；区别是向量是训练学出来的，且 `nn.Embedding` 就是这张查找表的可训练版本。
- **前置**：01 模块的文本表示（理解稀疏→稠密的动机）。

### 02 · GloVe 与 FastText

- **核心概念**：GloVe 补 Word2Vec「只看局部窗口」之短，用**全局共现统计**；FastText 补 Word2Vec「无法处理未登录词」之短，用**子词（字符 n-gram）**。
- **关键公式/API**：GloVe 目标 `vec(i)·vec(j) + bᵢ + bⱼ ≈ log(Xᵢⱼ)`（内积拟合共现对数）；FastText `词向量 = 所有字符 n-gram 向量之和`，加 `<`、`>` 边界标记；Gensim `FastText(..., min_n=3, max_n=6)`。
- **易错点**：① 三者在标准相似度基准上**差距很小，无绝对赢家**，别夸大；② FastText 对 OOV 有效的前提是「子词在训练中见过」，全新子词质量会下降；③ Word2Vec/GloVe 遇 OOV 直接 KeyError，只有 FastText 能给未登录词打向量。
- **Java 视角**：FastText 的子词 ≈ 把字符串拆成重叠的定长片段做特征，思路与 BPE 的字节合并同源——这也是它影响现代分词器设计的地方。
- **前置**：01 Word2Vec（GloVe/FastText 是它的两个改进方向）。

### 03 · 上下文嵌入 Contextual Embeddings

- **核心概念**：静态嵌入给 `bank` 永远同一个向量，无法区分「河岸 / 银行」；上下文嵌入让**同一个词在不同句子里产生不同向量**，是从词向量迈向预训练大模型的转折点。
- **关键 API**：`AutoModel.from_pretrained("bert-base-uncased")` → `outputs.last_hidden_state` 是 `[B, T, 768]` 词级向量；`[:, 0, :]` 取 `[CLS]`；句向量优先用 **Mean Pooling**（对非 padding token 按 mask 平均）或直接上 `SentenceTransformer`。
- **易错点**：① 原生 BERT 的 `[CLS]` 没为句相似度优化，直接拿来做相似度**普遍偏弱**，Mean Pooling 通常更好，最佳是 Sentence-BERT；② BERT 向量各向异性、余弦相似度普遍偏高，看**相对差异**而非绝对值；③ ELMo 是双向 LSTM 的**拼接**（信息未真正交互），BERT 是每层完全双向 self-attention（真交互），别混为一谈。
- **Java 视角**：静态嵌入像编译期常量 `final`，上下文嵌入像运行时按上下文动态取值——同一个「变量名」在不同运行环境返回不同结果。
- **前置**：01/02 静态嵌入（理解其多义词局限才懂上下文嵌入的价值）；直接衔接阶段六的注意力。

---

## 🎯 学习要点

- **训练一次 Word2Vec**：用 Gensim 在自己语料上跑 Skip-gram，验证 `most_similar` 与词类比，直观感受「语义方向」的存在。
- **手写负采样损失**：实现 `SkipGramNS` 的 `logsigmoid` 正负样本损失，彻底理解它为何比 softmax 快百倍——这是词向量能规模化的关键。
- **对比 CBOW/Skip-gram**：只改 `sg` 参数，实测速度与低频词相似度差异，把「快 vs 准」的取舍变成肌肉记忆。
- **验证 FastText 处理 OOV**：喂一个训练时没出现的词（如 `learnings`），确认它仍能由子词拼出向量且与 `learning` 高相似——这是静态嵌入做不到的。
- **用 BERT 区分多义词**：提取 `bank` 在金融/河岸多个语境下的向量，算两两余弦相似度，验证同义语境更近、跨义语境更远。
- **搭一个语义搜索**：用 `SentenceTransformer('all-MiniLM-L6-v2')` 编码文档库 + 余弦 top-k，体会「匹配含义而非字面」，为阶段六/RAG 打底。

---

## 🔗 关联

- **本阶段上一模块**：[01-text-preprocessing](../01-text-preprocessing/) — 稀疏表示的局限正是词嵌入要解决的问题。
- **本阶段下一模块**：[03-seq2seq-and-attention](../03-seq2seq-and-attention/) — 词向量作为 Encoder-Decoder 的输入嵌入层。
- **本阶段总览**：[阶段五 README](../README.md)
- **实战延伸**：[agent-course Day-16 Embedding 基础](../../agent-course/Day-16-embedding-basics.md)、[Day-19 Embedding ETL](../../agent-course/Day-19-embedding-etl.md) — 把句向量落地到向量库与 RAG 检索。
