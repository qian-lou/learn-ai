# 03-seq2seq-and-attention — Seq2Seq 与注意力

> **所属阶段**：阶段五 · NLP 基础
> **学习目标**：理解序列到序列架构和注意力机制，打通通往 Transformer 的「最后一公里」
> **预估时长**：5-7 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [encoder-decoder](./01-encoder-decoder.md) | 编码器-解码器架构 | RNN Seq2Seq、context vector 信息瓶颈、Teacher Forcing 与 Exposure Bias；Encoder-only/Decoder-only/Encoder-Decoder 三种变体 |
| 02 | [attention-mechanism](./02-attention-mechanism.md) | 注意力机制原理 | QKV 三元组、Scaled Dot-Product、Multi-Head、Self/Cross/Causal 三种模式与掩码 |
| 03 | [machine-translation-practice](./03-machine-translation-practice.md) | 机器翻译实战 | opus-mt/NLLB 预训练翻译、Greedy vs Beam Search、BLEU 评估（中文需分词） |

---

## 🔑 知识点详解

### 01 · 编码器-解码器架构 Encoder-Decoder

- **核心概念**：Encoder 把源序列压成上下文表示，Decoder 据此逐 token 生成目标序列——几乎所有生成任务（翻译/摘要/问答）的通用框架。
- **关键机制**：**Teacher Forcing**——训练时以 `teacher_forcing_ratio` 概率用真实目标 `yₜ₋₁` 而非模型预测作为下一步输入，收敛更快更稳。
- **两大坑**：① **信息瓶颈**——整个输入被压成一个固定大小 context vector，序列越长信息损失越重（这正是 Attention 要解决的问题）；② **Exposure Bias**——训练只见真实前缀、推理只能喂自身预测，一步错则误差沿序列累积；缓解靠 Scheduled Sampling（ratio 从高退火到低），且**最终评估必须在 `ratio=0` 全自回归下进行**，否则高估模型。
- **易错点**：损失要**错位一位**（用 `tgt[:,1:]` 当标签，因第 0 位是 BOS 输入而非预测目标），并用 `ignore_index=PAD` 跳过填充位；Decoder 必须单向（不能看未来），即使 Encoder 用双向 LSTM。
- **Java 视角**：Encoder-Decoder ≈ **编解码器模式**——Encoder 是 Serializer 把请求编成中间态，Decoder 是 Deserializer 逐步解出响应。
- **前置**：[阶段四 · RNN/LSTM](../../04-deep-learning-basics/04-rnn/)、02 模块的词嵌入（作为输入层）。

### 02 · 注意力机制 Attention

- **核心概念**：让模型处理每个位置时都能**动态关注输入的任意部分**，打破 Seq2Seq 的固定 context vector 瓶颈；这是整个大模型时代的基石。
- **核心公式**：`Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V`。Q=「我想查什么」、K=「有哪些索引」、V=「实际数据」。
- **为什么除以 √dₖ**：当元素独立服从 N(0,1) 时 `Var(q·k)=dₖ`，dₖ 大则 score 方差大、softmax 落入饱和区梯度趋零；除以 √dₖ 把方差拉回 1。
- **三种模式**：Self（Q=K=V，BERT/GPT 内部）、Cross（Q 来自 Decoder、K/V 来自 Encoder，翻译）、Causal（下三角掩码把 `j>i` 置 `-inf`，保证 GPT 自回归只看左边）。
- **易错点**：① 多头**并非越多越好**——`d_model` 固定时头越多每头维度 `d_head=d_model/n_heads` 越小，过小反而降表达力（BERT-base 用 12 头、d_head=64 是平衡点）；② 掩码在 softmax **之前**用 `-inf` 填充，不是之后置零；③ 无论几头，输出维度和总参数量都不变。
- **Java 视角**：Attention ≈ **动态查询系统**——Q 是 SQL 查询，K 是索引字段，V 是返回的行，按 Q-K 相关度加权聚合 V。
- **前置**：01 编码器-解码器（注意力最初就是加在 RNN Seq2Seq 上的）。

### 03 · 机器翻译实战 Machine Translation

- **核心概念**：翻译是 Transformer 诞生的验证舞台（"Attention Is All You Need" 即在翻译上验证），一个项目串联分词→嵌入→Encoder-Decoder→注意力→生成的全链路。
- **关键 API**：`pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")`；或 `AutoModelForSeq2SeqLM` + `model.generate(**inputs, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=..., early_stopping=True)`。
- **解码策略**：Greedy（每步取最大，`num_beams=1`）只看局部最优；Beam Search（保留 top-K 候选路径）更接近全局最优、更流畅，常用 `num_beams=4~5`，再大收益递减且偏好平淡短译文。
- **BLEU 一句话**：n-gram 精确率的几何平均 + 短句惩罚（BP），`∈[0,1]`（论文常 ×100 报告）；人工翻译约 40-60，现代 NMT 约 30-50——天花板不高是因为一句话有多种正确译法。
- **易错点**：① **中文算 BLEU 前必须分词**（jieba），否则 n-gram 统计无意义；② 用**语料级** `sacrebleu.corpus_bleu` 而非句子级平均，结果才可复现；③ 生成重复（"the the the"）用 `repetition_penalty` / `no_repeat_ngram_size` 抑制。
- **Java 视角**：一次翻译 ≈ 一条「编码请求 → 解码响应」管道，Encoder 序列化源句、Decoder 逐 token 反序列化目标句。
- **前置**：01 编码器-解码器 + 02 注意力（翻译是二者的综合落地）。

---

## 🎯 学习要点

- **跑通 toy Seq2Seq**：用 3.2 节的 `Encoder/Decoder/Seq2Seq` 做序列反转（`[1,2,3]→[3,2,1]`），亲手体会 context vector 的承载极限与损失错位一位的细节。
- **手写 Scaled Dot-Product Attention**：实现并验证输出形状 `[B,T_q,d_v]`、权重 `[B,T_q,T_k]` 且沿 `T_k` 求和为 1——这是理解 Transformer 的最小闭环。
- **实现 Causal Mask**：用 `torch.tril` 造下三角掩码，讲清它如何让 GPT「训练时一次并行、等价于推理时逐 token 自回归」。
- **对比解码策略**：同句子只改 `num_beams`（1 vs 4），对比译文质量与耗时，把「局部最优 vs 全局最优」的取舍看在眼里。
- **正确算一次 BLEU**：中文分词 + `sacrebleu.corpus_bleu` 评估 opus-mt，理解 BLEU 为何只是近似指标、需配人工/COMET。
- **可视化注意力**：用 `output_attentions=True` 取 BERT 各层各头的 `[B,n_heads,T,T]` 权重画热力图，观察「浅层偏局部、深层偏语义」的可解释模式。

---

## 🔗 关联

- **本阶段上一模块**：[02-word-embeddings](../02-word-embeddings/) — 词向量是 Encoder-Decoder 的输入嵌入层。
- **下一阶段**：[阶段六 · Transformer](../../06-llm-core-technology/01-transformer/) — 本模块的 Multi-Head Attention 直接组装成 Transformer。
- **本阶段总览**：[阶段五 README](../README.md)
- **实战延伸**：[agent-course Day-49 Eval 入门](../../agent-course/Day-49-eval-intro.md)、[Day-50 编写 Evals](../../agent-course/Day-50-writing-evals.md) — BLEU 之外，生成质量评估在 Agent 工程中的延续。
