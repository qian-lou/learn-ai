# 02-pretrained-models — 预训练模型

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：掌握主流预训练模型的架构差异、演进脉络与缩放规律，能做技术选型
> **预估时长**：4-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [bert](./01-bert.md) | BERT 详解 | Encoder-only 双向、MLM(15% 遮盖策略)+NSP、`[CLS]` 句向量、`2e-5` 小学习率微调、各层探针 |
| 02 | [gpt-series](./02-gpt-series.md) | GPT 系列 | GPT-1/2/3/4 演进、自回归 `P(x)=∏P(xᵢ\|x<ᵢ)`、In-Context Learning、涌现能力、temperature 采样 |
| 03 | [t5-and-others](./03-t5-and-others.md) | T5 及开源生态 | Text-to-Text 统一范式、FLAN-T5、LLaMA/Qwen/Mistral/DeepSeek、MoE、Decoder-only 为何胜出 |
| 04 | [scaling-laws](./04-scaling-laws.md) | 缩放定律 | Kaplan vs Chinchilla、`D≈20N` 经验法则、`C=6ND` 成本估算、过度训练与涌现的经济学 |

---

## 🔑 知识点详解

### 01 · BERT
- **核心概念**：Encoder-only 双向模型，开创「预训练 + 微调」范式；每个 token 都能看到左右完整上下文，擅长理解类任务。
- **关键 API**：`AutoModel`/`AutoModelForSequenceClassification`，取 `outputs.last_hidden_state[:,0,:]` 即 `[CLS]` 向量做句子表示；微调用 `AdamW(lr=2e-5)` 量级。
- **易错点**：① MLM 的 15% 遮盖里只有 80% 换 `[MASK]`、10% 随机词、10% 保持原词——目的是缓解「预训练有 `[MASK]`、下游没有」的不一致；② NSP 后被证明贡献很小，RoBERTa 已弃用。
- **Java 视角**：像 Spring Boot 自动配置——不从零搭，只在预训练好的骨架上做少量定制（微调）即得优秀效果。
- **前置**：Transformer Encoder、自注意力（模块 01）。

### 02 · GPT 系列
- **核心概念**：Decoder-only 自回归模型，一切能力都建立在「预测下一个 token」这一极简目标上，规模够大即涌现语法、知识、推理、编程。
- **关键公式**：`P(text)=∏ P(tᵢ|t₁…tᵢ₋₁)`，训练即最大化 `Σ log P(tᵢ|t<ᵢ)`；采样温度 `softmax(logits/T)`，T→0 趋近贪心。
- **易错点**：① Zero-shot(GPT-2)/Few-shot In-Context Learning(GPT-3) **不更新任何权重**，只靠 prompt 里的示例；② Chain-of-Thought 是涌现能力，仅在 ~100B+ 模型上稳定生效，GPT-2 加了也没用。
- **Java 视角**：自回归生成像流式处理——每产出一个 token 就把它追加进上下文再喂回去，类似逐条消费并回写的 pipeline。
- **前置**：Transformer Decoder、因果掩码（模块 01）。

### 03 · T5 及开源生态
- **核心概念**：T5 把所有 NLP 任务统一成「文本→文本」，用 prompt 前缀路由任务；当下开源主力(LLaMA/Qwen/Mistral/DeepSeek)则走 Decoder-only 路线。
- **关键 API/公式**：`T5ForConditionalGeneration`（Encoder-Decoder）；MoE 记账关键——**总参数=全部专家之和**，**单 token 计算量只算被路由选中的 Top-2 专家 + 共享部分**（如 Mixtral-8x7B ≈47B 总参 / ≈13B active）。
- **易错点**：① MoE 的「47B→13B」不是简单 `2/8`，注意力/嵌入等非专家部分对所有 token 共享；② 8 个专家都要常驻显存，省的是算力不是显存。
- **Java 视角**：Text-to-Text 像把所有业务收敛到同一个 `Function<String,String>` 接口；MoE 路由像按请求特征命中不同微服务实例，只调用命中的那几个。
- **前置**：01 BERT、02 GPT（对比三种架构）。

### 04 · 缩放定律
- **核心概念**：模型 loss 与参数量 N、数据量 D、算力 C 呈**可预测的幂律**，能在训练前估算最终性能与最优资源分配。
- **关键公式**：Chinchilla 计算最优 `D_opt ≈ 20·N`（token/参数比约 20）、`N_opt∝C^0.5`、`D_opt∝C^0.5`；训练算力 `C ≈ 6·N·D`。
- **易错点**：① Kaplan(2020) 高估模型、低估数据，Chinchilla(2022) 修正为「模型和数据等比例增长」，GPT-3 其实是「训练不足」；② 工业界故意**过度训练小模型**（LLaMA-3-8B 用 15T tokens 远超最优）——因为推理成本 ≫ 训练成本。
- **Java 视角**：像性能基准测试——用小规模压测拟合曲线，预测大规模系统表现，指导容量规划。
- **前置**：02 GPT（涌现能力）、模型参数量估算（模块 01）。

---

## 🎯 学习要点

- **两条路线一句话说清**：BERT(Encoder/双向)擅理解、GPT(Decoder/自回归)擅生成；大模型时代 Decoder-only + 指令微调全面胜出，理解型任务也被追平。
- **动手跑三类模型**：用 `transformers` 分别做 BERT 情感分类（IMDB，准确率≈93%）、GPT-2 few-shot 生成、FLAN-T5 一模型多任务，亲身体会范式差异。
- **背下 Chinchilla 法则**：`D≈20N`、`C≈6ND` 两个公式能反解「给定预算该训多大模型/用多少数据」，是 04 全节的抓手。
- **算一遍参数量**：手推 BERT-base≈110M、GPT-3≈175B、LLaMA-7B(SwiGLU 三矩阵)≈6.7B，把架构超参和参数规模建立直觉。
- **看懂 MoE 记账**：能解释「总参数 vs 激活参数」的区别，是理解 Mixtral/DeepSeek-V2 等现代高效模型的前提。
- **建立选型判断**：中文→Qwen、代码→DeepSeek-Coder、通用对话→LLaMA-3、传统分类→BERT/RoBERTa、翻译摘要→FLAN-T5，能按场景+成本给出理由。

---

## 🔗 关联

- **上一模块**：[01-transformer](../01-transformer/) — 本模块的 BERT/GPT/T5 都是 Transformer 三种用法的落地。
- **下一模块**：[03-training-techniques](../03-training-techniques/) — 这些模型如何被预训练、分布式训练与对齐出来。
- **本阶段总览**：[阶段六 README](../README.md)
- **实战延伸**：[agent-course · Day 01 首次调用](../../agent-course/Day-01-first-call.md) 与 [Day 52 模型路由](../../agent-course/Day-52-model-routing.md)（开源 vs 闭源选型、成本权衡）、[Day 51 成本优化](../../agent-course/Day-51-cost-optimization.md)（Scaling Laws 在推理成本上的现实映射）。
