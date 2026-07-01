# 阶段六：大模型核心技术

> **预估周期**：3-4 周
> **核心目标**：Transformer + BERT/GPT + 训练技术
> **学习目标**：吃透大模型的底层架构、主流模型谱系与「预训练→分布式→对齐」全链路，能从零写出 GPT 并做技术选型

---

## 📋 模块大纲

### [01-transformer](./01-transformer/) — Transformer 架构

现代 AI 的基石架构。彻底理解缩放点积自注意力、多头并行、位置编码(正弦/RoPE/ALiBi)、残差与 Pre-Norm，并从零写出可训练、可生成的 GPT。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [self-attention](./01-transformer/01-self-attention.md) | 自注意力机制详解 |
| 02 | [multi-head-attention](./01-transformer/02-multi-head-attention.md) | 多头注意力 |
| 03 | [positional-encoding](./01-transformer/03-positional-encoding.md) | 位置编码 |
| 04 | [transformer-architecture](./01-transformer/04-transformer-architecture.md) | Transformer 完整架构 |
| 05 | [transformer-from-scratch](./01-transformer/05-transformer-from-scratch.md) | 从零实现 Transformer |

### [02-pretrained-models](./02-pretrained-models/) — 预训练模型

主流模型谱系与缩放规律。理解 BERT(理解型)与 GPT(生成型)两条路线的取舍、T5 统一范式与开源生态(LLaMA/Qwen/Mistral/DeepSeek/MoE)，并用 Chinchilla 缩放定律指导资源分配。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [bert](./02-pretrained-models/01-bert.md) | BERT 详解 |
| 02 | [gpt-series](./02-pretrained-models/02-gpt-series.md) | GPT 系列（GPT-1/2/3/4） |
| 03 | [t5-and-others](./02-pretrained-models/03-t5-and-others.md) | T5 及其他模型 |
| 04 | [scaling-laws](./02-pretrained-models/04-scaling-laws.md) | 缩放定律（Scaling Laws） |

### [03-training-techniques](./03-training-techniques/) — 训练技术

大模型训练的关键技术：预训练目标(MLM/CLM/Span)决定能力边界、分布式训练(DDP/FSDP/ZeRO)突破单卡显存、人类对齐(RLHF/DPO/GRPO+RLVR)让模型贴合偏好。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [pretraining-strategies](./03-training-techniques/01-pretraining-strategies.md) | 预训练策略（MLM/CLM/Span） |
| 02 | [distributed-training](./03-training-techniques/02-distributed-training.md) | 分布式训练（DDP/FSDP/DeepSpeed） |
| 03 | [rlhf](./03-training-techniques/03-rlhf.md) | RLHF / DPO / GRPO 对齐技术 |

---

## 🔑 阶段核心脉络

三个模块层层递进，回答三个问题：**大模型长什么样 → 有哪些模型 → 怎么训出来。**

| 主线 | 一句话本质 | 最该记住的 |
|------|-----------|-----------|
| **架构** | 一切现代大模型 = N 层「注意力聚合信息 + FFN 处理信息」，包在残差与 Norm 里 | `Attention=softmax(QKᵀ/√d_k)V`；O(N²) 是长文本瓶颈 |
| **模型** | Encoder(BERT/理解) vs Decoder(GPT/生成)，大模型时代 Decoder-only 胜出 | Chinchilla `D≈20N`、训练算力 `C≈6ND` |
| **训练** | 预训练学知识 → SFT 学格式 → RLHF/DPO 学偏好；分布式让规模成为可能 | PPO 目标 `max E[RM]−β·KL`；ZeRO-3/FSDP 省显存 ~75% |

**贯穿全阶段的暗线是「O(N²) 与显存」**：自注意力的平方复杂度 → KV Cache/GQA/FlashAttention 优化 → 分布式训练的显存分片，理解这条线就抓住了大模型工程的核心矛盾。

---

## 🎯 学习要点

- **手写一个 GPT**：不用高级封装，从零实现自注意力→多头→Transformer Block→完整 GPT，并在 tinyshakespeare 上训到能生成文本——这是本阶段最硬的验收标准，也是 AI 面试高频考点。
- **形状驱动理解**：全程用 `[B, N, D]` 张量形状追踪数据流，多头的 reshape、注意力矩阵 `[B,H,N,N]`、KV Cache 的拼接都靠形状说清。
- **背两个公式、算三笔账**：记住 `Attention=softmax(QKᵀ/√d_k)V` 与 Chinchilla `D≈20N`；算清 BERT-base≈110M、LLaMA-7B≈6.7B 参数量、7B 训练≈104GB 显存。
- **建立模型选型判断**：能按「理解/生成」「中文/代码/通用」「预算/单卡」等维度，在 BERT/GPT/T5/LLaMA/Qwen/DeepSeek 间给出有理由的推荐。
- **对齐时间线要跟上 2026 视角**：RLHF(PPO) → DPO → GRPO+RLVR，能说清推理模型(o1/R1)为何用可验证奖励替代 RM、用组内基线替代 critic。
- **实践闭环**：用 `transformers` 跑 BERT 微调/GPT-2 生成/FLAN-T5 多任务，用 `torchrun` 跑 DDP，用 TRL 跑一次 DPO——把每个理论点落到能运行的代码上。

---

## 🔗 关联

- **上一阶段**：[阶段五 · NLP 基础](../05-nlp-fundamentals/) — 词向量、RNN/LSTM、seq2seq 与注意力雏形，是理解 Transformer 的前置。
- **下一阶段**：[阶段七 · 大模型应用](../07-llm-applications/) — 从「训练模型」转向「用模型构建 RAG、Agent 等应用」。
- **实战课程**：[agent-course（70 天 AI Agent 开发课）](../agent-course/) — 本阶段的原理是其工程实践的地基，尤其 [Day 01 首次调用](../agent-course/Day-01-first-call.md)、[Day 02 模型参数](../agent-course/Day-02-model-params.md)、[Day 40 性能](../agent-course/Day-40-performance.md)、[Day 52 模型路由](../agent-course/Day-52-model-routing.md)。
