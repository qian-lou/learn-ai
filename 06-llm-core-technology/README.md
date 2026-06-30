# 阶段六：大模型核心技术

> **预估周期**：3-4 周
> **核心目标**：Transformer + BERT/GPT + 训练技术

---

## 📋 模块大纲

### [01-transformer](./01-transformer/) — Transformer 架构

现代 AI 的基石架构，彻底理解自注意力和位置编码。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [self-attention](./01-transformer/01-self-attention.md) | 自注意力机制详解 |
| 02 | [multi-head-attention](./01-transformer/02-multi-head-attention.md) | 多头注意力 |
| 03 | [positional-encoding](./01-transformer/03-positional-encoding.md) | 位置编码 |
| 04 | [transformer-architecture](./01-transformer/04-transformer-architecture.md) | Transformer 完整架构 |
| 05 | [transformer-from-scratch](./01-transformer/05-transformer-from-scratch.md) | 从零实现 Transformer |

---

### [02-pretrained-models](./02-pretrained-models/) — 预训练模型

BERT、GPT 系列及其背后的缩放定律。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [bert](./02-pretrained-models/01-bert.md) | BERT 详解 |
| 02 | [gpt-series](./02-pretrained-models/02-gpt-series.md) | GPT 系列（GPT-1/2/3/4） |
| 03 | [t5-and-others](./02-pretrained-models/03-t5-and-others.md) | T5 及其他模型 |
| 04 | [scaling-laws](./02-pretrained-models/04-scaling-laws.md) | 缩放定律（Scaling Laws） |

---

### [03-training-techniques](./03-training-techniques/) — 训练技术

大模型训练的关键技术：预训练策略、分布式训练和人类对齐。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [pretraining-strategies](./03-training-techniques/01-pretraining-strategies.md) | 预训练策略（MLM/CLM/Span） |
| 02 | [distributed-training](./03-training-techniques/02-distributed-training.md) | 分布式训练（DDP/FSDP/DeepSpeed） |
| 03 | [rlhf](./03-training-techniques/03-rlhf.md) | RLHF 人类反馈强化学习 |
