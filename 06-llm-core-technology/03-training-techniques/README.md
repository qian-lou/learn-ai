# 03-training-techniques — 训练技术

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：掌握大模型训练的关键技术和人类对齐方法

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [pretraining-strategies](./01-pretraining-strategies.md) | 预训练策略 | MLM、CLM、Span Corruption、混合目标 |
| 02 | [distributed-training](./02-distributed-training.md) | 分布式训练 | DDP、FSDP、DeepSpeed ZeRO、流水线并行 |
| 03 | [rlhf](./03-rlhf.md) | RLHF | Reward Model、PPO 算法、DPO 简化方案 |

---

## 🎯 学习要点

- 预训练目标的选择直接决定模型的能力边界
- 分布式训练是训练数十亿参数模型的工程必备
- RLHF 是 ChatGPT 成功的关键技术之一
