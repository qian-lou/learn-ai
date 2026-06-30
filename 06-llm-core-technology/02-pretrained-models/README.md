# 02-pretrained-models — 预训练模型

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：掌握主流预训练模型的架构差异和缩放规律

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [bert](./01-bert.md) | BERT 详解 | MLM 预训练、NSP 任务、Fine-tuning 范式 |
| 02 | [gpt-series](./02-gpt-series.md) | GPT 系列 | GPT-1/2/3/4 演进、自回归生成、In-context Learning |
| 03 | [t5-and-others](./03-t5-and-others.md) | T5 及其他模型 | Text-to-Text 统一范式、LLaMA、Mistral |
| 04 | [scaling-laws](./04-scaling-laws.md) | 缩放定律 | Chinchilla Scaling Laws、参数与数据的关系 |

---

## 🎯 学习要点

- BERT（编码器）和 GPT（解码器）代表两种主流预训练范式
- 缩放定律揭示了模型规模、数据量和计算量的最优比例
- 理解不同模型的适用场景：理解型 vs 生成型
