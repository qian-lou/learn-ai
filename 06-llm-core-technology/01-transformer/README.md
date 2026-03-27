# 01-transformer — Transformer 架构

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：彻底理解 Transformer 架构的每个组件

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [self-attention](./01-self-attention.md) | 自注意力机制 | Q/K/V 计算、缩放点积注意力、复杂度分析 |
| 02 | [multi-head-attention](./02-multi-head-attention.md) | 多头注意力 | 多头并行、头数选择、注意力可视化 |
| 03 | [positional-encoding](./03-positional-encoding.md) | 位置编码 | 正弦位置编码、RoPE、ALiBi |
| 04 | [transformer-architecture](./04-transformer-architecture.md) | Transformer 完整架构 | Encoder-Decoder 结构、LayerNorm、残差连接 |
| 05 | [transformer-from-scratch](./05-transformer-from-scratch.md) | 从零实现 Transformer | 纯 PyTorch 实现、逐模块构建 |

---

## 🎯 学习要点

- **Attention Is All You Need** 论文是必读内容
- 自注意力的 $O(N^2)$ 复杂度是大模型长文本处理的瓶颈
- 从零实现是真正理解 Transformer 的最佳方式
