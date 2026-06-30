# 01-model-optimization — 模型优化

> **所属阶段**：阶段八 · 大模型部署与工程化
> **学习目标**：掌握模型压缩和推理加速技术

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [quantization](./01-quantization.md) | 模型量化 | INT8/INT4 量化、GPTQ/AWQ 算法、量化精度对比 |
| 02 | [inference-acceleration](./02-inference-acceleration.md) | 推理加速 | KV Cache、FlashAttention、TensorRT |
| 03 | [knowledge-distillation](./03-knowledge-distillation.md) | 知识蒸馏 | Teacher-Student 架构、蒸馏损失函数 |

---

## 🎯 学习要点

- 量化是降低部署成本的最直接方法
- KV Cache 和 FlashAttention 是 LLM 推理优化的核心技术
- 知识蒸馏可以用小模型逼近大模型的效果
