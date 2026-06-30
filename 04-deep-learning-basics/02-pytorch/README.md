# 02-pytorch — PyTorch 框架

> **所属阶段**：阶段四 · 深度学习基础
> **学习目标**：掌握 PyTorch 模型构建与训练全流程

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [tensor-basics](./01-tensor-basics.md) | Tensor 基础与自动求导 | Tensor 创建、运算、autograd 机制 |
| 02 | [nn-module](./02-nn-module.md) | nn.Module 模型构建 | 自定义网络、参数管理、模块组合 |
| 03 | [training-loop](./03-training-loop.md) | 训练循环与 DataLoader | Dataset/DataLoader、训练/验证循环 |
| 04 | [gpu-acceleration](./04-gpu-acceleration.md) | GPU 加速与混合精度 | CUDA 使用、AMP 混合精度训练 |
| 05 | [model-save-and-load](./05-model-save-and-load.md) | 模型保存与加载 | state_dict、Checkpoint、ONNX 导出 |

---

## 🎯 学习要点

- PyTorch 的动态计算图是其核心优势，便于调试
- 掌握标准训练循环模式，后续所有项目都会复用
- GPU 加速和混合精度是大模型训练的必备技能
