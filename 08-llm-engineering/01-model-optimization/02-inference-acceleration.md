# 推理加速（KV Cache/FlashAttention）/ Inference Acceleration

## 1. 背景（Background）
> 大模型推理延迟高，需要 KV Cache、FlashAttention、Speculative Decoding 等技术加速。

## 2-3. 知识点与内容
```
关键加速技术：

KV Cache:
- 自回归生成时，缓存已计算的 K/V，避免重复计算
- 显存占用 = 2 × num_layers × hidden_size × seq_len × batch_size

FlashAttention:
- IO 感知的注意力计算，减少 HBM 访问
- 不改变计算结果，只优化计算过程
- 速度 2-4x，显存节省 5-20x

Speculative Decoding:
- 用小模型快速生成候选 token，大模型验证
- 在保持输出质量的同时提速 2-3x
```

## 4-6. 推理/例题/习题
**练习：** 对比开启/关闭 KV Cache 的推理速度差异。
