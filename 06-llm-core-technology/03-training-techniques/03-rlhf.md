# RLHF 人类反馈强化学习

## 1. 背景（Background）
> RLHF 是让大模型"对齐"人类偏好的关键技术。ChatGPT 的成功很大程度归功于 RLHF。

## 2-3. 知识点与内容
```
RLHF 三阶段：
1. SFT (Supervised Fine-Tuning): 用高质量对话数据微调
2. RM (Reward Model): 训练奖励模型，学习人类偏好
3. PPO (Proximal Policy Optimization): 用 RL 优化策略

替代方案：DPO (Direct Preference Optimization)
  - 无需训练奖励模型，直接从偏好数据优化
  - 更简单，训练更稳定，效果接近 RLHF
```

## 4-6. 推理/例题/习题
**练习：** 理解 DPO 的损失函数推导，对比 RLHF 和 DPO 的优劣。
