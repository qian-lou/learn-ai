# RLHF 人类反馈强化学习

## 1. 背景（Background）

> **为什么要学这个？**
>
> RLHF（Reinforcement Learning from Human Feedback）是 **ChatGPT 成功的关键技术**。预训练只让模型学会了"预测下一个 token"，但不知道什么是好的回答、什么是有害的内容。RLHF 让模型与人类价值观**对齐（Alignment）**。
>
> 没有 RLHF，GPT-3 的回答经常乱讲、有害、跑题。加了 RLHF 后（InstructGPT），1.3B 的小模型就比 175B 的原始 GPT-3 更受用户欢迎！
>
> **在整个体系中的位置：** 预训练（基础知识）→ SFT（学会对话格式）→ RLHF/DPO（对齐人类偏好），三步构成大模型的完整训练流程。

## 2. 知识点（Key Concepts）

| 阶段 | 技术 | 目标 | 数据 |
|------|------|------|------|
| 阶段 1 | SFT（监督微调） | 学会对话格式 | 人工标注的对话 |
| 阶段 2 | RM（奖励模型） | 学习人类偏好 | 偏好对比数据 |
| 阶段 3 | PPO（策略优化） | 用 RL 优化模型 | RM 打分 |
| 替代方案 | DPO | 直接偏好优化 | 偏好对比数据 |

## 3. 内容（Content）

### 3.1 RLHF 三阶段详解

```
阶段 1: SFT (Supervised Fine-Tuning)

  用高质量对话数据微调预训练模型:
  
  输入: "写一首关于春天的诗"
  输出: "春风拂面花开早，细雨润物草色新..."
  
  数据来源: 人工标注员编写高质量回答
  数据量: ~10K-100K 条
  → 模型学会了对话格式和基本的指令跟随


阶段 2: RM (Reward Model) 训练

  人工标注偏好数据:
  
  问题: "写一首关于春天的诗"
  回答 A: "春风拂面花开早..." (优雅，有意境)  ← 更好
  回答 B: "春天来了花开了..."  (简单，无意境)
  
  训练奖励模型: RM(prompt, response) → score
  → 学会给好的回答高分，差的回答低分


阶段 3: PPO (Proximal Policy Optimization)

  用奖励模型的打分做强化学习:
  
  目标: max E[RM(prompt, response)] - β·KL(π || π_ref)
  
  π:     当前策略（正在优化的模型）
  π_ref: 参考策略（SFT 后的模型，防止过度优化）
  KL:    KL 散度惩罚（防止偏离太远变"鬼话连篇"）
  
  循环:
  1. 用当前模型生成回答
  2. 用奖励模型打分
  3. 用 PPO 更新模型参数
  4. 重复
```

### 3.2 DPO（Direct Preference Optimization）

```
DPO 的核心思想:
  RLHF 需要训练 RM + PPO → 很复杂、不稳定
  DPO 跳过 RM，直接从偏好数据优化模型

DPO 损失函数:
  L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

  y_w: 被偏好的回答 (winner)
  y_l: 被拒绝的回答 (loser)
  π:   当前模型
  π_ref: 参考模型（SFT）

直觉:
  让模型对好回答的概率增大
  让模型对差回答的概率减小
  同时不要偏离参考模型太远

优势:
  ✅ 无需训练奖励模型
  ✅ 训练更稳定
  ✅ 实现更简单
  ❌ 可能没有 RLHF+PPO 的上限高
```

### 3.3 DPO 代码示例

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,     # π(y_w|x) 的 log 概率
    policy_rejected_logps: torch.Tensor,   # π(y_l|x) 的 log 概率
    reference_chosen_logps: torch.Tensor,  # π_ref(y_w|x) 的 log 概率
    reference_rejected_logps: torch.Tensor, # π_ref(y_l|x) 的 log 概率
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO 损失函数 / DPO loss function.
    
    Args:
        beta: KL 散度惩罚系数，越大越保守
    """
    # 计算 log ratio
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    
    # DPO 损失
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # 隐式奖励
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    
    return loss, chosen_rewards, rejected_rewards
```

### 3.4 偏好数据构建

```
偏好数据的构建方式:

1. 人工标注 (最贵但最好):
   标注员比较两个回答，选择更好的
   成本: ~$1-5 / 标注对
   
2. AI 标注 (RLAIF):
   用强模型 (GPT-4) 做评判
   成本: ~$0.01 / 标注对
   质量: 接近人工标注

3. 自动构建:
   正样本: 模型的高温采样中选最好的
   负样本: 模型的高温采样中选最差的

4. 开源数据集:
   HH-RLHF (Anthropic): 人类偏好数据
   UltraFeedback: GPT-4 标注的偏好数据
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么需要对齐？

```
未对齐的 GPT-3 的典型问题:

1. 有害内容: 会生成歧视、暴力、违法内容
2. 幻觉: 自信地编造错误事实
3. 不跟随指令: 经常跑题或过度回答
4. 无用回答: 重复问题或给出空洞回答

对齐后 (InstructGPT/ChatGPT):
  1.3B InstructGPT > 175B GPT-3（人类评估）
  → 对齐的价值甚至超过模型规模！
```

### 4.2 RLHF vs DPO vs 其他

```
对齐技术演进:

RLHF (PPO):  复杂但效果上限高
  OpenAI (ChatGPT), Anthropic (Claude)

DPO:  简单稳定，效果接近 RLHF
  Meta (LLaMA-2), 大部分开源模型

ORPO:  无需参考模型
  更新更简单

KTO:   只需要好/坏标签（不需要成对比较）
  数据要求更低

SimPO:  序列级别的简单偏好优化

趋势: 从复杂 (RLHF) → 简单 (DPO) → 更简单 (KTO/SimPO)
```

## 5. 例题（Worked Examples）

### 例题：用 TRL 库做 DPO 训练

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# DPO 训练的基本流程
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")

training_args = DPOConfig(
    output_dir="dpo-model",
    beta=0.1,                    # KL 惩罚系数
    per_device_train_batch_size=4,
    learning_rate=5e-7,          # 很小的学习率
    num_train_epochs=1,
)

# 偏好数据格式: {"prompt": ..., "chosen": ..., "rejected": ...}
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 解释 SFT → RM → PPO 三步训练流程中每一步的输入输出。

**练习 2：** 对比 RLHF 和 DPO 的优缺点。

### 进阶题

**练习 3：** 用 TRL 库对一个小模型（如 GPT-2）做 DPO 训练，体验完整工作流。

**练习 4：** 构建偏好数据集：让 GPT-2 生成多个回答，人工标注偏好，用于训练 DPO。
