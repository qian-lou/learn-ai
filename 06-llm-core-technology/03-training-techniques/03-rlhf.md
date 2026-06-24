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
| 推理模型主力 | GRPO + RLVR | 组内相对优化 + 可验证奖励 | 规则可验证信号（数学/代码） |

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO 损失函数 / DPO loss function. 返回 (loss, chosen_rewards, rejected_rewards).
    
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

### 4.3 GRPO 与可验证奖励 RL（推理模型 2024-2025）

2024-2025 的 **推理模型（reasoning model）** 浪潮（OpenAI o1、DeepSeek-R1）让 RL 重新成为对齐主线，但用的不再是 PPO，而是 **GRPO + RLVR** 这对组合。

```
GRPO (Group Relative Policy Optimization) —— 群体相对策略优化:

  PPO 的痛点: 要额外训练一个 value/critic 网络估计基线 V(s)，
              与 policy 同尺寸 → 显存×2、还难训。

  GRPO 的做法 (DeepSeek 提出，R1/数学推理主力算法):
    1. 对同一个 prompt 采样一组 G 个回答 {o_1, ..., o_G}
    2. 每个回答用奖励函数打分得 {r_1, ..., r_G}
    3. 直接用"组内"统计做基线，免去 critic:
         A_i = (r_i - mean(r)) / std(r)      ← 组内标准化即优势
    4. 用 A_i 做 PPO 式裁剪更新 + KL(π‖π_ref) 惩罚

  为何更省显存: 砍掉了和 policy 等大的 value 网络，
                只需 policy + 冻结的 ref，显存与工程量都显著下降。

RLVR (Reinforcement Learning with Verifiable Rewards) —— 可验证奖励 RL:

  奖励不来自"学出来的奖励模型 RM"，而来自规则可验证信号:
    • 数学题: 抽取最终答案与标准答案比对 → 对=1 / 错=0
    • 代码题: 跑单元测试，全过=1 / 否则=0
  → 没有 RM 被 hack 的风险，信号客观、可无限生成。
  → 配合 GRPO 催生 o1/R1 式"长思维链(long CoT) + 自我反思"行为:
    模型为了拿到可验证奖励，自发学会拉长推理、回溯、检查。

与 PPO / DPO 的关系:
  PPO : on-policy RL，需 RM + critic，通用但重。
  DPO : 离线、无 RL、无 RM，靠成对偏好；不适合"答案可判对错"的推理任务。
  GRPO: on-policy RL，无 critic；奖励既可用 RM 也可用 RLVR 规则，
        是推理模型(可验证任务)的当前首选。
```

```python
# TRL GRPOTrainer 最小示例（trl>=0.14；reward 用可验证打分函数）
# Minimal GRPO example; reward_func is a verifiable (rule-based) scorer.
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 可验证奖励：答案里出现目标字符串就给 1 分，否则 0 分
# Verifiable reward: 1.0 if the gold token appears, else 0.0 (no learned RM)
def reward_correct(completions: list[str], **kwargs) -> list[float]:
    # 真实场景应做"抽取最终答案→与标准答案/单测比对"
    return [1.0 if "42" in c else 0.0 for c in completions]

ds = load_dataset("trl-lib/tldr", split="train")           # 仅需含 prompt 列
args = GRPOConfig(
    output_dir="qwen-grpo",
    num_generations=8,            # 每个 prompt 采样一组(G=8)算组内基线 / group size
    per_device_train_batch_size=8,
    max_completion_length=512,    # 给长思维链留足生成长度 / room for long CoT
)
GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_correct,  # 可传多个奖励函数，分数相加 / list is allowed
    args=args,
    train_dataset=ds,
).train()
```

## 5. 例题（Worked Examples）

### 例题：用 TRL 库做 DPO 训练

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# DPO 训练的基本流程
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

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
    processing_class=tokenizer,  # TRL 0.12+ 用 processing_class 取代 tokenizer
)
trainer.train()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 解释 SFT → RM → PPO 三步训练流程中每一步的输入输出。

*参考答案*：

```
① SFT (监督微调)
   输入: 预训练模型 + 高质量(prompt, 理想回答)对
   输出: 会按对话格式跟随指令的 SFT 模型 π_ref
   损失: 对"理想回答"部分做交叉熵(next-token)

② RM (奖励模型)
   输入: SFT 模型(常用其骨干) + 偏好对(prompt, 回答A, 回答B, 谁更好)
   输出: 打分函数 RM(prompt, response) → 标量
   损失: 排序损失 -log σ(r_好 - r_差)，让更优回答得分更高

③ PPO (策略优化)
   输入: SFT 模型(初始策略 π) + RM + 一批 prompt
   输出: 对齐后的策略模型 π
   目标: max E[RM(x, y)] - β·KL(π || π_ref)
   流程: π 生成回答 → RM 打分 → PPO 更新 π，KL 项防止偏离 π_ref 太远
```
一句话串联：SFT 教"会说话"，RM 学"什么是好回答"，PPO 用 RM 当奖励把策略推向人类偏好。

**练习 2：** 对比 RLHF 和 DPO 的优缺点。

*参考答案*：

```
RLHF (RM + PPO):
  ✅ 效果上限高，可在线探索新回答；奖励信号可复用、可叠加多目标
  ❌ 流程复杂(需训 RM + 跑 PPO)，训练不稳定、超参敏感
  ❌ 显存/算力大(同时持有 policy/ref/RM/critic 多个模型)
  代表: OpenAI(ChatGPT)、Anthropic(Claude)

DPO (直接偏好优化):
  ✅ 跳过 RM 与 RL，直接在偏好对上做分类式优化；训练稳定、实现简单、开销小
  ✅ 理论上等价于带 KL 约束的 RLHF 最优解(隐式奖励)
  ❌ 离线方法，只能用已有偏好数据，无在线探索；上限可能略低于精调的 RLHF
  ❌ 对偏好数据质量/分布更敏感
  代表: LLaMA-2 及大量开源模型
```
选型：追求极致效果且有工程能力 → RLHF；要快速、稳定、低成本对齐 → DPO（及 KTO/SimPO 等更简方案）。



### 进阶题

**练习 3：** 用 TRL 库对一个小模型（如 GPT-2）做 DPO 训练，体验完整工作流。

*参考答案*：复用 5. 例题的 `DPOTrainer` 框架，把模型换成 GPT-2、数据集换成现成的偏好数据即可跑通。

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

base = "gpt2"                                   # 也可先对 GPT-2 做 SFT 再 DPO
model = AutoModelForCausalLM.from_pretrained(base)
ref   = AutoModelForCausalLM.from_pretrained(base)   # 冻结的参考模型 π_ref
tok = AutoTokenizer.from_pretrained(base); tok.pad_token = tok.eos_token
# 偏好数据需含 prompt / chosen / rejected 三列
ds = load_dataset("trl-internal-testing/hh-rlhf-helpful-base-trl-style", split="train")

args = DPOConfig(output_dir="gpt2-dpo", beta=0.1, learning_rate=5e-7,
                 per_device_train_batch_size=2, num_train_epochs=1)
DPOTrainer(model=model, ref_model=ref, args=args,
           train_dataset=ds, processing_class=tok).train()
```
要点：DPO 用**极小学习率**(5e-7 量级)，`beta` 控 KL 约束强度；GPT-2 太小，效果有限，重在跑通"加载 SFT/ref 模型 → 喂偏好对 → 训练"的完整链路。可对比训练前后对同一 prompt 生成回答的偏好倾向变化。

**练习 4：** 构建偏好数据集：让 GPT-2 生成多个回答，人工标注偏好，用于训练 DPO。

*参考答案*：流程为"**一个 prompt 多次采样 → 人工/AI 标注两两偏好 → 落成 (prompt, chosen, rejected) 格式**"。

```python
import torch
def sample_candidates(prompt, k=4, T=1.0):
    """对同一 prompt 高温采样 k 个不同回答 / sample k diverse responses."""
    inp = tok(prompt, return_tensors="pt")
    outs = model.generate(**inp, do_sample=True, temperature=T,
                          num_return_sequences=k, max_new_tokens=64)
    return [tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True) for o in outs]

# 标注: 人工(或用 GPT-4 做 RLAIF)从候选里选最好/最差，构成一条偏好对
record = {"prompt": prompt, "chosen": best, "rejected": worst}
```
要点：(1) 采样要有**多样性**(适当高温/`top_p`)，否则候选雷同无法比较；(2) `chosen`/`rejected` 必须基于**同一 prompt**；(3) 标注可用人工（贵、准）或强模型 AI 标注 RLAIF（便宜、接近人工）；(4) 也可直接复用 HH-RLHF、UltraFeedback 等开源偏好集。把若干条 `record` 存成 `datasets.Dataset` 即可直接喂给练习 3 的 `DPOTrainer`。
