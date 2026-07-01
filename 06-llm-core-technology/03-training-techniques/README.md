# 03-training-techniques — 训练技术

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：掌握大模型「预训练 → 分布式工程 → 人类对齐」的完整训练链路
> **预估时长**：4-5 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [pretraining-strategies](./01-pretraining-strategies.md) | 预训练策略 | MLM(BERT)/CLM(GPT)/Span Corruption(T5)/Prefix-LM 的目标差异、数据配比、继续预训练 |
| 02 | [distributed-training](./02-distributed-training.md) | 分布式训练 | DDP/TP/PP/FSDP、ZeRO 三级分片、3D 并行、显存与通信瓶颈、梯度累积 |
| 03 | [rlhf](./03-rlhf.md) | 对齐技术 | SFT→RM→PPO 三阶段、DPO 及其损失、GRPO+RLVR 推理模型新范式、TRL 实战 |

---

## 🔑 知识点详解

### 01 · 预训练策略
- **核心概念**：所有预训练本质都是「给上下文、预测缺失/后续内容」，区别只在「给多少上下文、预测什么」——MLM 挖洞填词、CLM 顺序续写、Span 补片段。
- **关键公式**：CLM 目标 `∏ P(xᵢ|x<ᵢ)`；MLM 遮盖 15%，其中 80% `[MASK]` / 10% 随机词 / 10% 原词。
- **易错点**：① CLM 成为主流的根本原因——「完美预测下一个词」本身就要求深度理解，故规模够大后 CLM 也能做理解任务，且天然支持生成；② 数据质量 > 数量，配比(Web/代码/百科/论文)直接影响能力特长。
- **Java 视角**：预训练像框架初始化——Spring Boot 启动时自动扫描注入依赖，大模型预训练时从海量文本「注入」语言与世界知识。
- **前置**：模块 02（BERT 的 MLM、GPT 的 CLM）。

### 02 · 分布式训练
- **核心概念**：单卡放不下大模型（GPT-3 约 350GB / A100 仅 80GB），必须把数据/参数/梯度/优化器状态切分到多卡多节点。
- **关键公式/API**：训练显存 ≈ 参数(2N) + 梯度(2N) + Adam 优化器(FP32 的 m+v，8N) + 激活值；ZeRO-3/FSDP 把前三者均分到 N 卡，显存约降 75%。启动用 `torchrun --nproc_per_node=K`。
- **易错点**：① DDP 要求**每张卡放得下完整模型**，放不下才需要 FSDP/ZeRO 或模型并行；② 跨节点通信（InfiniBand/以太网）远慢于节点内 NVLink，通信常是瓶颈，用梯度累积可降低同步频率。
- **Java 视角**：就是分布式系统——数据分片(Sharding)、All-Reduce 负载均衡、节点通信、一致性保证，概念完全相通。
- **前置**：模型参数量与显存估算（模块 01/02）、PyTorch 训练循环。

### 03 · 对齐技术（RLHF / DPO / GRPO）
- **核心概念**：预训练只会「预测下一个 token」，不知何为好回答；对齐让模型贴合人类偏好——SFT 教格式、RM 学偏好、PPO/DPO 优化策略。
- **关键公式**：PPO 目标 `max E[RM(x,y)] − β·KL(π‖π_ref)`；DPO 直接在偏好对上优化，`L = −log σ(β·(log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)))`，无需训练 RM。
- **易错点**：① DPO/PPO 都要一个**冻结的参考模型 π_ref** 与 KL 项，防止模型跑偏「胡言乱语」；② 2024-2025 推理模型(o1/R1)用 **GRPO+RLVR**——GRPO 用组内标准化优势 `A_i=(r_i−mean)/std` 免去 critic，RLVR 用规则可验证奖励(数学答案对错/单测通过)替代易被 hack 的 RM。
- **Java 视角**：三阶段像分层交付——SFT 定接口契约、RM 是自动化评分器、PPO/DPO 是按评分持续优化的 CI 回路；KL 项则是防止「优化到面目全非」的回归护栏。
- **前置**：02 GPT（生成与采样）、强化学习基础（策略/奖励/优势）。

---

## 🎯 学习要点

- **记住完整流水线**：预训练(知识) → SFT(对话格式) → RLHF/DPO(人类偏好)，三步缺一不可，能说清每步的输入输出。
- **算一遍显存账**：手推「7B 模型训练需 ~104GB」的构成（参数+梯度+Adam+激活），并算出用 FSDP 4 卡后每卡降到 ~21GB——这是理解分布式必要性的关键。
- **跑通 DDP 与 DPO**：单机多卡 `torchrun` 跑一次 DDP；用 TRL `DPOTrainer` 对 GPT-2 做一次 DPO，用**极小学习率(5e-7)**，走通「加载 SFT/ref → 喂偏好对 → 训练」全链路。
- **分清 RLHF vs DPO vs GRPO**：RLHF(重、上限高、需 RM+critic)、DPO(离线、稳、无 RM)、GRPO+RLVR(在线、无 critic、可验证奖励、推理模型首选)——能按任务选型。
- **理解「对齐 > 规模」**：1.3B InstructGPT 在人类评估上胜过 175B 原始 GPT-3，说明对齐的价值可超过单纯堆参数。
- **掌握 ZeRO 三级**：ZeRO-1(切优化器)/ZeRO-2(+梯度)/ZeRO-3(+参数≈FSDP)，并知道 `offload` 用带宽换显存的取舍。

---

## 🔗 关联

- **上一模块**：[02-pretrained-models](../02-pretrained-models/) — 本模块解释这些模型是怎样被训练与对齐出来的。
- **下一模块**：[阶段七 · 大模型应用](../../07-llm-applications/) — 从「训练模型」转向「用模型构建应用」。
- **本阶段总览**：[阶段六 README](../README.md)
- **实战延伸**：[agent-course · Day 03 提示基础](../../agent-course/Day-03-prompt-basics.md)（对齐后的指令跟随是提示工程的前提）、[Day 49 评估入门](../../agent-course/Day-49-eval-intro.md) 与 [Day 50 编写评估](../../agent-course/Day-50-writing-evals.md)（RM/RLVR 的可验证奖励思想与离线评估同源）。
