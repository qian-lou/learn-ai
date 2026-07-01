# 04-fine-tuning — 模型微调

> **所属阶段**：阶段七 · 大模型应用实战
> **学习目标**：掌握 LoRA/QLoRA 高效微调，用单张消费级 GPU 定制专属领域模型
> **预估时长**：4-5 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [lora-qlora](./01-lora-qlora.md) | LoRA 与 QLoRA | 低秩分解 `h=Wx+BAx` 原理、`r`/`lora_alpha`/`target_modules` 调参、4-bit NF4 量化、PEFT 保存与合并、DoRA/rsLoRA/unsloth/all-linear 现代配方 |
| 02 | [instruction-tuning](./02-instruction-tuning.md) | 指令微调（SFT） | Alpaca/ShareGPT/OpenAI 三种数据格式、SFTTrainer 完整流程、只对 Response 计算 loss、数据质量 > 数量 |

---

## 🔑 知识点详解

### 01 · LoRA / QLoRA

- **核心概念**：冻结原始权重 W，只训练一对低秩矩阵 B、A（旁路），用 0.1%-1% 的可训练参数逼近全参微调效果；QLoRA 再把基座 4-bit 量化，让 7B/13B 模型能塞进单张 24GB 卡。
- **关键公式/API**：
  - 前向：`h = Wx + BAx`，其中 `B∈R^{d×r}`、`A∈R^{r×k}`，`r≪d`；初始化 `B=0` 使初始 `BA=0`（行为不变）。
  - 参数量：`d×r + r×k` 远小于 `d×k`（如 4096×4096=16.7M → r=16 时仅 131K，降 99.2%）。
  - `LoraConfig(task_type, r, lora_alpha, target_modules)` + `get_peft_model(model, cfg)`；缩放 = `lora_alpha/r`（rsLoRA 用 `alpha/√r`）。
  - QLoRA：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` + `prepare_model_for_kbit_training`。
- **易错点**：
  - `target_modules` 因架构而异——Qwen/Llama 用 `q_proj/k_proj/v_proj/o_proj`，**GPT-2 用合并的 `c_attn`**；写错会报"找不到目标模块"。现代可用 `"all-linear"` 自动选层。
  - `bnb_4bit_compute_dtype` 要传 **torch dtype**（`torch.bfloat16`）而非字符串。
  - 量化模型训练前必须 `prepare_model_for_kbit_training`（启用梯度检查点等），否则不收敛/报错。
- **Java 视角**：LoRA ≈ AOP 面向切面——不改原始类（权重冻结），通过织入切面（低秩旁路）改变行为；多 adapter 切换 ≈ 同一 Bean 挂不同增强按需启用。
- **前置**：模块 01（Trainer/模型加载）；线性代数中的矩阵秩概念。

### 02 · 指令微调（SFT）

- **核心概念**：SFT（Supervised Fine-Tuning）把只会"续写"的基座模型教成会"遵循指令对话"的模型——这是从 LLaMA 到 ChatGPT 的关键一步。
- **关键格式/机制**：
  - 三种数据格式：**Alpaca**（instruction/input/output 三字段）、**ShareGPT**（多轮 conversations）、**OpenAI**（messages 数组 system/user/assistant）。
  - **损失只算 Response**：instruction 部分的 labels 设为 `-100`（忽略）——只教"如何回答"，不教"如何提问"。
  - `SFTTrainer(model, args=SFTConfig(...), train_dataset, processing_class=tokenizer)`。
- **易错点**：
  - TRL 0.12+ 用 `processing_class=` 传分词器，**旧的 `tokenizer=` 参数已弃用**。
  - GPT-2 无 pad token，需 `tok.pad_token = tok.eos_token`。
  - 盲目堆数据量——**1000 条高质量 > 10 万条低质量**，噪音数据反而伤模型。
- **Java 视角**：SFT ≈ 给框架写 Controller——基座是框架（Spring），指令数据是路由配置（`@RequestMapping`），告诉模型不同请求如何响应。
- **前置**：01（LoRA 是 SFT 的高效底座）。

---

## 🎯 学习要点

- **先想清楚该不该微调**：能用 Prompt/RAG 解决就别微调；微调适合"教领域术语和固定回答风格"，事实性知识交给 RAG。
- **默认配方 QLoRA**：消费级卡上 = 4-bit NF4 量化基座 + `prepare_model_for_kbit_training` + `r=16, lora_alpha=32` LoRA + `SFTTrainer`，这是 2026 最省钱的可复现路径。
- **`r` 从 8-16 起步**：`r=8~16` 性价比最高，`r=64` 收益递减且更慢；`lora_alpha` 常取约 `2*r`。
- **只存 adapter、按需切换**：LoRA 权重仅几十 MB，一个基座可挂多个 adapter 用 `set_adapter` 运行时切换，显存高效。
- **数据质量是 SFT 成败关键**：指令清晰、回答准确、格式一致、任务多样、语言自然五条标准，宁缺毋滥。
- **现代加速件了解并按需上**：unsloth（QLoRA 提速 2-5×、显存再降 ~60%）、DoRA（幅度+方向分解、低秩下更接近全参）、`all-linear`（自动选层、对新架构更稳）。

---

## 🔗 关联

- **上一模块**：[03-rag](../03-rag/) — RAG 供最新事实，微调供风格与术语，二者互补；取舍对比见 03 章 RAG-basics。
- **下一模块**：[05-langchain](../05-langchain/) — 把微调好的模型接入 LangChain/Agent 编排成应用。
- **本阶段总览**：[阶段七 README](../README.md)
- **相关 Day**：微调偏"造模型"，agent-course 侧重"用模型"；概念底座可回查本仓库 [06-llm-core-technology 训练技术](../../06-llm-core-technology/03-training-techniques/)。
