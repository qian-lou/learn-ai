# 01-huggingface — HuggingFace 生态

> **所属阶段**：阶段七 · 大模型应用实战
> **学习目标**：掌握 HuggingFace 开源生态，能用统一接口加载、推理、微调任意开源大模型
> **预估时长**：4-5 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [transformers-library](./01-transformers-library.md) | Transformers 库入门 | `pipeline` 一行完成 NLP 任务、`Auto*` 类自动推断并加载模型、`generate` 采样参数（temperature/top_p）、fp16 与 4-bit 量化加载 |
| 02 | [datasets](./02-datasets.md) | Datasets 库 | 从 Hub/本地一行加载、Arrow 后端零拷贝与内存映射、`.map(batched=True)` 向量化预处理、`streaming=True` 处理 TB 级语料不 OOM |
| 03 | [trainer-api](./03-trainer-api.md) | Trainer API | `TrainingArguments` 一份配置封装完整训练循环、`compute_metrics` 自定义指标、`DataCollatorWithPadding` 动态填充、EarlyStopping 与自定义 loss |

---

## 🔑 知识点详解

### 01 · Transformers 库

- **核心概念**：Transformers 把"分词器 + 模型 + 生成逻辑"封装成统一 API；`Auto*` 类根据模型名自动推断架构并加载权重，一套代码换个名字就能切 GPT-2 / Qwen / Mistral。
- **关键 API**：
  - `pipeline("task", model=...)` — 最快上手，一行完成生成/分类/问答/翻译。
  - `AutoTokenizer.from_pretrained(name)` + `AutoModelForCausalLM.from_pretrained(name)` — 手动模式，控制更细。
  - `model.generate(**inputs, max_new_tokens=, temperature=, top_p=, do_sample=True)` — 采样生成；`temperature` 低→确定保守、高→发散随机。
- **易错点**：
  - 大模型 4-bit 量化加载**不能再直传 `load_in_4bit=True`**，须用 `BitsAndBytesConfig(load_in_4bit=True, ...)` 传给 `quantization_config`（新版 API）。
  - GPT-2 等模型无 `pad_token`，批量生成前需 `tokenizer.pad_token = tokenizer.eos_token`，否则报错。
- **Java 视角**：`Auto*` 类 ≈ Spring 的自动配置 + 依赖注入——你只声明"要什么模型"，框架负责推断类型并装配；Hub ≈ Maven Central。
- **前置**：阶段六的 Transformer 原理；Python 基础。

### 02 · Datasets 库

- **核心概念**：基于 Apache Arrow 的列式 + 内存映射数据层，把"加载、切分、预处理、喂给 DataLoader"标准化，超大数据集也不占内存。
- **关键 API**：
  - `load_dataset(name, split="train[:1000]", streaming=True)` — 从 Hub/本地加载，支持切片与流式。
  - `ds.map(fn, batched=True, num_proc=4)` — **批量向量化预处理**，比逐条快一个数量级。
  - `ds.set_format("torch", columns=[...])` — 转成 PyTorch tensor，直接对接 `DataLoader`。
- **易错点**：
  - 忘了 `batched=True`：逐条分词会慢 10-100 倍。
  - 部分老数据集脚本已迁移（如裸 `"c4"` → `allenai/c4` 且需传语言子集），必要时加 `trust_remote_code=True`。
- **Java 视角**：`load_dataset` ≈ Spring Data 的统一 Repository 抽象——屏蔽底层存储（本地/远程/多格式），你只面向数据集对象编程。
- **前置**：01（分词器用于 `.map`）。

### 03 · Trainer API

- **核心概念**：`Trainer` 把梯度计算、优化器更新、学习率调度、混合精度、分布式、日志、断点保存、评估全封装进一个"训练引擎"，你只提供模型 + 数据 + 一份 `TrainingArguments`。
- **关键 API**：
  - `TrainingArguments(output_dir, num_train_epochs, per_device_train_batch_size, learning_rate, eval_strategy="epoch", fp16=True, load_best_model_at_end=True)`。
  - `Trainer(model, args, train_dataset, eval_dataset, data_collator, compute_metrics).train()`。
- **易错点**：
  - `evaluation_strategy` 自 transformers 4.41 起**改名为 `eval_strategy`**（旧名在 4.46 移除），4.41 以下仍用旧名——别混用。
  - `load_best_model_at_end=True` 要求 `eval_strategy` 与 `save_strategy` 一致（都按 epoch 或都按 steps），否则报错。
  - 自定义 `compute_loss` 新签名带 `num_items_in_batch=None` 参数。
- **Java 视角**：`Trainer` ≈ Spring Boot 的"约定优于配置"——给一份配置就跑起整条流程；`TrainingArguments` ≈ `application.yml`。
- **前置**：01、02（模型与数据集）。

---

## 🎯 学习要点

- **三层上手路径**：先用 `pipeline` 一行跑通 → 再用 `Auto*` 手动加载控制生成参数 → 最后用 `Trainer` 训练，逐层揭开封装。
- **HuggingFace 是本阶段的公共地基**：`transformers`（推理）、`datasets`（数据）、`peft`（LoRA）、`trl`（SFT）、`accelerate`（分布式）会在 RAG、微调各章反复出现，务必先熟。
- **动手对比采样参数**：用 GPT-2 跑 `temperature=0.1` vs `1.5`，亲眼看确定性 vs 发散性的差别，建立对生成参数的直觉。
- **量化加载用新 API**：想在消费级卡跑 7B，记 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` + `device_map="auto"` 这套组合，而非已过时的直传参数。
- **预处理必开 `batched=True`**：这是数据管线最容易被忽视、收益却最大的一处提速。
- **训练配置先跑小样本**：先用 `split="train[:1000]"` 与少量 epoch 验证整条流程能跑通，再放全量，避免踩坑后浪费算力。

---

## 🔗 关联

- **上一模块**：本模块是阶段七起点，原理承接 [06-llm-core-technology](../../06-llm-core-technology/)。
- **下一模块**：[02-prompt-engineering](../02-prompt-engineering/) — 学会加载模型后，用 Prompt 把它调到最优。
- **本阶段总览**：[阶段七 README](../README.md)
- **相关 Day**：[Day 1 第一次模型调用](../../agent-course/Day-01-first-call.md) · [Day 2 模型参数与流式输出](../../agent-course/Day-02-model-params.md)。
