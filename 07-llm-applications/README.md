# 阶段七：大模型应用实战

> **预估周期**：3-4 周
> **核心目标**：从"会训模型"跨到"会用大模型交付应用"——打通 HuggingFace 生态、Prompt 工程、RAG、LoRA 微调、LangChain 五条主线。
> **学完能做**：本地跑通开源大模型推理与微调、搭一套带引用的文档问答（RAG）、写出稳定的结构化输出 Prompt、用 LangChain/LangGraph 拼出带工具的 Agent 应用。

---

## 🗺️ 阶段学习路径

```
01 HuggingFace 生态        → 会加载/推理/训练任意开源模型（地基工具）
        ↓
02 Prompt 工程             → 不改参数，靠输入把模型调到最优（性价比最高的技能）
        ↓
03 RAG 检索增强            → 给模型接外部知识库，解决知识过时 + 幻觉（企业落地第一方案）
        ↓
04 LoRA / QLoRA 微调       → 单卡定制专属模型，改的是模型本身（0.1% 参数达到近全参效果）
        ↓
05 LangChain 框架          → 把 Prompt + RAG + 工具编排成完整应用（综合收口）
```

> **主线关系一句话**：Prompt / RAG 是"不动模型"的外挂增强，微调是"改动模型"的内化定制，两者互补而非替代；LangChain 是把前四者组装成产品的胶水层。**先能不改模型就不改**——Prompt → RAG → 微调，成本从低到高。

---

## 📋 模块大纲

### [01-huggingface](./01-huggingface/) — HuggingFace 生态

AI 开源生态的核心平台（"AI 界的 GitHub + Maven"），用统一接口加载、推理、训练 50 万+ 预训练模型。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [transformers-library](./01-huggingface/01-transformers-library.md) | Transformers 库入门 | `pipeline` 一行推理、`Auto*` 类自动加载、`generate` 采样参数、4-bit 量化加载 |
| 02 | [datasets](./01-huggingface/02-datasets.md) | Datasets 库 | Arrow 后端零拷贝、`.map(batched=True)` 向量化预处理、`streaming=True` 免 OOM |
| 03 | [trainer-api](./01-huggingface/03-trainer-api.md) | Trainer API | `TrainingArguments` 一份配置封装完整训练循环、`compute_metrics`、混合精度 |

---

### [02-prompt-engineering](./02-prompt-engineering/) — Prompt 工程

不改模型参数，仅靠设计输入把同一个模型调到最好——LLM 落地性价比最高的技能。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [prompt-basics](./02-prompt-engineering/01-prompt-basics.md) | Prompt 基础与原则 | Zero/Few-shot、System Prompt 角色设定、JSON 输出约束、六大设计原则 |
| 02 | [advanced-techniques](./02-prompt-engineering/02-advanced-techniques.md) | 高级技巧 | CoT 思维链、Self-Consistency 多数投票、ReAct、推理模型与 Structured Outputs |

---

### [03-rag](./03-rag/) — 检索增强生成（RAG）

先检索外部知识、再让模型基于真实资料作答——企业落地大模型最常用、面试最高频的方案。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [rag-basics](./03-rag/01-rag-basics.md) | RAG 基础原理 | 加载→切分→嵌入→检索→生成五步、RAG vs 微调、reranker、agentic RAG |
| 02 | [vector-databases](./03-rag/02-vector-databases.md) | 向量数据库 | Faiss/Chroma/Qdrant、Flat/IVF/HNSW/PQ 索引、余弦/L2/内积度量 |
| 03 | [rag-practice](./03-rag/03-rag-practice.md) | RAG 实战项目 | Hybrid Search + Re-ranking、RAGAS 四维评估、幻觉与检索失败排障 |

---

### [04-fine-tuning](./04-fine-tuning/) — 模型微调

用 LoRA/QLoRA 只训练 0.1% 参数、单张消费级卡定制专属模型。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [lora-qlora](./04-fine-tuning/01-lora-qlora.md) | LoRA 与 QLoRA | 低秩分解 `h=Wx+BAx`、`target_modules`、4-bit NF4 量化、DoRA/unsloth/all-linear |
| 02 | [instruction-tuning](./04-fine-tuning/02-instruction-tuning.md) | 指令微调（SFT） | Alpaca/ShareGPT/OpenAI 三种数据格式、只对 Response 算 loss、质量>数量 |

---

### [05-langchain](./05-langchain/) — LangChain 框架

用 Chain/Agent/Memory/Retriever 抽象，把前四模块编排成可交付的 LLM 应用。

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [langchain-basics](./05-langchain/01-langchain-basics.md) | LangChain 核心概念 | LCEL 管道 `prompt\|llm\|parser`、Memory 会话记忆、Pydantic 结构化输出 |
| 02 | [agents-and-tools](./05-langchain/02-agents-and-tools.md) | Agents 与工具集成 | `@tool` 自定义工具、`create_react_agent`、Function Calling、HITL 确认 |
| 03 | [full-application](./05-langchain/03-full-application.md) | 完整应用开发实战 | FastAPI 流式 SSE、Gradio/Streamlit UI、限流/计费/日志上线清单 |

---

## 🎯 阶段学习要点

- **先外挂、后微调**：能用 Prompt 解决就别上 RAG，能用 RAG 就别急着微调——成本、可维护性依次上升，别一上来就训模型。
- **RAG 与微调是互补而非二选一**：微调教"风格与领域术语"，RAG 供"最新事实"，生产系统常常两者叠加。
- **HuggingFace 是贯穿全阶段的地基**：`Auto*` 类、`Trainer`、`datasets`、`peft`、`trl` 在推理、RAG 嵌入、微调各环节反复出现，务必先打牢。
- **检索质量决定 RAG 上限**：切分（chunk_size/overlap）、嵌入模型、Top-K、reranker 四处任何一环拖后腿，生成再强也救不回来。
- **2026 现代写法要跟上**：`initialize_agent`→`create_react_agent`、`RetrievalQA`→LCEL 检索链、`load_in_4bit` 直传→`BitsAndBytesConfig`、`evaluation_strategy`→`eval_strategy`、TRL `tokenizer`→`processing_class`；SDK 迭代快，运行前对齐本机版本。
- **安全默认在线**：工具执行绝不 `eval()` 不可信输入（用 AST 白名单）、SQL Agent 用只读账号、危险操作加 Human-in-the-loop、上线前过一遍限流/计费/日志清单。

---

## 🔗 关联

- **上一阶段**：[06-llm-core-technology](../06-llm-core-technology/) — Transformer/预训练/训练技术，是本阶段"用模型"的原理底座。
- **配套实战**：[agent-course/](../agent-course/) — 70 天每日课程把本阶段知识拆成可跑的 Day：
  - Prompt → [Day 3-4](../agent-course/Day-03-prompt-basics.md)
  - RAG → [Day 16-25](../agent-course/Day-16-embedding-basics.md)（Day 25 文档问答 Capstone）
  - Agent/工具 → [Day 6-15](../agent-course/Day-06-tool-calling-basics.md)、[Day 26-45](../agent-course/Day-26-langgraph-intro.md)
  - 上线/安全 → [Day 46-60](../agent-course/Day-46-observability-concepts.md)
- **课程总纲**：[AI-Agent-学习路线](../AI-Agent-学习路线.md) · [每日学习计划](../AI-Agent-每日学习计划.md)
