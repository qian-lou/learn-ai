# 📌 2026 版本快照 / Version Snapshot

> **内容快照日期：2026-07**
>
> 本教材通篇宣称「2026 现代写法」。前沿 SDK 迭代极快，本页把全课程依赖的关键库**锁定版本**与**已知迁移点**集中在一处，作为"对版本号"的统一基准。
>
> **用法**：跑不通时，先回本页对一下版本；教材代码在这些**大版本**上验证过。你的版本更高时，重点看"迁移提示"列——多数报错是 API 改名/参数迁移，而非逻辑变化。**一切以官方 changelog 为准。**

---

## 1. LLM 调用 SDK

| 库 | 锁定大版本 | 迁移提示（版本更高时重点看） |
|----|:---:|------|
| `openai` (Python) | 1.x | 结构化输出用 `client.beta.chat.completions.parse(response_format=模型)`；Responses/Agents SDK 为新范式，注意与旧 `chat.completions` 并存 |
| `anthropic` | 0.4x+ | 视觉输入用 `source(type=base64, media_type=...)`；模型名如 `claude-sonnet-5` |
| `google-genai` | 1.x | 新包名 `google-genai` 取代旧 `google-generativeai` |
| `tiktoken` | 0.7+ | `gpt-4o` 家族用 `o200k_base` 编码，非旧 `cl100k_base` |

## 2. Agent 编排

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `langgraph` | 0.2+ | 动态并行用 `from langgraph.types import Send`；流式 `graph.astream(stream_mode=...)`；`checkpointer` 持久化 API |
| `langchain` | 0.3+ | 各厂商集成已从主包拆出——OpenAI 走 `langchain_openai`，需单独安装；LCEL 管道为主流写法 |
| `langchain-core` | 0.3+ | `Runnable` 接口稳定；`create_retrieval_chain` 输入键为 `input` |
| `crewai` | 0.1x | 角色化 API，与 LangGraph 取舍见 Day-34 |

## 3. RAG / 向量检索

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `pgvector` | 0.7+ | 支持 HNSW 索引；配合 `psycopg`/SQLAlchemy 使用 |
| `sentence-transformers` | 3.x | 本地嵌入模型如 `BAAI/bge-m3`；`encode(normalize_embeddings=True)` |
| `chromadb` | 0.5+ | 内嵌式向量库，快速原型；生产多用 pgvector |
| `rank-bm25` / rerank | — | 混合检索的稀疏侧；rerank 用 cross-encoder |

## 4. 训练 / 微调

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `transformers` | 4.4x | `evaluation_strategy` 自 **4.41 改名 `eval_strategy`**（旧名 4.46 移除）；自定义 `compute_loss` 新签名带 `num_items_in_batch` |
| `trl` | 0.1x+ | `SFTTrainer` 用 **`processing_class` 取代 `tokenizer`**；`max_seq_length` 在 0.20+ 改为 `max_length` |
| `peft` | 0.1x | LoRA/QLoRA/DoRA；`get_peft_model` + `LoraConfig` |
| `bitsandbytes` | 0.4x+ | 4-bit 量化经 `BitsAndBytesConfig(load_in_4bit=True, ...)` 传 `quantization_config`，不再直传 `load_in_4bit` |
| `torch` | 2.x | 混合精度用 **`torch.amp.autocast("cuda")`**，旧 `torch.cuda.amp.autocast()` 自 2.4 废弃；`weights_only=True` 成 `load` 默认 |
| `xgboost` | 2.x | `early_stopping_rounds` 移入**构造器**，`eval_set` 传给 `fit` |
| `lightgbm` | 4.x | 早停用 `callbacks=[early_stopping(...)]` |

## 5. 推理 / 服务 / 部署

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `vllm` | V1 引擎 | PagedAttention、投机解码；V1 架构与 V0 的启动参数有别 |
| `fastapi` | 0.11x | 配合 `pydantic` v2；异步端点 |
| `pydantic` | **v2** | `BaseModel` 校验、`model_dump()`（旧 `.dict()`）；v1 语法需迁移 |
| `docker` / `uvicorn` | — | 部署见 Day-56、08-llm-engineering |

## 6. 可观测 / 评估

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `langfuse` | 2.x/3.x | trace 埋点 + `score()` 回写；对齐 OTel GenAI 语义约定 |
| `ragas` | 0.2+ | RAG 评估（faithfulness/answer_relevancy）；真实调用见 Day-50 |
| `prometheus-client` | 0.2x | `/metrics` 暴露；配合告警见 Day-58 |

## 7. 数据科学基础

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| `numpy` | 2.x | `ndarray.ptp()` 已移除，用 `np.ptp(arr)`；`float`/`int` 别名清理 |
| `pandas` | 2.x | 位置访问用 `s.iloc[0]`，`s[0]` 在 3.x 会 KeyError；`dtype_backend="pyarrow"`（2.0+） |
| `scikit-learn` | 1.5+ | `KMeans(n_init="auto")`；波士顿房价数据集已移除 |
| `imbalanced-learn` | 0.12+ | `SMOTE`/`RandomUnderSampler`，须用 `imblearn.pipeline` 包进 CV 折内 |

## 8. Java 栈（对照日）

| 库 | 锁定大版本 | 迁移提示 |
|----|:---:|------|
| Spring AI | 1.x | `@Tool` 注解映射工具、`ChatClient`、`QuestionAnswerAdvisor` 做 RAG；1.0 后 API 趋稳 |
| Spring Boot | 3.x | JDK 17+ |

---

## 🔗 关联

- **根总览**：[README.md](./README.md) · [OUTLINE.md](./OUTLINE.md)
- **Agent 课程**：[agent-course/README.md](./agent-course/README.md)——每日讲义的「🧰 本阶段工具箱」会标注该阶段的版本注意点，本页是它们的汇总基准。

> **升级策略**：学习期建议锁大版本、按本页对照；真实项目里把关键库写进 `pyproject.toml`/`requirements.txt` 并 pin 到具体小版本，升级时逐一读 changelog 的 Breaking Changes。
