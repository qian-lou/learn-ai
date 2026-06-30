# Java + Python 程序员转 AI Agent 开发 · 学习路线

> 适用人群：有 Java 工程经验、同时会 Python，想转 AI Agent 开发。
> 路线同时覆盖两条方向：**A) 在现有 Java 体系里加 Agent 能力**、**B) 转型纯 AI 工程师**。
> 总时长建议：3~5 个月（每周 8~12 小时）。

---

## 一、先做知识盘点：你已有什么 / 需要补什么

### ✅ 你已经具备的优势（别浪费）
- **工程能力**：API 设计、并发、错误处理、日志、部署、CI/CD —— Agent 生产化最缺的就是这些，你是优势方。
- **强类型思维**：结构化输出、Schema 校验是企业级 Agent 的核心需求，你的直觉天然契合。
- **后端架构**：RAG、向量库、消息队列、缓存这些都是后端范畴，迁移成本低。
- **双语言**：能驾驭"Python 编排层 + Java 服务层"的混合架构，这是很多团队的真实形态。

### 🔧 需要补的核心知识点（按重要性排序）

| 知识点 | 为什么要补 | 深度要求 |
|---|---|---|
| LLM 基础原理 | 理解 token、上下文窗口、temperature、采样、为什么会幻觉 | 中（够用即可，不必训模型） |
| Prompt Engineering | 系统提示、few-shot、结构化输出、思维链 | 高（每天都用） |
| Tool / Function Calling | Agent 的灵魂：让模型调用你的代码 | 高 |
| Embedding & 向量检索 | RAG 的基础：语义相似度、向量库 | 中高 |
| RAG 全流程 | 企业 Agent 标配：切分→嵌入→检索→增强→生成 | 高 |
| Agent 循环与编排 | ReAct、planning、多 Agent 协作、状态管理 | 高 |
| MCP / A2A 协议 | 2026 工具互操作 & Agent 间通信的新标准 | 中 |
| 评估与可观测性 | Agent 上线后能否稳定工作的决定因素 | 高（最易被忽视、最值钱） |
| AI 安全与防护 | Prompt 注入、越权、数据泄露、成本失控 | 中 |

### ⚠️ 不必一开始就补的（按需，别陷进去）
- 深度学习数学（反向传播、梯度下降）—— 做集成/Agent **用不到**，想做模型研究再说。
- 自己训练/微调大模型 —— 99% 的 Agent 工作是"用模型"而非"造模型"。
- 复杂的 ML 理论 —— 了解概念即可，不必系统学。

> 一句话原则：**模型训练才需要 Python+数学+GPU；模型集成（也就是 Agent）需要的是工程能力，你已经有了。**

---

## 二、分阶段学习路线

### 🟢 Phase 0：打地基（1~2 周）

**目标**：搞懂"Agent 到底是什么"，跑通第一次 LLM 调用。

要做的事：
1. 直接用 OpenAI / Anthropic / Gemini 的原生 SDK，写一个最简单的对话脚本，跑通流式输出。
2. 理解几个核心参数：token 计费、context window、temperature、system prompt。
3. 建立心智模型：**Agent = LLM + 工具 + 记忆 + 一个"思考→调工具→观察→再思考"的循环**。

补的知识点：LLM 基础原理、Prompt Engineering 入门。

产出：一个能流式对话的命令行小工具。

---

### 🟢 Phase 1：单 Agent + 工具调用（2~3 周）

**目标**：让模型自己决定调用你写的函数。这是从"调 API"到"Agent"的第一道分水岭。

要做的事：
1. 用厂商官方 SDK（**OpenAI Agents SDK** 或 **Claude Agent SDK**）做一个带 Tool Calling 的 Agent，最快看到效果。
2. 实现 3~4 个工具：查数据库、调外部 API、做计算、读文件。
3. 体会"模型自己决定调哪个工具、调几次、怎么串结果"。
4. 强制结构化输出（用 Pydantic / JSON Schema），把自由文本变成强类型对象。

补的知识点：Tool Calling、结构化输出、ReAct 模式。

> **Java 方向并行任务**：在 Spring Boot 项目里用 **Spring AI** 做同样一件事，把工具映射到带 `@Description` 的 Spring Bean，对比两种生态的体感。

产出：一个能查真实数据、返回结构化结果的工具型 Agent。

---

### 🟢 Phase 2：RAG（检索增强生成）（2~3 周）

**目标**：让 Agent 基于"你的知识库"回答问题。这是面试和实际项目里最高频的需求。

要做的事：
1. 理解 Embedding：把文本变向量、用余弦相似度找最相关内容。
2. 跑通完整 RAG 链路：文档切分 → 嵌入 → 存向量库 → 检索 → 增强 prompt → 生成。
3. 用一个向量库（**pgvector**（最易上手，复用 Postgres）/ Qdrant / Milvus / Pinecone 任选）。
4. 进阶：metadata 过滤、混合检索（关键词+语义）、重排序、引用来源。

补的知识点：Embedding、向量数据库、chunking 策略、Agentic RAG。

> **Java 方向并行任务**：Spring AI 有 DocumentReader/Transformer/Writer 的 ETL 框架 + QuestionAnswerAdvisor；LangChain4j 有 DocumentSplitter/EmbeddingStore/ContentRetriever。任选一套实现同样的 RAG。

产出：一个"问答你自己的文档"的应用，回答带出处引用。

---

### 🟢 Phase 3：多 Agent 编排 + 状态管理（3~4 周）

**目标**：从"会调一次"升级到"真正能自主完成多步任务"的 Agent 系统。

要做的事：
1. 用 **LangGraph**（生产级首选）把前面的东西重做一遍，加上：状态机、循环、条件分支、人在回路（human-in-the-loop）、checkpointing（断点续跑）。
2. 实现一个多 Agent 协作场景（比如：研究员→分析师→报告生成）。
3. 加错误处理与重试：工具调用失败怎么办、模型循环卡住怎么办、token 爆了怎么办。
4. 了解 MCP（工具互操作）和 A2A（Agent 间通信）协议。

补的知识点：编排模型（graph / 角色 / handoff）、状态持久化、HITL、MCP/A2A。

> 框架选择提示：快速原型可以先用 **CrewAI**（角色化，20~50 行起步），等需要生产级状态控制再迁到 LangGraph —— 这是很多团队的真实迁移路径。

产出：一个能自主完成多步任务、有状态、可断点续跑的多 Agent 系统。

---

### 🟢 Phase 4：生产化 —— 评估 + 可观测性 + 安全（2~3 周）

**目标**：这一阶段决定你和"只会写 demo 的人"的差距。企业最看重、最值钱的部分。

要做的事：
1. **可观测性**：接入 tracing（LangSmith / OpenTelemetry / Langfuse），记录每一步的 token、延迟、成本、工具调用。
2. **评估（Eval）**：建立测试集，量化 Agent 的准确率、幻觉率，回归测试防止改一处崩一片。
3. **成本控制**：缓存、token 管理、智能选模型（简单任务用小模型）。
4. **安全防护**：Prompt 注入防御、工具权限边界、敏感数据处理、输出校验、guardrails。

补的知识点：Eval 方法论、tracing/监控、成本优化、AI 安全（OWASP for LLM）。

产出：一个带完整监控、评估、防护的"可上线"Agent。

---

### 🟢 Phase 5（可选但强烈推荐）：双栈架构（2 周）

**目标**：把你 Java + Python 双修的优势变成作品。

要做的事：
- 搭一个 **Python Agent 编排层 + Java 后端服务层** 的完整系统，通过 API / MCP 协作。
- Python 端负责 Agent 推理与编排，Java 端负责稳定的业务服务、数据、鉴权、可观测性。

产出：一个能在面试里讲清楚的、有架构深度的项目。

---

## 三、两条方向的分叉点

虽然路线大体共用，但侧重点不同：

**方向 A：在 Java 体系里加 Agent（适合留在现公司 / 企业内部场景）**
- 主力框架：Spring AI（Spring Boot 项目）或 LangChain4j（Quarkus / 灵活栈）。
- 重点：和现有系统集成、企业级可观测性、强类型与可审计输出。
- 注意：新项目用 Spring AI 1.1 稳定版；关注 2.0（基于 Spring Boot 4）迁移路径。

**方向 B：转纯 AI 工程师（适合换岗 / 进 AI 团队）**
- 主力框架：LangGraph（生产编排）+ 厂商 SDK + Pydantic AI（喜欢强类型的话）。
- 重点：前沿模型能力、复杂多 Agent、Eval 与可观测性深度、跟 Python 主流社区。
- 招聘市场上"AI 工程师"岗位大多默认 Python，作品集要以 Python 为主。

> 我的建议：**两条线并行走到 Phase 3，再根据兴趣 / 工作机会决定 Phase 4-5 往哪边深挖。** Phase 1-3 的每个阶段都附了"Java 方向并行任务",照着做就能两边都不落下。

---

## 四、框架与工具速查（2026 现状）

**Python 生态**
- LangGraph —— 生产级有状态编排的默认选择（状态机、checkpointing、HITL）。
- CrewAI —— 上手最快的角色化多 Agent，适合原型。
- OpenAI Agents SDK / Claude Agent SDK / Google ADK —— 厂商官方，原型到生产最短路径，但有锁定。
- Pydantic AI —— type-safe，最贴近 Java 直觉。
- LlamaIndex / Haystack —— 文档密集型 RAG。
- LangSmith / Langfuse —— 可观测性与评估。

**Java 生态**
- Spring AI —— Spring Boot 首选，Advisor 模式、自动配置、MCP。
- LangChain4j —— 灵活、20+ 模型支持、agentic 模块；支持 Quarkus/Helidon。
- LangGraph4j —— Java 版有状态多 Agent 图，兼容上面两者。
- Quarkus LangChain4j —— GraalVM 原生编译、内置可观测性。

**向量库**：pgvector（入门首选）、Qdrant、Milvus、Pinecone、Weaviate。

---

## 五、推荐练手项目（由浅入深）

1. **流式对话 CLI** —— Phase 0，跑通基础。
2. **数据查询 Agent** —— Phase 1，工具调用 + 结构化输出。
3. **文档问答系统（带引用）** —— Phase 2，完整 RAG，最实用、最好讲。
4. **自动化研究 Agent** —— Phase 3，多 Agent + 状态 + 断点续跑。
5. **生产级客服 / 运维 Agent** —— Phase 4，带监控、Eval、防护。
6. **Java 后端 + Python Agent 混合系统** —— Phase 5，展示架构能力（你的差异化王牌）。

> 作品集策略：项目 3、4、6 是面试里最有说服力的三个。把"可观测性 + Eval + 安全"讲清楚，比堆功能更能打动人。

---

## 六、学习资源建议
- 各框架**官方文档** —— 这领域变化太快，官方文档永远是第一手，优先于任何教程/视频。
- Anthropic、OpenAI 的 **prompt engineering 与 agent 构建指南**。
- 关注几个高质量信息源持续跟进（框架更新非常频繁，推荐每月扫一次更新日志）。
- 实践 >> 看课。每个 Phase 都务必产出一个能跑的东西。

---

## 七、时间线参考（每周 8~12h）

| 阶段 | 内容 | 周数 |
|---|---|---|
| Phase 0 | 基础 + 首次调用 | 1~2 |
| Phase 1 | 单 Agent + 工具 | 2~3 |
| Phase 2 | RAG | 2~3 |
| Phase 3 | 多 Agent 编排 | 3~4 |
| Phase 4 | 生产化（Eval/监控/安全） | 2~3 |
| Phase 5 | 双栈架构（可选） | 2 |
| **合计** | | **约 12~17 周** |

> 你有工程底子，Phase 0-1 会比一般人快很多；真正的硬骨头在 Phase 3-4，别赶进度，那两段才是拉开差距的地方。
