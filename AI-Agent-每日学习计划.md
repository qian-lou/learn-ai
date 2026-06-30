# AI Agent 开发 · 每日学习计划（70 天）

> **节奏假设**：每天约 2 小时、每周 5 个学习日（周末留作缓冲/复习）。
> **总量**：**70 个学习日 ≈ 14 周 ≈ 约 3.5~4 个月日历时间**（含周末缓冲）。
> **如何缩放**：
> - 每天只有 1 小时 → 日历时间翻倍（每个"学习日"拆成两天）。
> - 每天能投 3~4 小时 → 可压到 ~8~9 周。
> - 你工程底子好，Phase 0 觉得太简单可直接跳到 Day 6。
>
> **用法**：每天做完打个勾 ✅。每个 Phase 末尾都有一个"阶段项目",务必产出能跑的东西。

---

## 🟢 Week 1 · Phase 0：打地基（Day 1–5）

- **Day 1 — 环境 + 第一次调用**：注册 API key，装 SDK，跑通第一个 LLM 调用。理解 token、context window、计费。
- **Day 2 — 模型参数**：temperature / top_p / max_tokens / stop。实现流式输出。同一 prompt 调不同参数对比结果。
- **Day 3 — Prompt 基础**：system/user/assistant 三种角色、few-shot、思维链(CoT)。反复改写 prompt 观察差异。
- **Day 4 — 结构化输出**：让模型稳定返回 JSON，用 Pydantic 校验。理解企业为什么必须要强类型输出。
- **Day 5 — 复盘 + 小项目**：合成一个「流式对话 CLI」(多轮 + 能识别结构化指令)。写下你自己的 Agent 心智模型一句话定义。

---

## 🟢 Week 2 · Phase 1：单 Agent + 工具（上）（Day 6–10）

- **Day 6 — Tool Calling 原理**：模型如何决定调用函数，function schema 怎么定义，调用返回怎么回填。
- **Day 7 — 上手厂商 SDK**：用 OpenAI Agents SDK 或 Claude Agent SDK 定义第一个 tool，跑通一次完整工具调用。
- **Day 8 — 多工具 Agent**：实现 3~4 个工具（计算 / 查外部 API / 读文件 / 查数据库），观察模型如何选工具。
- **Day 9 — ReAct 循环**：理解「思考→行动→观察→再思考」。让 agent 连续调多个工具完成一个任务。
- **Day 10 — 错误处理**：工具抛异常、参数非法、超时怎么办。让 agent 优雅降级而不是崩。

## 🟢 Week 3 · Phase 1：单 Agent + 工具（下）（Day 11–15）

- **Day 11 — 记忆与会话状态**：短期记忆、对话历史管理、上下文超长时的截断策略。
- **Day 12 — 工具 + 结构化输出结合**：把工具结果转成强类型对象返回，端到端可校验。
- **Day 13 — ☕ Java 对照日 1**：Spring Boot + Spring AI，把工具映射到带 `@Description` 的 Bean，做同一个工具 Agent。
- **Day 14 — ☕ Java 对照日 2**：体会 Spring AI 的 Advisor 模式 vs Python SDK 的差异，记下两种生态各自的优劣。
- **Day 15 — 🎯 阶段项目**：完成「数据查询 Agent」——查真实数据、返回结构化结果、带错误处理。复盘。

---

## 🟢 Week 4 · Phase 2：RAG（上）（Day 16–20）

- **Day 16 — Embedding 原理**：文本→向量、余弦相似度。亲手算一次两段文本的相似度。
- **Day 17 — 向量库入门**：pgvector 安装配置（复用 Postgres，最易上手），存取向量。
- **Day 18 — 文档切分(chunking)**：固定大小 vs 语义切分、overlap、chunk size 如何影响检索质量。
- **Day 19 — 嵌入 pipeline**：一批文档 读入→切分→嵌入→入库，跑通 ETL。
- **Day 20 — 检索**：query 嵌入→相似度检索→取 top-k。调 k 值和阈值看效果变化。

## 🟢 Week 5 · Phase 2：RAG（下）（Day 21–25）

- **Day 21 — 完整 RAG 链路**：检索结果拼进 prompt→生成答案。跑出第一个能用的 RAG。
- **Day 22 — 引用与溯源**：让答案带出处，明确"不知道就说不知道",压低幻觉。
- **Day 23 — 进阶检索**：metadata 过滤、混合检索(关键词+语义)、重排序(rerank)。
- **Day 24 — ☕ Java 对照日**：用 Spring AI ETL + QuestionAnswerAdvisor 或 LangChain4j EmbeddingStore 做同样的 RAG。
- **Day 25 — 🎯 阶段项目**：完成「文档问答系统(带引用)」。整理 RAG 常见坑清单。

---

## 🟢 Week 6 · Phase 3：编排与多 Agent（一）（Day 26–30）

- **Day 26 — LangGraph 入门**：状态机心智模型 —— node / edge / state schema。建第一个 graph。
- **Day 27 — 条件分支**：conditional edges，根据状态决定走哪条边。
- **Day 28 — 循环**：让 graph 循环执行直到满足条件（用 graph 重写之前的 ReAct 循环）。
- **Day 29 — 状态持久化**：checkpointer（SQLite / Postgres），实现断点续跑。
- **Day 30 — 人在回路(HITL)**：在关键步骤暂停、等人确认再继续。

## 🟢 Week 7 · Phase 3：编排与多 Agent（二）（Day 31–35）

- **Day 31 — 多 Agent 概念**：编排模型对比 —— graph / 角色(role) / handoff 三种范式。
- **Day 32 — 搭多 Agent 流水线**：用 LangGraph 实现 研究员→分析师→报告生成。
- **Day 33 — Agent 间通信与共享状态**：怎么传递中间结果、避免状态污染。
- **Day 34 — CrewAI 速览**：用角色化框架快速搭多 agent，对比 LangGraph 的取舍。
- **Day 35 — 复盘 + 重构**：把多 agent 系统整理干净，抽出可复用结构。

## 🟢 Week 8 · Phase 3：编排与多 Agent（三）（Day 36–40）

- **Day 36 — 健壮性**：重试策略、死循环检测、token 超限处理、整体超时。
- **Day 37 — MCP 协议**：理解 Model Context Protocol，接一个现成的 MCP server 当工具源。
- **Day 38 — A2A 协议**：Agent 间通信标准，搞清它解决什么问题、何时用。
- **Day 39 — 复杂工具编排**：让 agent 组合多个工具 + 子 agent 完成一个复杂任务。
- **Day 40 — 性能**：并行工具调用、流式输出中间结果。

## 🟢 Week 9 · Phase 3：阶段项目（Day 41–45）

- **Day 41 — 项目设计**：设计「自动化研究 Agent」的架构（输入→检索→分析→产出报告）。
- **Day 42 — 实现①**：搭主干 graph + 状态 schema。
- **Day 43 — 实现②**：多 agent 协作 + 工具接入。
- **Day 44 — 实现③**：加断点续跑 + HITL + 错误恢复。
- **Day 45 — 🎯 完成 + 复盘**：这是简历主力项目之一,务必能完整跑通并讲清楚。

---

## 🟢 Week 10 · Phase 4：生产化 —— 可观测 + 评估（Day 46–50）

- **Day 46 — 可观测性概念**：为什么 Agent 必须 trace、不 trace 会死在什么坑上。
- **Day 47 — 接入 tracing**：LangSmith / Langfuse / OpenTelemetry 任选一个接上。
- **Day 48 — 看 trace 调试**：逐步看 token、延迟、成本、每次工具调用，定位一个真实 bug。
- **Day 49 — 评估(Eval)入门**：怎么量化 agent 质量，构建一个测试集。
- **Day 50 — 写 eval**：跑准确率 / 幻觉率，建立回归测试,改一处不再担心崩别处。

## 🟢 Week 11 · Phase 4：成本 + 安全（Day 51–55）

- **Day 51 — 成本优化**：缓存、token 管理、控制上下文膨胀。
- **Day 52 — 智能路由**：简单任务用小模型、难任务用大模型，按难度选型。
- **Day 53 — 安全①**：Prompt 注入的原理与防御。
- **Day 54 — 安全②**：工具权限边界、输出校验、guardrails、敏感数据处理。
- **Day 55 — OWASP for LLM**：过一遍清单,逐条自查自己的 agent。

## 🟢 Week 12 · Phase 4：部署 + 阶段项目（Day 56–60）

- **Day 56 — 部署**：把 agent 包成服务（FastAPI），容器化。
- **Day 57 — 生产关注点**：限流、超时、并发、降级、健康检查。
- **Day 58 — 阶段项目①**：给前面的 agent 加上完整监控。
- **Day 59 — 阶段项目②**：加上 eval + 安全防护。
- **Day 60 — 🎯 完成 + 复盘**：这是"能上线"的证明,面试里最值钱的部分。

---

## 🟢 Week 13 · Phase 5：双栈架构（上）（Day 61–65）

- **Day 61 — 架构设计**：Python 编排层 + Java 服务层,定清楚两边接口(REST / MCP)。
- **Day 62 — Java 服务层**：Spring Boot 提供业务服务 / 数据 / 鉴权 / 可观测性。
- **Day 63 — Python Agent 层**：LangGraph 编排,通过 API/MCP 调用 Java 服务。
- **Day 64 — 端到端打通**：让混合系统完整跑通一个真实场景。
- **Day 65 — 可观测性贯通两端**：trace 能从 Python 一路追到 Java。

## 🟢 Week 14 · Phase 5 收尾 + 作品集（Day 66–70）

- **Day 66 — 完善①**：处理混合系统的边界情况与失败路径。
- **Day 67 — 完善②**：压测、加缓存、优化延迟。
- **Day 68 — 整理作品集**：把三个主力项目(RAG问答 / 研究Agent / 混合系统)收拢。
- **Day 69 — 文档与表达**：写 README + 架构图,练习把每个项目的 eval / 可观测 / 安全讲清楚。
- **Day 70 — 🎯 总复盘 + 规划**：决定深挖方向 A(Java 体系内) 还是 B(纯 AI 工程),制定下一步(求职 / 深造)。

---

## 📊 进度总览

| Phase | 内容 | 天数 | 累计 |
|---|---|---|---|
| Phase 0 | 打地基 | Day 1–5 | 5 |
| Phase 1 | 单 Agent + 工具 | Day 6–15 | 15 |
| Phase 2 | RAG | Day 16–25 | 25 |
| Phase 3 | 多 Agent 编排 | Day 26–45 | 45 |
| Phase 4 | 生产化(Eval/监控/安全) | Day 46–60 | 60 |
| Phase 5 | 双栈架构 + 作品集 | Day 61–70 | 70 |

**合计 70 个学习日。** 主力难点在 Phase 3–4（Day 26–60),那 35 天别赶进度,它们才是和"只会写 demo 的人"拉开差距的地方。

---

## ✅ 四个里程碑（每个都是一个能讲的作品）

1. **Day 15** — 数据查询 Agent（工具调用 + 结构化输出）
2. **Day 25** — 文档问答系统（完整 RAG，带引用）⭐ 面试高频
3. **Day 45** — 自动化研究 Agent（多 Agent + 状态）⭐ 主力项目
4. **Day 60** — 生产级 Agent（监控 + Eval + 安全）⭐⭐ 最值钱
5. **Day 70** — Java + Python 混合架构系统 ⭐⭐ 你的差异化王牌
