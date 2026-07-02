# 🤖 AI Agent 开发 · 70 天每日课程

> 面向 **有 Java 后端经验、也会 Python** 的工程师，从"会调模型"到"能交付生产级 Agent"。
>
> 这是 [`../AI-Agent-每日学习计划.md`](../AI-Agent-每日学习计划.md) 的配套讲义，每天一份、约 2 小时；底层概念可回查本仓库的 [Python + 大模型主课程](../README.md)。
>
> 🔧 代码跑不通先对版本：关键库锁定版本与迁移点见 [`../VERSIONS.md`](../VERSIONS.md)。

---

## 🧭 怎么用

- **开课前**：先过一遍每日计划里的「📚 开课前置（Day 0 自检）」（[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)）——环境自检 + 成本护栏，缺哪补哪。
- **节奏**：每天 1 份（~2h），每周 5 天，周末复习。完成一天就把下面的 `☑` 勾掉。
- **每份结构统一**：今日目标 & 产出 → 概念（含 Java 类比）→ 跟着做（可运行代码）→ 今日任务（带验收）→ 自测清单 → 延伸 & 关联。
- **代码均为 2026 现代写法**（OpenAI/Claude SDK、LangGraph、pgvector、MCP/A2A、Langfuse、Spring AI 等）；前沿 SDK 迭代快，运行前请对一下本机实装版本。
- **底子好可跳**：Phase 0 觉得太简单可直接从 Day 6 开始。

---

## 🏆 五个里程碑（每个都是一个能写进简历的作品）

| 里程碑 | 作品 | 价值 |
|:---:|---|---|
| **[Day 15](Day-15-data-query-agent.md)** | 数据查询 Agent（工具调用 + 结构化输出） | 入门作品 |
| **[Day 25](Day-25-rag-capstone.md)** | 文档问答系统（完整 RAG，带引用） | ⭐ 面试高频 |
| **[Day 45](Day-45-research-agent-wrapup.md)** | 自动化研究 Agent（多 Agent + 状态） | ⭐ 主力项目 |
| **[Day 60](Day-60-capstone-ship-review.md)** | 生产级 Agent（监控 + Eval + 安全 + 可复用 eval harness） | ⭐⭐ 最值钱 |
| **[Day 70](Day-70-retrospective-and-roadmap.md)** | Java + Python 混合架构系统 | ⭐⭐ 差异化王牌 |

> 每个里程碑按 **三档自评标尺** 勾选：🟩 跑通（demo 端到端不崩）→ 🟨 能讲清取舍（为什么这么设计、替代方案与代价）→ 🟥 能扛住追问（失败路径 / 成本 / 安全 / 规模）。
> 并产出一张 **量化亮点卡**（前后对比指标 + 一句话讲法），如 `eval 准确率 71%→89%`、`p95 延迟 8s→2.3s`、`注入用例 20/20 拦截`。

---

## 🟢 Phase 0 · 打地基（Day 1–5）

- ☐ **[Day 1](Day-01-first-call.md)** — 环境搭建 + 第一次模型调用
- ☐ **[Day 2](Day-02-model-params.md)** — 模型参数 + 流式输出
- ☐ **[Day 3](Day-03-prompt-basics.md)** — Prompt 基础：角色、few-shot、CoT
- ☐ **[Day 4](Day-04-structured-output.md)** — 结构化输出：Pydantic + `parse` 拿强类型对象 + 原生 JSON Schema 严格模式
- ☐ **[Day 5](Day-05-chat-cli.md)** — 复盘 + 小项目：流式对话 CLI

## 🟢 Phase 1 · 单 Agent + 工具（Day 6–15）

- ☐ **[Day 6](Day-06-tool-calling-basics.md)** — Tool Calling 原理：模型如何"调用"你的函数
- ☐ **[Day 7](Day-07-agents-sdk-first-tool.md)** — 上手厂商 SDK：用 Agents SDK 定义第一个 tool
- ☐ **[Day 8](Day-08-multi-tool-agent.md)** — 多工具 Agent：模型如何在 4 个工具里选对的那个
- ☐ **[Day 9](Day-09-react-loop.md)** — ReAct 循环：思考 → 行动 → 观察 → 再思考
- ☐ **[Day 10](Day-10-error-handling.md)** — 工具错误处理：让 Agent 优雅降级，而不是崩
- ☐ **[Day 11](Day-11-memory-and-context.md)** — 记忆与会话状态：上下文超长时怎么办
- ☐ **[Day 12](Day-12-tools-structured-output.md)** — 工具 + 结构化输出结合：端到端强类型可校验
- ☐ ☕ **[Day 13](Day-13-spring-ai-tools.md)** — Java 对照日 1：Spring AI 把工具映射成带 `@Tool` 的 Bean
- ☐ ☕ **[Day 14](Day-14-spring-ai-advisors.md)** — Java 对照日 2：Advisor 模式 vs Python SDK
- ☐ 🎯 **[Day 15](Day-15-data-query-agent.md)** — 阶段项目：数据查询 Agent

## 🟢 Phase 2 · RAG（Day 16–25）

- ☐ **[Day 16](Day-16-embedding-basics.md)** — Embedding 原理与余弦相似度
- ☐ **[Day 17](Day-17-pgvector-intro.md)** — 向量库入门：pgvector（复用 Postgres）
- ☐ **[Day 18](Day-18-chunking.md)** — 文档切分（Chunking）：固定 vs 语义 + overlap
- ☐ **[Day 19](Day-19-embedding-etl.md)** — 嵌入 ETL Pipeline：读入 → 切分 → 嵌入 → 入库
- ☐ **[Day 20](Day-20-retrieval-topk.md)** — 检索：query 嵌入 → 相似度搜索 → top-k 调参
- ☐ **[Day 21](Day-21-full-rag-chain.md)** — 完整 RAG 链路：检索 → 拼 Prompt → 生成
- ☐ **[Day 22](Day-22-citation-grounding.md)** — 引用与溯源：让答案带出处、不知道就说不知道
- ☐ **[Day 23](Day-23-advanced-retrieval.md)** — 进阶检索：metadata 过滤 + 混合检索 + rerank
- ☐ ☕ **[Day 24](Day-24-java-rag-spring-ai.md)** — Java 对照日：Spring AI ETL + QuestionAnswerAdvisor 做 RAG
- ☐ 🎯 **[Day 25](Day-25-rag-capstone.md)** — 阶段项目：文档问答系统（带引用）+ RAG 常见坑清单

## 🟢 Phase 3 · 多 Agent 编排（Day 26–45）

- ☐ **[Day 26](Day-26-langgraph-intro.md)** — LangGraph 入门：把 Agent 想成状态机
- ☐ **[Day 27](Day-27-conditional-edges.md)** — 条件分支：让边会思考
- ☐ **[Day 28](Day-28-graph-react-loop.md)** — 循环：用 graph 重写 ReAct
- ☐ **[Day 29](Day-29-checkpointer-persistence.md)** — 状态持久化：checkpointer 与断点续跑
- ☐ **[Day 30](Day-30-human-in-the-loop.md)** — 人在回路（HITL）：暂停、等确认、再继续
- ☐ **[Day 31](Day-31-multi-agent-paradigms.md)** — 多 Agent 三范式：graph / role / handoff
- ☐ **[Day 32](Day-32-multi-agent-pipeline.md)** — 搭多 Agent 流水线：研究员 → 分析师 → 报告
- ☐ **[Day 33](Day-33-agent-communication-shared-state.md)** — Agent 间通信与共享状态：怎么传、怎么不污染
- ☐ **[Day 34](Day-34-crewai-overview.md)** — CrewAI 速览：角色化框架，对比 LangGraph 取舍
- ☐ **[Day 35](Day-35-refactor-reusable.md)** — 复盘 + 重构：把多 Agent 系统抽成可复用结构
- ☐ **[Day 36](Day-36-robustness.md)** — Agent 健壮性：重试 / 死循环 / token 超限 / 整体超时 + 幂等与重放安全
- ☐ **[Day 37](Day-37-mcp-protocol.md)** — MCP 协议：接一个现成 MCP server 当工具源
- ☐ **[Day 38](Day-38-a2a-protocol.md)** — A2A 协议：Agent 间通信标准，解决什么问题
- ☐ **[Day 39](Day-39-tool-orchestration.md)** — 复杂工具编排：工具 + 子 agent 完成一个复杂任务
- ☐ **[Day 40](Day-40-performance.md)** — 性能：并行工具调用 + 流式中间结果
- ☐ **[Day 41](Day-41-research-agent-design.md)** — 阶段项目①：自动化研究 Agent 架构设计
- ☐ **[Day 42](Day-42-research-agent-graph.md)** — 阶段项目②：搭主干 graph + 状态 schema
- ☐ **[Day 43](Day-43-research-agent-multiagent.md)** — 阶段项目③：多 Agent 协作 + 工具接入
- ☐ **[Day 44](Day-44-research-agent-resilience.md)** — 阶段项目④：断点续跑 + HITL + 错误恢复
- ☐ 🎯 **[Day 45](Day-45-research-agent-wrapup.md)** — 阶段项目完成 + 复盘：自动化研究 Agent

## 🟢 Phase 4 · 生产化（Day 46–60）

- ☐ **[Day 46](Day-46-observability-concepts.md)** — 可观测性概念：为什么 Agent 必须 trace
- ☐ **[Day 47](Day-47-tracing-setup.md)** — 接入 tracing：Langfuse / LangSmith / OTel
- ☐ **[Day 48](Day-48-trace-debugging.md)** — 看 trace 调试：用 token / 延迟 / 成本定位真实 bug
- ☐ **[Day 49](Day-49-eval-intro.md)** — 评估（Eval）入门：把"它答得好不好"变成数字 + 数据集三来源 / benchmark 认知
- ☐ **[Day 50](Day-50-writing-evals.md)** — 写 eval：准确率、幻觉率、回归测试与 LLM-as-judge + 轨迹级评估
- ☐ **[Day 51](Day-51-cost-optimization.md)** — 成本优化：缓存、token 管理、控制上下文膨胀
- ☐ **[Day 52](Day-52-model-routing.md)** — 智能路由：简单任务用小模型，难任务用大模型 + 推理模型选型
- ☐ **[Day 53](Day-53-prompt-injection.md)** — 安全①：Prompt 注入的原理与防御 + 间接注入
- ☐ **[Day 54](Day-54-guardrails-and-permissions.md)** — 安全②：工具权限边界、输出校验、guardrails、敏感数据
- ☐ **[Day 55](Day-55-owasp-llm-top10.md)** — OWASP for LLM：逐条自查你的 Agent
- ☐ **[Day 56](Day-56-deploy-fastapi-docker.md)** — 部署：把 Agent 包成 FastAPI 服务 + Docker
- ☐ **[Day 57](Day-57-production-concerns.md)** — 生产关注点：限流 · 超时 · 并发 · 降级 · 健康检查
- ☐ **[Day 58](Day-58-capstone-monitoring.md)** — 阶段项目①：给 Agent 服务加完整监控
- ☐ **[Day 59](Day-59-capstone-eval-security.md)** — 阶段项目②：加 Eval 回归测试 + 安全防护 + 数据飞轮 / CI 门禁
- ☐ 🎯 **[Day 60](Day-60-capstone-ship-review.md)** — 完成 + 复盘：交付一个"能上线"的 Agent 服务

## 🟢 Phase 5 · 双栈架构 + 作品集（Day 61–70）

- ☐ **[Day 61](Day-61-dual-stack-architecture.md)** — 双栈架构设计：Python 编排层 + Java 服务层
- ☐ **[Day 62](Day-62-java-service-layer.md)** — Java 服务层：Spring Boot 提供业务 / 数据 / 鉴权 / 可观测
- ☐ **[Day 63](Day-63-python-agent-layer.md)** — Python Agent 层：LangGraph 经 REST / MCP 调 Java 服务
- ☐ **[Day 64](Day-64-end-to-end-integration.md)** — 端到端打通：让双栈系统跑通一个真实场景
- ☐ **[Day 65](Day-65-observability-across-stacks.md)** — 可观测性贯通两端：trace 从 Python 追到 Java
- ☐ **[Day 66](Day-66-harden-edge-cases.md)** — 完善①：混合系统的边界情况与失败路径
- ☐ **[Day 67](Day-67-load-cache-latency.md)** — 完善②：压测 + 加缓存 + 优化延迟
- ☐ **[Day 68](Day-68-portfolio-assembly.md)** — 整理作品集：三个主力项目收拢
- ☐ **[Day 69](Day-69-docs-and-storytelling.md)** — 文档与表达：README + 架构图 + 把生产实力讲清楚
- ☐ 🎯 **[Day 70](Day-70-retrospective-and-roadmap.md)** — 总复盘 + 规划：选定深挖方向，定下一步

---

## 📊 进度总览

| Phase | 内容 | 天数 | 累计 |
|---|---|:---:|:---:|
| 0 | 打地基 | Day 1–5 | 5 |
| 1 | 单 Agent + 工具 | Day 6–15 | 15 |
| 2 | RAG | Day 16–25 | 25 |
| 3 | 多 Agent 编排 | Day 26–45 | 45 |
| 4 | 生产化（Eval / 监控 / 安全） | Day 46–60 | 60 |
| 5 | 双栈架构 + 作品集 | Day 61–70 | 70 |

> 主力难点在 **Phase 3–4（Day 26–60）**，那 35 天别赶进度——它们才是和"只会写 demo 的人"拉开差距的地方。

## 🔁 节奏与巩固

- **缓冲日**：每个 Phase 末（Day 15 / 25 / 45 / 60 之后）留 0.5~1 天，补跑没跑通的代码、对齐依赖版本。
- **每周回顾三件事**：① 不看笔记复述本周 3 个核心概念；② 重跑本周一个 demo，故意改坏一处看报错；③ 把本周的 Java 类比补进持续累积的对照表。

---

## 🧩 扩展 / 进阶主题（70 天之外）

> 均为增量加餐，不占 Day 1–70 编号；按需深挖，非主线。

- ☐ **[EX-01](EX-01-context-engineering.md)** — 上下文工程（P0）：预算分配、编排排序、压缩与子 agent 隔离
- ☐ **[EX-02](EX-02-long-term-memory.md)** — 长期 / 跨会话记忆（P0）：记忆分层、写入 / 检索 / 遗忘、mem0 / Letta
- ☐ **[EX-03](EX-03-agent-system-design.md)** — Agent 系统设计面试专项（P0）：可复用 checklist + 45 分钟白板题
- 未成文主题（见[每日计划](../AI-Agent-每日学习计划.md)末尾）：RAG 进阶（P1，详见 [Day 23](Day-23-advanced-retrieval.md)）· computer-use / 浏览器 Agent（P2）· 多模态 Vision（P2）· 实时语音（P2）

---

> 📎 相关：[每日学习计划](../AI-Agent-每日学习计划.md) · [学习路线（择业视角）](../AI-Agent-学习路线.md) · [Python + 大模型主课程](../README.md)
