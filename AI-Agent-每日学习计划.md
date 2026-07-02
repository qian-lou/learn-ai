# AI Agent 开发 · 每日学习计划（70 天）

> **节奏假设**：每天约 2 小时、每周 5 个学习日（周末留作缓冲/复习）。
> **总量**：**70 个学习日 ≈ 14 周 ≈ 约 3.5~4 个月日历时间**（含周末缓冲）。
> **如何缩放**：
> - 每天只有 1 小时 → 日历时间翻倍（每个"学习日"拆成两天）。
> - 每天能投 3~4 小时 → 可压到 ~8~9 周。
> - 你工程底子好，Phase 0 觉得太简单可直接跳到 Day 6。
>
> **用法**：每天做完打个勾 ✅。每个 Phase 末尾都有一个"阶段项目",务必产出能跑的东西。
> **方向 A/B 分叉（轻量提示）**：方向 B（纯 AI 工程）作品集以 Python 为主，遇到 ☕ Java 对照日可"浏览"；方向 A（企业内加 Agent）则认真做每个 Java 对照日。首个里程碑（Day 15）后有一个"初步选向"检查点，随时可改。

---

## 📚 开课前置（Day 0 自检）

> 半天内自检，缺哪补哪；已具备的直接跳过。

**Python / 环境桥（缺则回查主课程对应章）**
| 你应先会 | 不熟就回查 |
|---|---|
| `uv` / venv 建虚拟环境（2026 事实标准：`uv` + `pyenv`） | 主课程「环境搭建」章 |
| `async` / `await`、typing 类型标注 | 主课程「高级特性」章 |
| Pydantic 定义模型 + 校验 | 主课程「LLM 应用」章 |

**30 分钟环境自检清单**：Python 3.11+、`uv` 可用、能建虚拟环境并装包、拿到 1 个厂商 API key 并跑通首个调用、`.env` 管理密钥不进 Git。

**成本护栏（1 分钟，开课强制，不必等 Day 51）**：在厂商控制台设置每月 spend limit；学习期默认用小模型，只在需要时切大模型。

---

## 🟢 Week 1 · Phase 0：打地基（Day 1–5）
🧰 **本阶段工具箱**：厂商 SDK（`openai` / `anthropic`）、`pydantic`；学习期锁定大版本，跑不通先对版本号；官方文档/changelog 为准。

- **Day 1 — 环境 + 第一次调用**：注册 API key，装 SDK，跑通第一个 LLM 调用。理解 token、context window、计费。
- **Day 2 — 模型参数**：temperature / top_p / max_tokens / stop。实现流式输出。同一 prompt 调不同参数对比结果。
- **Day 3 — Prompt 基础**：system/user/assistant 三种角色、few-shot、思维链(CoT)。反复改写 prompt 观察差异。
- **Day 4 — 结构化输出**：用 native structured outputs（JSON Schema strict 模式 / constrained decoding）在解码层强约束保证合法 JSON，再用 Pydantic 校验。心智统一：工具调用本质也是结构化输出。理解企业为什么必须要强类型输出。
- **Day 5 — 复盘 + 小项目**：合成一个「流式对话 CLI」(多轮 + 能识别结构化指令)。写下你自己的 Agent 心智模型一句话定义。

---

## 🟢 Week 2 · Phase 1：单 Agent + 工具（上）（Day 6–10）
🧰 **本阶段工具箱**：OpenAI Agents SDK / Claude Agent SDK、`pydantic`（Java 侧 Spring AI）；学习期锁定大版本，跑不通先对版本号；以官方文档/changelog 为准。

- **Day 6 — Tool Calling 原理**：模型如何决定调用函数，function schema 怎么定义，调用返回怎么回填。
- **Day 7 — 上手厂商 SDK**：用 OpenAI Agents SDK 或 Claude Agent SDK 定义第一个 tool，跑通一次完整工具调用。
- **Day 8 — 多工具 Agent**：实现 3~4 个工具（计算 / 查外部 API / 读文件 / 查数据库），观察模型如何选工具。
- **Day 9 — ReAct 循环**：理解「思考→行动→观察→再思考」。让 agent 连续调多个工具完成一个任务。
- **Day 10 — 错误处理**：工具抛异常、参数非法、超时怎么办。让 agent 优雅降级而不是崩。

## 🟢 Week 3 · Phase 1：单 Agent + 工具（下）（Day 11–15）

- **Day 11 — 记忆与会话状态**：短期记忆、对话历史管理、上下文超长时的截断策略。
  - ↳ 2026 补充：区分短期记忆 vs 长期/跨会话记忆——截断 ≠ 记忆架构，长期记忆（mem0 / Letta）见文末「扩展主题」`Day EX-MEM`；上下文的预算/编排/压缩已独立为「上下文工程」，见 `Day EX-CTX`。
- **Day 12 — 工具 + 结构化输出结合**：把工具结果转成强类型对象返回，端到端可校验。
- **Day 13 — ☕ Java 对照日 1**：Spring Boot + Spring AI，把工具映射到带 `@Tool`/`@ToolParam` 的 Bean，做同一个工具 Agent。
- **Day 14 — ☕ Java 对照日 2**：体会 Spring AI 的 Advisor 模式 vs Python SDK 的差异，记下两种生态各自的优劣。
- **Day 15 — 🎯 阶段项目**：完成「数据查询 Agent」——查真实数据、返回结构化结果、带错误处理。复盘。
  - ↳ 2026 补充（可选升级）：接一个真实带副作用的外部 API（GitHub / Slack / 日历 / 工单），做"读取→决策→写回/触发"闭环，处理真实鉴权、限流、脏数据——这是"跟过教程 vs 做过真东西"的区分点。
  - ↳ 里程碑自评：按文末三档标尺（跑通 / 能讲清取舍 / 能扛住追问）勾选，并产出一张"量化亮点卡"（见「五个里程碑」节）。
  - ↳ 初步选向检查点：此处可先定方向 A 还是 B（随时可改），据此决定后续 4 个 Java 对照日"精做 or 浏览"。

---

## 🟢 Week 4 · Phase 2：RAG（上）（Day 16–20）
🧰 **本阶段工具箱**：`pgvector`（+ Postgres）、嵌入模型 SDK、切分/rerank 工具（Java 侧 Spring AI ETL / LangChain4j）；学习期锁定大版本，跑不通先对版本号；以官方文档为准。

- **Day 16 — Embedding 原理**：文本→向量、余弦相似度。亲手算一次两段文本的相似度。
- **Day 17 — 向量库入门**：pgvector 安装配置（复用 Postgres，最易上手），存取向量。
- **Day 18 — 文档切分(chunking)**：固定大小 vs 语义切分、overlap、chunk size 如何影响检索质量。
- **Day 19 — 嵌入 pipeline**：一批文档 读入→切分→嵌入→入库，跑通 ETL。
- **Day 20 — 检索**：query 嵌入→相似度检索→取 top-k。调 k 值和阈值看效果变化。

## 🟢 Week 5 · Phase 2：RAG（下）（Day 21–25）

- **Day 21 — 完整 RAG 链路**：检索结果拼进 prompt→生成答案。跑出第一个能用的 RAG。
- **Day 22 — 引用与溯源**：让答案带出处，明确"不知道就说不知道",压低幻觉。
- **Day 23 — 进阶检索**：metadata 过滤、混合检索(关键词+语义)、重排序(rerank)。
  - ↳ 2026 补充：contextual retrieval（chunk 入库前补全上下文，廉价高收益已成标配）；GraphRAG / 多跳关系检索（纯向量答不了跨文档多跳时用）；检索质量评测用 recall@k / MRR；心里要有"何时不该用 RAG"的判断。
  - ↳ 多模态延伸（扩展主题，见 `Day EX-MM`）：截图/含图表或扫描件 PDF 的视觉 RAG，2026 主流模型已原生多模态，纯文本 RAG 常不够用。
- **Day 24 — ☕ Java 对照日**：用 Spring AI ETL + QuestionAnswerAdvisor 或 LangChain4j EmbeddingStore 做同样的 RAG。
- **Day 25 — 🎯 阶段项目**：完成「文档问答系统(带引用)」。整理 RAG 常见坑清单。

---

## 🟢 Week 6 · Phase 3：编排与多 Agent（一）（Day 26–30）
🧰 **本阶段工具箱**：`langgraph`、`crewai`、MCP SDK；checkpointer 用 SQLite/Postgres；LangGraph 版本迭代快，学习期锁定大版本，跑不通先对版本号与文档。

- **Day 26 — LangGraph 入门**：状态机心智模型 —— node / edge / state schema。建第一个 graph。
  - ↳ 2026 补充（过渡台阶）：先用纯 SDK 手写一个 3 步以上带分支的 Agent，直到自己写崩，再引入 LangGraph——把"框架来解决什么痛"显式化，别把它当"又一个要背的 API"。
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
  ↳ **2026 补充**：Agent 可靠性语义——工具幂等键、重放安全、副作用 exactly-once、失败补偿/saga 回滚（"重试一个已扣款的工具"怎么办）。这是你 Java 后端的差异化强项。
  - ↳ 2026 补充（Java 后端强项落点）：可靠性语义——工具幂等键设计、重放安全、副作用 exactly-once、失败补偿/saga 回滚。想清"重试一个已发邮件/已扣款的工具会怎样""重放（配合 Day 29 checkpointer 续跑）时如何不重复执行外部动作"。
- **Day 37 — MCP 协议**：理解 Model Context Protocol，接一个现成的 MCP server 当工具源。
- **Day 38 — A2A 协议**：Agent 间通信标准，搞清它解决什么问题、何时用。
- **Day 39 — 复杂工具编排**：让 agent 组合多个工具 + 子 agent 完成一个复杂任务。
- **Day 40 — 性能**：并行工具调用、流式输出中间结果。

## 🟢 Week 9 · Phase 3：阶段项目（Day 41–45）

- **Day 41 — 项目设计**：设计「自动化研究 Agent」的架构（输入→检索→分析→产出报告）。
  - ↳ 2026 补充：planning / reflection 节点用推理模型（thinking / o 系列 / extended reasoning），执行类节点用常规模型——"能力/推理"与"成本"是正交两轴（详见 Day 52）。
- **Day 42 — 实现①**：搭主干 graph + 状态 schema。
- **Day 43 — 实现②**：多 agent 协作 + 工具接入。
- **Day 44 — 实现③**：加断点续跑 + HITL + 错误恢复。
- **Day 45 — 🎯 完成 + 复盘**：这是简历主力项目之一,务必能完整跑通并讲清楚。
  - ↳ 2026 补充：复盘讲法点名"长时程可靠性 + 长期记忆"作为亮点（长期记忆见 `Day EX-MEM`）；按三档标尺自评并产出"量化亮点卡"（见「五个里程碑」节）。

---

## 🟢 Week 10 · Phase 4：生产化 —— 可观测 + 评估（Day 46–50）
🧰 **本阶段工具箱**：Langfuse / LangSmith / OpenTelemetry、eval 框架、`fastapi` + Docker、guardrails 工具；学习期锁定大版本，跑不通先对版本号；以官方文档为准。

- **Day 46 — 可观测性概念**：为什么 Agent 必须 trace、不 trace 会死在什么坑上。
- **Day 47 — 接入 tracing**：LangSmith / Langfuse / OpenTelemetry 任选一个接上。
- **Day 48 — 看 trace 调试**：逐步看 token、延迟、成本、每次工具调用，定位一个真实 bug。
- **Day 49 — 评估(Eval)入门**：怎么量化 agent 质量，构建一个测试集。
  ↳ **2026 补充**：eval 数据集三来源（人工种子 + 生产 trace 回流 + 合成扩充）、标注 rubric、LLM-as-judge 与人工的一致性校准。共识："eval 数据集就是护城河"。
  - ↳ 2026 补充（数据集与判分）：eval 数据集三来源——人工种子 + 生产 trace 回流 + 合成扩充；设计标注 rubric；LLM-as-judge 校准（与人工标注对齐、pairwise 优于绝对打分）；数据集要版本化。共识："eval 数据集就是护城河"。
  - ↳ benchmark 认知（半天）：先认识主流基准再自建——τ-bench / τ²-bench（工具+对话策略，pass^k）、SWE-bench Verified（代码 agent）、GAIA（通用助手）、WebArena / OSWorld（web/computer use），搞清各测什么、指标形态。
- **Day 50 — 写 eval**：跑准确率 / 幻觉率，建立回归测试,改一处不再担心崩别处。
  ↳ **2026 补充**：不止看最终答案（outcome），还要看过程——轨迹级/工具级评估：工具选择准确率、参数正确性、调用顺序、无效/冗余调用、`pass^k` 稳定性。trace（Day 48）即轨迹数据源。
  - ↳ 2026 补充（轨迹级/工具级评估）：除结果评估外加过程评估——工具选择准确率、参数正确性、调用顺序、无效/冗余调用率、死循环检测；`pass^k` 看稳定性（同一任务多跑结果飘）。trace（Day 48）即轨迹数据源。衡量"一次 agent 运行是否健康"，而非只看最终答案。
  - ↳ 非确定性测试进 CI：用语义断言（非精确匹配）+ 容差阈值做发布门禁，处理 flaky，抽样与成本权衡（工程收口详见 Day 59）。

## 🟢 Week 11 · Phase 4：成本 + 安全（Day 51–55）

- **Day 51 — 成本优化**：聚焦成本/缓存侧——prompt/上下文缓存、token 管理、控制上下文膨胀。（上下文的整体编排/预算/压缩属"上下文工程"，见 `Day EX-CTX`，此处不重复。）
- **Day 52 — 智能路由**：按"能力/推理 × 成本"两轴选型，而非成本单轴。简单任务用小模型、难任务用大模型。
  - ↳ 2026 补充（推理模型当大脑）：推理模型（thinking / o 系列 / extended reasoning）与常规模型是正交轴；何时开 extended thinking / 调 reasoning effort；推理模型 prompt 写法不同（给目标不给步骤、少喂 CoT）；工程决策："什么任务值得为推理付延迟/token"。
- **Day 53 — 安全①**：Prompt 注入的原理与防御。
  ↳ **2026 补充**：区分直接 vs **间接注入**——恶意指令藏在被检索的网页/文档/邮件/工具返回里；不可信内容边界标记、数据/指令分离；带 RAG+联网工具的 agent 暴露面最大。
  - ↳ 2026 补充：区分直接 vs 间接注入——恶意指令藏在被检索网页/文档/邮件/工具返回里；用不可信内容边界标记/隔离，坚持"数据与指令分离"；带 RAG + 联网工具的 agent 暴露面最大。
- **Day 54 — 安全②**：工具权限边界、输出校验、guardrails、敏感数据处理。
  - ↳ 2026 补充（执行沙箱）：容器 / 微VM（gVisor 类）隔离、网络出口白名单、文件系统与资源配额、高危动作人类审批门控——对应 OWASP LLM"过度代理(Excessive Agency)"。
  - ↳ 多租户与数据隔离在 Day 62（Java 鉴权视角）落地：RAG 检索强制 tenant 过滤 + per-tenant 配额。
- **Day 55 — OWASP for LLM**：过一遍清单,逐条自查自己的 agent。

## 🟢 Week 12 · Phase 4：部署 + 阶段项目（Day 56–60）

- **Day 56 — 部署**：把 agent 包成服务（FastAPI），容器化。
  - ↳ 2026 补充（上线策略）：灰度/金丝雀 + 影子(shadow)评估、prompt/模型版本化与回滚；"离线 eval 通过 ≠ 线上安全"，把 eval 当发布门禁（回归接入见 Day 59）。
- **Day 57 — 生产关注点**：限流、超时、并发、降级、健康检查。
  - ↳ 2026 补充（运营纪律）：定义延迟与成功率 SLO；per-request / per-tenant 的 token/成本上限与熔断；预算告警 + 超预算自动降级（区别于 Day 51-52 的"如何更省"）。
- **Day 58 — 阶段项目①**：给前面的 agent 加上完整监控。
  - ↳ 2026 补充：把 Day 57 的 SLO / 成本护栏做成看板 + 告警（延迟、成功率、per-tenant token/成本、预算触顶）。
- **Day 59 — 阶段项目②**：加上 eval + 安全防护。
  ↳ **2026 补充**：搭**数据飞轮**——生产 trace 挑失败样本 → 回灌 eval 集 → 迭代 prompt/工具/路由 → 灰度回归 → 再上线，串起 trace(48)/eval(49-50)/回归(59)。
  - ↳ 2026 补充（数据飞轮 + CI 门禁）：串起零件——生产 trace(48) → 挑失败样本回灌 eval 集(49-50) → 迭代 prompt/工具/路由 → 灰度回归 → 再上线；把 eval 接入 CI 做发布门禁（阈值/容差/flaky 处理）。
- **Day 60 — 🎯 完成 + 复盘**：这是"能上线"的证明,面试里最值钱的部分。里程碑升级为「生产级 Agent + **可复用 eval harness（能对任意 agent 跑回归）**」。
  - ↳ 2026 补充（on-call 演练）：注入一个故障（幻觉暴涨 / 工具级联失败 / 注入得手 / 供应商限流 / 成本失控）→ 从 trace 定位 → 止血 → 回滚/降级 → 写 postmortem。
  - ↳ 用一个公开 benchmark 子集（见 Day 49）给自己 agent 打分，复盘时讲清数据飞轮闭环（见 Day 59）。
  - ↳ 按三档标尺自评并产出"量化亮点卡"（见「五个里程碑」节）。

---

## 🟢 Week 13 · Phase 5：双栈架构（上）（Day 61–65）
🧰 **本阶段工具箱**：Spring Boot（Java 服务层）、`langgraph`（Python 编排层）、REST / MCP 打通、跨端 tracing；两端各自锁定大版本，跑不通先对版本号与协议契约。

- **Day 61 — 架构设计**：Python 编排层 + Java 服务层,定清楚两边接口(REST / MCP)。
- **Day 62 — Java 服务层**：Spring Boot 提供业务服务 / 数据 / 鉴权 / 可观测性。
  - ↳ 2026 补充（多租户与数据隔离）：租户隔离模型；RAG 检索强制 tenant 过滤（元数据/命名空间，防"检索到别租户文档"）；per-tenant 速率/成本配额；行级权限传导到检索——与 Java 鉴权层天然衔接。
- **Day 63 — Python Agent 层**：LangGraph 编排,通过 API/MCP 调用 Java 服务。
- **Day 64 — 端到端打通**：让混合系统完整跑通一个真实场景。
- **Day 65 — 可观测性贯通两端**：trace 能从 Python 一路追到 Java。

## 🟢 Week 14 · Phase 5 收尾 + 作品集（Day 66–70）

- **Day 66 — 完善①**：处理混合系统的边界情况与失败路径。
- **Day 67 — 完善②**：压测、加缓存、优化延迟。
- **Day 68 — 整理作品集**：把三个主力项目(RAG问答 / 研究Agent / 混合系统)收拢。
- **Day 69 — 文档与表达**：写 README + 架构图,练习把每个项目的 eval / 可观测 / 安全讲清楚。
  - ↳ 2026 补充：为每个里程碑产出一张"量化亮点卡"（前后对比指标 + 一句话讲法）；练一道 Agent 系统设计现场题（客服 / 代码 / 深度研究 agent 任选），45 分钟白板讲清 + 权衡（checklist 见文末 `Day EX-SD`）。
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

## 🔁 节奏与巩固

- **缓冲/追赶日**：每个 Phase 末尾（Day 15 / 25 / 45 / 60 之后）显式留 0.5~1 个缓冲日，用于补跑没跑通的代码 + 版本对齐——把"周末缓冲"从假设升级为固定动作。
- **每周回顾协议**（周末固定 3 件事）：① 不看笔记，复述本周 3 个核心概念；② 重跑本周一个 demo，故意改坏一处看报错；③ 把本周的 Java 类比补进一张持续累积的对照表。

---

## ✅ 五个里程碑（每个都是一个能讲的作品）

> **三档自评标尺**（每个里程碑都按此勾选）：
> - 🟩 **跑通**：demo 能端到端跑，不崩。
> - 🟨 **能讲清取舍**：能说明为什么这么设计、有哪些替代方案、各自代价。
> - 🟥 **能扛住追问**：面试官追问失败路径 / 成本 / 安全 / 规模时能接住。
>
> **量化亮点卡模板**（每个里程碑产出一张）：前后对比指标 + 一句话讲法。
> 示例：`eval 准确率 71%→89%`、`p95 延迟 8s→2.3s`、`路由降本 60%`、`注入用例 20/20 拦截`。

1. **Day 15** — 数据查询 Agent（工具调用 + 结构化输出）
2. **Day 25** — 文档问答系统（完整 RAG，带引用）⭐ 面试高频
3. **Day 45** — 自动化研究 Agent（多 Agent + 状态）⭐ 主力项目
4. **Day 60** — 生产级 Agent（监控 + Eval + 安全 + **可复用 eval harness**）⭐⭐ 最值钱
5. **Day 70** — Java + Python 混合架构系统 ⭐⭐ 你的差异化王牌

---

## 🧩 扩展 / 进阶主题（70 天之外）

> 均为增量加餐，**不占 Day 1–70 编号**；按需深挖，非主线。优先级：**P0 应补** / **P1 建议** / **P2 可选**。

- **`Day EX-CTX` 上下文工程**（P0）：上下文预算分配；工具结果/检索片段/历史/系统指令的编排排序（`lost in the middle`）；上下文压缩/摘要；子 agent 上下文隔离与中间步骤压缩；`context rot`。心智：2026 重心从 prompt engineering → context engineering。产出：画出"一次 agent 调用的上下文预算表"。
- **`Day EX-MEM` 长期 / 跨会话记忆**（P0）：短期 vs 长期记忆分层；事实抽取/用户画像；语义 vs 情节记忆；写入/检索/遗忘策略；与 RAG 的区别（记忆是"关于用户/会话的状态"，RAG 是"外部知识"）。2026 主流：mem0 / Letta（原 MemGPT）。
- **`Day EX-SD` Agent 系统设计面试专项**（P0）：可复用设计 checklist（任务边界 / 上下文与记忆 / 工具集与权限 / 失败与回退 / 评估与护栏 / 成本）；高频命题（客服 / 代码 / 深度研究 agent）；45 分钟白板讲清 + 权衡。
- **`Day EX-RAG` RAG 进阶**（P1）：contextual retrieval、GraphRAG / 多跳检索、recall@k / MRR、"何时不该用 RAG"（详见 Day 23）。
- **`Day EX-CUA` Computer-use / 浏览器 Agent**（P2，可选加餐）：computer-use（Anthropic）/ operator 类 / `browser-use` / Playwright-MCP；视觉定位 + 动作循环 + 安全护栏 + 失败截图取证。与函数式工具调用是不同范式，招聘高频加分作品线。
- **`Day EX-MM` 多模态（Vision）Agent**（P2，可选加餐）：截图理解、图表/扫描件 PDF 解析、视觉 RAG；2026 主流模型原生多模态；是 computer-use 的基础。
- **`Day EX-VOICE` 实时语音 Agent**（P2，可选加餐，ROI 低于前几条）：Realtime API / speech-to-speech（STT→LLM→TTS 或原生语音模型）；打断处理、端到端延迟预算、语音工具调用。
