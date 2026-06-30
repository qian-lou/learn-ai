# Day 68 · 整理作品集：三个主力项目收拢

> **今日目标**：把 RAG 问答、研究 Agent、Java+Python 混合系统三个主力项目，从"散落的代码"收拢成统一、能跑、能讲的作品集。
> **时长**：~2h ｜ **前置**：Day 25 / Day 45 / Day 65（三个里程碑项目已分别完成）
> **今日产出**：一个 `portfolio/` 顶层目录（三个项目结构统一、各自一键可跑、各带一段简历项目描述），外加一份《项目自检清单》逐项打勾。

## 1. 为什么 & 是什么（概念 + Java 类比）

代码会写 ≠ 作品集能用。招聘方看的是：**项目能不能一键跑起来、README 能不能让人 5 分钟看懂、亮点能不能用一句话讲清**。今天不写新功能，做的是"产品化收口"——把三个项目对齐成同一套骨架，去掉只有你自己电脑能跑的隐性依赖，给每个项目配一段**简历级描述**。这是把"学过"变成"能展示"的关键一步。

给 Java 工程师的类比，全是你熟的工程规范：

| 作品集要素 | Java/工程世界类比 | 要点 |
|---|---|---|
| 统一项目结构 | 多模块 Maven 工程 / 标准目录布局 | 三个项目同一套骨架，读者零学习成本 |
| 一键启动 | `mvn spring-boot:run` / `docker compose up` | `make run` 或 `docker compose up` 必须开箱即跑 |
| 依赖锁定 | `pom.xml` 锁版本 + Dockerfile | `requirements.txt`/`uv.lock` 锁死，杜绝"我这能跑" |
| `.env.example` | `application-example.yml` | 给出占位配置，密钥绝不入库 |
| 简历项目描述 | 述职/晋升材料里的项目段 | STAR + 量化指标，一句话能说清亮点 |

**核心心智：作品集是给"别人"看的，不是给你自己看的。** 默认读者只有 5 分钟、且电脑上什么都没装。凡是"需要你口头解释才能跑/才能懂"的地方，都是要补的债。

## 2. 跟着做（Hands-on）

**Step 1 — 立统一骨架**。三个项目对齐成同一结构，读者扫一眼就懂：

```text
portfolio/
├── README.md                  # 作品集总览：三个项目一句话 + 跳转
├── 01-rag-qa/                 # Day 25：带引用的文档问答
│   ├── README.md              # 用途/架构图/一键跑/eval 截图/亮点
│   ├── docker-compose.yml     # 一键起：app + pgvector
│   ├── .env.example           # 占位密钥，绝不放真 key
│   ├── requirements.txt       # 锁版本
│   ├── src/ …                 # 源码
│   └── eval/ …                # 测试集 + 评测脚本（Day 49~50 产物）
├── 02-research-agent/         # Day 45：多 Agent + 状态持久化 + HITL
│   └── （同上骨架）
└── 03-hybrid-system/          # Day 65：Python 编排 + Java 服务
    ├── README.md
    ├── docker-compose.yml     # 一键起：python-orchestrator + java-service + db
    ├── orchestrator/          # Python / LangGraph
    └── java-service/          # Spring Boot
```

**Step 2 — 一键可跑（这是硬门槛）**。每个项目至少给一种零配置启动方式：

```bash
# 进任一项目目录，复制占位配置后一键起 / copy env, then one-command up
cp .env.example .env            # 填入你自己的 key（已 .gitignore）
docker compose up --build       # 起全套依赖：app + db/向量库/Java 服务
```

> 验收口径：**在一台干净机器上**（或新建虚拟环境/新 clone）执行上面两行就能跑起来。凡是还要你手动建表、手动塞数据、手动改路径的，今天补成脚本（如 `make seed`）。

**Step 3 — 给每个项目写「简历项目描述」**。用下面这个模板，**一个项目一段**，亮点量化：

```text
【项目名】Java+Python 混合 AI Agent 平台（个人项目，2026）
【一句话】Python(LangGraph) 编排层 + Java(Spring Boot) 服务层的混合架构，
         把 LLM 不确定性隔离在编排层、把业务/鉴权/数据稳定在 Java 层。
【背景 S】单语言 Agent 难以复用既有 Java 业务资产，且生产化能力缺失。
【任务 T】设计可观测、可降级、可压测的双栈 Agent 系统并端到端打通。
【行动 A】① REST/MCP 定义两层接口；② 调用边界加超时/断路/降级 + 幂等；
         ③ 语义缓存 + Java @Cacheable 优化延迟；④ trace 从 Python 贯通到 Java。
【结果 R】p95 延迟 ↓<X>%、单请求成本 ↓<Y>%；Java 宕机时 Agent 优雅降级不崩；
         eval 回归覆盖 N 条用例，幻觉率 <Z>%。
【技术栈】Python / LangGraph / Spring Boot / pgvector / Langfuse / Docker
```

> 三段都套这个骨架（RAG 问答突出"引用溯源 + 幻觉率"，研究 Agent 突出"多 Agent 协作 + 断点续跑 + HITL"，混合系统突出"双栈架构 + 生产化"）。**`<X>/<Y>/<Z>` 用 Day 67 的真实数字填，别编。**

**Step 4 — 清理与脱敏**：跑一遍 `git status`，确认没有 `.env`/真实 key/大数据文件入库；删掉调试脚本、注释掉的死代码、TODO；统一 README 标题层级与命令格式。

## 3. 今日任务

1. **建 `portfolio/` 骨架**：三个项目对齐到同一目录结构，各自就位。
2. **打通一键跑**：每个项目都能用 `docker compose up`（或 `make run`）在干净环境零配置启动；不满足的补脚本。
3. **锁依赖 + 脱敏**：`requirements.txt`/`uv.lock` 锁版本，`.env.example` 就位，确认无密钥/大文件入库。
4. **写三段简历描述**：用上面模板，每个项目一段，量化指标取自真实数据。
5. **过自检清单**（见下）：逐项打勾，缺一补一。

**验收标准**：①三个项目结构一致；②每个项目在干净环境一键可跑（你亲自验证过）；③依赖锁定、密钥未入库；④三段简历描述齐全且含真实量化指标。

## 4. 自测清单

- [ ] 三个项目能在**干净环境/新 clone**上一键跑起来（我真的试过，不是"应该能"）。
- [ ] 每个项目的依赖都锁了版本，杜绝"我这能跑你那不行"。
- [ ] 仓库里没有真实密钥、`.env`、超大数据文件。
- [ ] 每个项目有一段 STAR + 量化的简历描述，亮点一句话能讲清。
- [ ] 三个项目各有差异化亮点（引用溯源 / 多 Agent + 续跑 / 双栈生产化），不是三个"差不多的 demo"。

## 5. 延伸 & 关联

- 容器化与一键部署（作品集"开箱即跑"的底座）：[../08-llm-engineering/02-model-serving/03-docker-deployment.md](../08-llm-engineering/02-model-serving/03-docker-deployment.md)
- RAG 实战（01-rag-qa 项目对应章节）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
- LangChain 完整应用（项目工程化参考）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
- CI/CD（让"一键跑"延伸为"一键测/一键部署"）：[../08-llm-engineering/03-mlops/03-cicd.md](../08-llm-engineering/03-mlops/03-cicd.md)
- 总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)

> 衔接 Day 69：骨架立好了，明天给每个项目配 **README + 架构图**，并把 eval / 可观测 / 安全这三块"生产实力"讲到位。
