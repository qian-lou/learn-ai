# 03-mlops — MLOps 工程化

> **所属阶段**：阶段八 · 大模型部署与工程化
> **学习目标**：掌握 ML/LLM 项目的工程化闭环——实验可复现、上线可评估、退化可监控、交付可自动化
> **预估时长**：4-5 天（含跑通 MLflow/W&B + LLM-as-judge 评测 + GitHub Actions 流水线）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [experiment-tracking](./01-experiment-tracking.md) | 实验追踪 | W&B / MLflow / TensorBoard、`log_param` vs `log_metric`、HuggingFace Trainer 一行集成、远程 Tracking Server + PostgreSQL |
| 02 | [evaluation-and-monitoring](./02-evaluation-and-monitoring.md) | 评估与监控 | ROUGE/BLEU 的局限、LLM-as-judge（pointwise/pairwise + 偏差缓解）、Ragas、防污染基准、Prometheus 监控 TTFT/ITL、幻觉检测 |
| 03 | [cicd](./03-cicd.md) | ML 项目 CI/CD | GitHub Actions ML 流水线、DVC 数据版本、MLflow Model Registry、GPU Self-hosted Runner、评估门控、Secrets 管理 |

---

## 🔑 知识点详解

### 01 · 实验追踪（Experiment Tracking）

- **核心概念**：把每次训练的**超参数 + 指标 + 模型版本**结构化记录下来，让 50 次 LoRA 实验可对比、可复现、可协作——没有它，调参就是"薛定谔的最优配置"。
- **关键公式/API**：核心区分两个 API——`log_param()`（训练前设定、过程中不变，如 lr/rank）vs `log_metric(name, value, step=...)`（随 step 变化，如 loss/accuracy，带 step 才能画时序曲线）。HuggingFace 一行集成：`TrainingArguments(report_to="wandb", run_name=...)`。
- **易错点**：① `log_param` 和 `log_metric` 别用反——用 `log_param` 记 loss 会丢失时序；② 团队/多节点场景要起**远程 Tracking Server**（`mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...`），本地文件后端无法共享；③ 只记指标不记完整 config，实验就不可复现。
- **Java 视角**：≈ ELK 日志系统 + 配置中心——每次运行的参数和结果都落库，支持聚合对比与看板。
- **前置**：无强依赖；与阶段五微调（LoRA 超参搜索）直接配套。

### 02 · 评估与监控（Evaluation & Monitoring）

- **核心概念**：LLM 上线后"benchmark 好 ≠ 生产好"；评估要从**字面匹配转向语义/LLM 打分**，监控要覆盖质量、安全、性能、成本四维。
- **关键公式/API**：**LLM-as-judge** 是 2026 事实标准，分 pointwise（单条 1-10 打分）与 pairwise（A/B 对比，更稳）。RAG 场景用 Ragas 四指标：`faithfulness`（忠实度）、`answer_relevancy`、`context_precision`、`context_recall`。线上监控关键类型是 Prometheus 的 **`Histogram`**（可算 P95/P99 分位），观测 `llm_ttft_seconds`（首 token 延迟）和 `llm_inter_token_latency_seconds`（ITL）。
- **易错点**：① ROUGE/BLEU 只做 N-gram 硬匹配，同义改写会被判低分，别拿它当 LLM 生成质量的唯一标准；② LLM-as-judge 有 **position bias**（偏靠前）、**verbosity bias**（偏长答案）、**self-preference**（偏同源模型），需交换 A/B 顺序两跑取一致、rubric 明确"简洁不扣分"、用异源裁判缓解；③ 延迟指标用 `Counter`/`Gauge` 算不出分位数，必须用 `Histogram`。
- **Java 视角**：≈ APM（应用性能监控）——Prometheus + Grafana 盯 QPS/延迟/错误率，再叠加 LLM 特有的幻觉率、token 成本等质量指标。
- **前置**：模块 02 的 API 服务（监控挂在服务端点上）；与 `agent-course` 的 eval/可观测性直接呼应。

### 03 · ML 项目 CI/CD

- **核心概念**：ML 的 CI/CD 比传统软件多一层——要同时管住**代码 + 数据 + 模型**三重版本，并在流水线里加入训练、评估门控与上线后监控。
- **关键公式/API**：`GitHub Actions` 定义流水线（`on.push` 触发、`jobs.needs` 串联、`runs-on: [self-hosted, gpu]` 跑训练）；`DVC` 做数据版本（`dvc add` 生成 `.dvc` 小指针提交 Git，实体存 S3/GCS，`git checkout + dvc checkout` 版本切换）；`MLflow Model Registry` 做模型注册。Secrets 用 `${{ secrets.NAME }}` 引用。
- **易错点**：① 密钥**绝不硬编码**进 yaml，放 repo Secrets 用 `${{ secrets.* }}`；② 重型 GPU 训练不要跑在 GitHub 免费 CPU runner 上，也不要每次提交都触发——用 tag/`workflow_dispatch` 触发 + `self-hosted` GPU runner 分离；③ 没有**评估门控**（评估不过不部署）等于把风险直接推到生产。
- **Java 视角**：Java 工程师熟悉的 CI/CD（代码→测试→构建→部署）之上，插入"数据验证→训练→评估→模型注册→上线后监控"的额外环节，本质是同一套流水线思想。
- **前置**：01（实验/模型注册）+ 02（评估门控与监控）+ 模块 02 的 Docker（部署产物）。

---

## 🎯 学习要点

- 跑通一次 MLflow 全流程：`set_experiment` → `start_run` → `log_param`/`log_metric(step=...)` → `set_tag`，并在 MLflow UI 里对比多次 run 的 loss 曲线。
- 理解并演示 **`log_param` vs `log_metric`** 的本质区别（静态超参 vs 带 step 的时序指标），能搭一个带 PostgreSQL 后端的远程 Tracking Server。
- 亲手写一个 **pairwise LLM-as-judge**，并实现"交换 A/B 顺序两跑取一致判平局"来缓解 position bias；再用 Ragas 对一个 RAG 系统跑 `faithfulness/answer_relevancy` 自动评测。
- 说清楚为什么 ROUGE/BLEU 不再适合评 LLM 生成质量，以及现代评测的两条主线：**LLM-as-judge + 防污染基准**（LiveBench / MMLU-Pro / GPQA / IFEval / Chatbot Arena）。
- 用 Prometheus `Histogram` 监控 **TTFT 与 ITL** 并配 P99 告警；掌握四类监控告警规则（延迟/错误率/幻觉率/GPU 利用率）。
- 写一条完整 GitHub Actions 流水线：Lint（ruff）→ 单测（pytest）→ 多阶段 Docker 构建推送；理解 GPU 训练用 tag/`workflow_dispatch` + self-hosted runner 分离，并用 DVC 管数据版本、加评估门控。

---

## 🔗 关联

- **上一模块**：[02-model-serving](../02-model-serving/) — 上线的服务正是本模块要追踪实验、评估监控与自动化交付的对象
- **下一模块**：无（本模块为阶段八收尾）→ 回到 [阶段总览](../README.md) 进入下一阶段
- **阶段总览**：[阶段八 · 大模型部署与工程化](../README.md)
- **配套实战**：`agent-course/` [Day 46 可观测性概念](../../agent-course/Day-46-observability-concepts.md)、[Day 49 评估入门](../../agent-course/Day-49-eval-intro.md)、[Day 50 写 eval（幻觉率 + LLM-as-judge + 回归）](../../agent-course/Day-50-writing-evals.md)、[Day 58 阶段项目·给 Agent 服务加完整监控](../../agent-course/Day-58-capstone-monitoring.md)
