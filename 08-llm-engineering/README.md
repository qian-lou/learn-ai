# 阶段八：大模型部署与工程化

> **预估周期**：2-3 周
> **核心目标**：把"能跑通"的大模型变成"跑得起、跑得快、扛得住、可迭代"的生产系统——量化优化 + 服务部署 + MLOps 三段闭环

---

## 📋 模块大纲

### [01-model-optimization](./01-model-optimization/) — 模型优化

减小模型体积、提升推理速度的关键技术：量化（NF4/INT8/INT4、GPTQ/AWQ/HQQ、FP8/NVFP4、GGUF）、推理加速（KV Cache、FlashAttention、continuous batching、speculative decoding、PagedAttention）、知识蒸馏（软标签蒸馏 + 数据蒸馏）。这是把 A100 级模型压进消费级 GPU 的地基。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [quantization](./01-model-optimization/01-quantization.md) | 模型量化（线性量化原理 / NF4 / GPTQ / AWQ / FP8 / NVFP4 / GGUF IQ / torchao） |
| 02 | [inference-acceleration](./01-model-optimization/02-inference-acceleration.md) | 推理加速（KV Cache 显存估算 / FlashAttention / continuous batching / speculative decoding / PagedAttention） |
| 03 | [knowledge-distillation](./01-model-optimization/03-knowledge-distillation.md) | 知识蒸馏（Temperature 软标签 / KL 蒸馏损失 / 数据蒸馏 / on-policy 蒸馏） |

---

### [02-model-serving](./02-model-serving/) — 模型服务

将训练/优化好的模型部署为高吞吐、OpenAI 兼容、可容器化的 API 服务：vLLM V1 引擎（PagedAttention + continuous batching）、FastAPI 网关（SSE 流式 + 生产要点）、Docker 容器化（GPU 直通 + 多阶段构建 + 编排）。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [vllm-tgi](./02-model-serving/01-vllm-tgi.md) | vLLM / TGI 部署（`vllm serve` / prefix caching / TP·PP 多卡 / 引擎选型对比） |
| 02 | [api-service](./02-model-serving/02-api-service.md) | API 服务设计（FastAPI ≈ Spring Boot / Pydantic 校验 / SSE 流式 / 限流认证监控） |
| 03 | [docker-deployment](./02-model-serving/03-docker-deployment.md) | Docker 容器化部署（`--gpus all` / 多阶段构建 / compose GPU 声明 / Nginx 反代关 buffering） |

---

### [03-mlops](./03-mlops/) — MLOps 工程化

机器学习项目的工程化闭环：实验追踪（MLflow/W&B）、评估与监控（LLM-as-judge + Ragas + Prometheus）、CI/CD（GitHub Actions + DVC + 评估门控 + GPU runner）。让模型迭代可复现、退化可发现、交付可自动化。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [experiment-tracking](./03-mlops/01-experiment-tracking.md) | 实验追踪（`log_param` vs `log_metric` / Trainer 集成 / 远程 Tracking Server） |
| 02 | [evaluation-and-monitoring](./03-mlops/02-evaluation-and-monitoring.md) | 评估与监控（LLM-as-judge 偏差缓解 / Ragas / 防污染基准 / Histogram 监控 TTFT·ITL） |
| 03 | [cicd](./03-mlops/03-cicd.md) | ML 项目 CI/CD（GitHub Actions / DVC 数据版本 / MLflow Registry / self-hosted GPU runner） |

---

## 🎯 阶段学习要点

- **优化**：会用 `bitsandbytes` / `torchao` / `GPTQModel` 至少一种量化工具压模型，并亲手量出显存下降；手算一次 KV Cache 显存（GQA 用 `n_kv_heads`）。
- **部署**：用 `vllm serve` 起一个 OpenAI 兼容服务，前面套 FastAPI 网关（同步 + SSE 流式），再用多阶段 Dockerfile + compose 声明 GPU 打包上线。
- **工程化**：跑通 MLflow 实验追踪，写一个带偏差缓解的 pairwise LLM-as-judge，配 Prometheus `Histogram` 监控 TTFT/ITL，并搭一条含评估门控的 GitHub Actions 流水线。
- **贯通思维**：三个模块是一条流水线——量化/加速产出更廉价的模型，服务层把它变成 API，MLOps 保证这条链路可复现、可观测、可持续交付。
- **2026 视角**：低比特已进入 FP8/NVFP4 浮点量化时代；vLLM 全面切 V1 引擎；生成质量评测的事实标准是 LLM-as-judge + 防污染基准。

---

## 🔗 关联

- **上一阶段**：[阶段七 · 大模型应用实战](../07-llm-applications/README.md)（本阶段部署/工程化的正是上一阶段跑通的模型与应用，含 LoRA 微调产出）
- **配套实战课**：`agent-course/` 第 56~58、67 天从 Agent 视角落地本阶段——[Day 56 部署 FastAPI+Docker](../agent-course/Day-56-deploy-fastapi-docker.md)、[Day 57 生产关注点](../agent-course/Day-57-production-concerns.md)、[Day 58 完整监控](../agent-course/Day-58-capstone-monitoring.md)、[Day 67 压测/缓存/延迟](../agent-course/Day-67-load-cache-latency.md)
