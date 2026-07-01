# 02-model-serving — 模型服务

> **所属阶段**：阶段八 · 大模型部署与工程化
> **学习目标**：把训练/优化好的大模型部署为高吞吐、OpenAI 兼容、可容器化的生产 API 服务
> **预估时长**：4-5 天（含跑通 vLLM serve + FastAPI 网关 + Docker 编排）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [vllm-tgi](./01-vllm-tgi.md) | vLLM / TGI 部署 | vLLM V1 引擎、PagedAttention、prefix caching、continuous batching、structured outputs / tool calling、TP/PP 多卡、与 TGI/Ollama/SGLang/TensorRT-LLM 的选型对比 |
| 02 | [api-service](./02-api-service.md) | API 服务设计 | FastAPI（Python 版 Spring Boot）、Pydantic 校验、SSE 流式输出、`def` vs `async def`、限流/认证/监控生产要点 |
| 03 | [docker-deployment](./03-docker-deployment.md) | Docker 容器化部署 | GPU 基础镜像、`--gpus all`、多阶段构建、docker compose GPU 声明、Nginx 反代（流式需关 buffering）、`.dockerignore` 排除权重 |

---

## 🔑 知识点详解

### 01 · vLLM / TGI 部署

- **核心概念**：vLLM/TGI 是大模型推理的**工业级引擎**——用 PagedAttention + continuous batching 把 GPU 吞吐拉满，并暴露 OpenAI 兼容 API 让客户端无缝切换。
- **关键公式/API**：**PagedAttention** 把 KV Cache 切成固定大小的 block（页），经 block table（页表）映射到非连续物理块，显存利用率从 ~30% 提到 90%+。启动命令记 `vllm serve <model> --host 0.0.0.0 --port 8000 --tensor-parallel-size N --enable-prefix-caching`（2026 首选 CLI，替代旧的 `python -m vllm.entrypoints.openai.api_server`）。
- **易错点**：① `--tensor-parallel-size` 必须能**整除注意力头数**，否则启动报错；② 别再用旧的 `api_server` 模块路径；③ prefix caching 在 V1 引擎默认开启，重复设置无害但要知道它已生效。
- **Java 视角**：vLLM ≈ 高性能应用服务器（Tomcat/Undertow）+ 内置连接池；PagedAttention ≈ 操作系统虚拟内存的分页调度；prefix caching ≈ 对相同请求前缀做结果缓存。
- **前置**：模块 01 的 KV Cache 与 PagedAttention 概念。

### 02 · API 服务设计

- **核心概念**：把推理引擎封装为 RESTful API 是生产落地的标准姿势；LLM 生成耗时长，**SSE 流式输出**（打字机效果）是刚需。
- **关键公式/API**：`StreamingResponse(gen, media_type="text/event-stream")` + 异步生成器 `yield f"data: {chunk}\n\n"`，以 `data: [DONE]\n\n` 收尾（SSE 协议约定）。请求体用 Pydantic `BaseModel` + `Field(..., ge=0, le=2)` 做类型/范围校验。
- **易错点**：① `async def` 里**禁止**放同步阻塞调用（会卡死事件循环），要用 `httpx`/异步库；纯 CPU 密集或同步 IO 应写成 `def`（FastAPI 自动丢线程池）；② SSE 每条消息必须 `data:` 前缀 + 双换行 `\n\n`，少一个换行客户端解析不到；③ 关键生产指标是 **TTFT（首 token 延迟）** 和 **tokens/s**，普通 HTTP 中间件测的是整请求耗时，需在流里挂钩子才能测 TTFT。
- **Java 视角**：FastAPI ≈ Python 版 Spring Boot——`@app.post`↔`@PostMapping`、`BaseModel`↔DTO、`Depends`↔`@Autowired`、`middleware`↔`Filter`、`uvicorn`↔Tomcat。
- **前置**：01（作为上游推理后端）；阶段内需理解异步事件循环。

### 03 · Docker 容器化部署

- **核心概念**：容器化保证**环境一致性**（开发能跑 = 生产能跑），AI 部署相对普通服务只是多了 GPU 直通与巨大的模型权重管理。
- **关键公式/API**：GPU 直通用 `docker run --gpus all`；compose 中用 `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`（Compose Spec 已不需要 `version` 字段）。多阶段构建：`FROM ... AS builder` 装依赖 → `FROM ... AS runner` 只拷产物。
- **易错点**：① **绝不把模型权重打进镜像**——用 volume 挂载 + `.dockerignore` 排除 `*.safetensors`/`*.bin`/`.cache/`，否则镜像十几 GB 且无法换模型；② 基础镜像用 `runtime`（~3GB）而非 `devel`（~8GB）；③ Nginx 反代流式接口必须 `proxy_buffering off`，否则打字机效果被缓冲吞掉，且要调大 `proxy_read_timeout`（LLM 生成慢）。
- **Java 视角**：Java 工程师对 Docker/compose/多阶段构建已很熟——差异只在 NVIDIA Container Toolkit 和"权重当数据挂载、不进镜像层"这条铁律。
- **前置**：02（容器里跑的正是 FastAPI/vLLM 服务）。

---

## 🎯 学习要点

- 用 `vllm serve` 一条命令部署 OpenAI 兼容服务，并用标准 `curl /v1/chat/completions` 与 OpenAI SDK（`base_url=".../v1"`）验证——理解底层自动启用 PagedAttention + continuous batching。
- 亲手对比 **vLLM vs HuggingFace 原生**推理吞吐（同模型/同采样、丢弃预热），直观感受 continuous batching 带来的数倍到一个数量级差距。
- 掌握**多卡部署**：`--tensor-parallel-size`（层内切分，须整除头数）+ 跨节点 `--pipeline-parallel-size`，并知道单节点优先 NVLink 降通信开销。
- 用 FastAPI 写一个含**同步 + SSE 流式**两个端点的网关，正确区分 `def` 与 `async def` 的使用场景，并挂中间件统计 TTFT 与 tokens/s。
- 会写**多阶段 Dockerfile** + `.dockerignore`，用 compose 声明 GPU、挂载模型 volume、配 Nginx 反代（关 buffering），并加 `HEALTHCHECK`。
- 建立横向选型直觉：吞吐通用首选 **vLLM**；多轮/Agent 前缀复用看 **SGLang(RadixAttention)**；NVIDIA 卡极致低延迟看 **TensorRT-LLM**；国产/量化模型开箱看 **LMDeploy**；个人开发用 **Ollama**。

---

## 🔗 关联

- **上一模块**：[01-model-optimization](../01-model-optimization/) — 量化/加速后的模型正是本模块要部署的对象
- **下一模块**：[03-mlops](../03-mlops/) — 服务上线后如何做实验追踪、评估监控与 CI/CD
- **阶段总览**：[阶段八 · 大模型部署与工程化](../README.md)
- **配套实战**：`agent-course/` [Day 56 部署 FastAPI + Docker](../../agent-course/Day-56-deploy-fastapi-docker.md)、[Day 57 生产关注点（限流/超时/并发/降级/健康检查）](../../agent-course/Day-57-production-concerns.md) — 从 Agent 服务视角落地本模块的部署与生产加固
