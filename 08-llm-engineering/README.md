# 阶段八：大模型部署与工程化

> **预估周期**：2-3 周
> **核心目标**：量化优化 + 服务部署 + MLOps

---

## 📋 模块大纲

### [01-model-optimization](./01-model-optimization/) — 模型优化

减小模型体积、提升推理速度的关键技术。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [quantization](./01-model-optimization/01-quantization.md) | 模型量化（INT8/INT4/GPTQ/AWQ） |
| 02 | [inference-acceleration](./01-model-optimization/02-inference-acceleration.md) | 推理加速 |
| 03 | [knowledge-distillation](./01-model-optimization/03-knowledge-distillation.md) | 知识蒸馏 |

---

### [02-model-serving](./02-model-serving/) — 模型服务

将训练好的模型部署为可用的 API 服务。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [vllm-tgi](./02-model-serving/01-vllm-tgi.md) | vLLM / TGI 部署 |
| 02 | [api-service](./02-model-serving/02-api-service.md) | API 服务设计 |
| 03 | [docker-deployment](./02-model-serving/03-docker-deployment.md) | Docker 容器化部署 |

---

### [03-mlops](./03-mlops/) — MLOps 工程化

机器学习项目的工程化最佳实践。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [experiment-tracking](./03-mlops/01-experiment-tracking.md) | 实验追踪（MLflow/W&B） |
| 02 | [evaluation-and-monitoring](./03-mlops/02-evaluation-and-monitoring.md) | 评估与监控 |
| 03 | [cicd](./03-mlops/03-cicd.md) | ML 项目 CI/CD |
