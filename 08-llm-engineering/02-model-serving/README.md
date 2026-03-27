# 02-model-serving — 模型服务

> **所属阶段**：阶段八 · 大模型部署与工程化
> **学习目标**：将训练好的大模型部署为高性能 API 服务

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [vllm-tgi](./01-vllm-tgi.md) | vLLM / TGI 部署 | vLLM PagedAttention、TGI 使用、性能对比 |
| 02 | [api-service](./02-api-service.md) | API 服务设计 | OpenAI-compatible API、流式输出、限流 |
| 03 | [docker-deployment](./03-docker-deployment.md) | Docker 容器化部署 | Dockerfile 编写、GPU 容器配置、编排 |

---

## 🎯 学习要点

- vLLM 是当前最高效的 LLM 推理框架之一
- 设计兼容 OpenAI 格式的 API 便于客户端集成
- Docker 容器化是生产部署的标准方式
