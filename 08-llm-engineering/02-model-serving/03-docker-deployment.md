# Docker 容器化部署 / Docker Containerization

## 1. 背景（Background）
> Docker 容器化是大模型部署的标准方式，确保环境一致性。Java 工程师对 Docker 应该很熟悉。

## 2-3. 知识点与内容
```dockerfile
# Dockerfile for LLM service
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  llm-api:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
```

## 4-6. 推理/例题/习题
**练习：** 用 Docker Compose 部署 vLLM + FastAPI + Nginx 的完整服务。
