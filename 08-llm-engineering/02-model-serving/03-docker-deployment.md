# Docker 容器化部署 / Docker Containerization

## 1. 背景（Background）

> **为什么要学这个？**
>
> Docker 容器化是大模型部署的标准方式，确保**环境一致性**——开发环境能跑的模型，生产环境也一定能跑。Java 工程师对 Docker 应该非常熟悉，AI 部署只是加了 GPU 支持。

## 2. 知识点（Key Concepts）

| 组件 | 功能 |
|------|------|
| nvidia/cuda 基础镜像 | GPU 支持 |
| docker compose | 多服务编排 |
| GPU passthrough | `--gpus all` |
| Volume mount | 模型文件挂载 |

## 3. 内容（Content）

### 3.1 Dockerfile

```dockerfile
# ============================================================
# LLM 服务 Dockerfile
# ============================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 模型缓存目录
ENV HF_HOME=/app/models
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 Docker Compose

```yaml
# docker-compose.yml
version: "3.8"
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
      - ./models:/app/models    # 模型文件
      - ./logs:/app/logs        # 日志
    environment:
      - MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
      - MAX_MODEL_LEN=4096
    restart: unless-stopped

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - llm-api

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 3.3 Nginx 配置

```nginx
# nginx.conf
upstream llm_backend {
    server llm-api:8000;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;      # LLM 生成可能很慢
        proxy_buffering off;          # 流式输出必须关闭缓冲
    }
}
```

## 4. 详细推理（Deep Dive）

### 4.1 GPU Docker 要点

```
关键注意事项：

1. 安装 NVIDIA Container Toolkit
   → sudo apt install nvidia-container-toolkit

2. 镜像大小优化
   基础镜像: nvidia/cuda:12.1.0-runtime (~3GB)
   不要用 devel 版本 (~8GB)

3. 模型文件挂载
   不要把模型打进镜像！用 volume 挂载
   → 镜像小，模型可更换

4. 多阶段构建减小镜像
```

## 5-6. 例题/习题

**练习 1：** 用 Docker Compose 部署 vLLM + Nginx。

**练习 2：** 实现多 GPU 部署（tensor parallel）。

**练习 3：** 添加 Prometheus metrics 收集和 Grafana 监控面板。
