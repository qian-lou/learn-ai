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
# docker-compose.yml（Compose Spec 已不需要 version 字段）
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

## 5. 例题（Worked Examples）

### 例题 1：编写面向生产环境的大模型 API 容器化多阶段 Dockerfile / Multi-stage Dockerfile

为了最小化发布体积，我们通常使用多阶段构建（Multi-stage Build）。以下例题演示如何打包运行 FastAPI 大模型推理服务器。

```dockerfile
# ============================================================
# 阶段 1: 构建依赖环境 / Stage 1: Build dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装必要的编译工具 / Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 将 Python 包安装到本地隔离路径以防止环境冲突 / Install deps locally
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================
# 阶段 2: 运行环境 / Stage 2: Final runtime
# ============================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# 拷贝阶段 1 中安装完的 python packages / Copy packages
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# 使用 uvicorn 运行大模型服务 / Start server
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：大模型镜像体积通常极大（十几GB以上），在编写 `.dockerignore` 时，我们必须忽略哪些目录以防止将本地庞大的模型权重文件打包进镜像层中？
*参考答案*：
应该在 `.dockerignore` 中写入所有的本地模型缓存和日志目录，如：
```
.git
.cache/
huggingface_hub/
weights/
*.bin
*.safetensors
```

### 进阶题
**练习 2**：在运行基于 GPU 推理容器（如 vLLM）时，如何通过 Docker Compose 启动容器并声明申请宿主机上的所有 GPU 显卡卡槽？
*参考答案*：
必须在 Docker Compose 文件的服务下添加 `deploy.resources.reservations.devices` 节点：
```yaml
services:    # Compose Spec 已不需要 version 字段
  llm-service:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
