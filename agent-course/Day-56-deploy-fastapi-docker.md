# Day 56 · 部署：把 Agent 包成 FastAPI 服务 + Docker 容器化

> **今日目标**：把一个本地能跑的 agent 包成 **FastAPI** 服务（含同步 + SSE 流式两个端点），再用多阶段 **Dockerfile** 容器化，本地 `docker run` 起来能调通。
> **时长**：~2h ｜ **前置**：Day 1~55（手上有任意一个能跑的 agent / LangGraph 图）
> **今日产出**：`app/main.py` + `Dockerfile` + `requirements.txt`，本地容器内 `curl /chat` 返回答案，`curl /chat/stream` 流式吐字。

## 1. 为什么 & 是什么（概念 + Java 类比）

前 55 天你的 agent 一直跑在 `python xxx.py` 里——那是"能 demo"。要"能上线"，第一步是把它变成**长驻的、可被 HTTP 调用的服务**。给 Java 工程师的直接类比：

| Python / FastAPI | Java 世界类比 | 说明 |
|---|---|---|
| `@app.post("/chat")` | `@PostMapping` | 路由声明，几乎一一对应 |
| Pydantic `BaseModel` 入参 | `@RequestBody DTO` + Bean Validation | 入参强类型校验，非法直接 422 |
| `uvicorn`（ASGI server） | 内嵌 Tomcat / Netty | 真正监听端口的 HTTP server |
| `async def` + `await` | WebFlux 的 `Mono`/`Flux` | 单线程事件循环扛高并发 I/O |
| `lifespan` 启动钩子 | `@PostConstruct` | 启动时建连接池、初始化客户端 |

两个必须建立的心智模型：①**Agent 是 I/O 密集型**——一次请求 90% 时间在等 LLM/工具返回，所以用 **async**（事件循环）而非"一请求一线程"，更像 WebFlux 而非阻塞的 Spring MVC，一个进程就能扛几百个在途请求；②**流式必须用 SSE**——普通 `POST` 是算完一次性返回，要像 ChatGPT 逐字冒得用 `text/event-stream`，Java 侧对应 `SseEmitter`/`Flux`。

## 2. 跟着做（Hands-on）

**Step 1 — 装依赖**（2026 现代栈）

```bash
pip install "fastapi>=0.115" "uvicorn[standard]>=0.32" "openai>=1.50" "pydantic>=2.9"
```

**Step 2 — `app/main.py`：把 agent 包成服务**

```python
"""Day 56: 把 agent 包成 FastAPI 服务 / wrap an agent as a FastAPI service.

端点 / endpoints: POST /chat（同步）, POST /chat/stream（SSE 流式）, GET /healthz（存活）。
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# 全局客户端容器，在 lifespan 里初始化（等价 Spring @PostConstruct）/ global holder
clients: dict[str, AsyncOpenAI] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动建连接、关闭释放 / set up & tear down clients."""
    clients["llm"] = AsyncOpenAI()  # 复用连接池，勿每请求新建 / reuse the pool
    yield
    await clients["llm"].close()


app = FastAPI(title="Agent Service", lifespan=lifespan)
MODEL = "gpt-4o-mini"


class ChatRequest(BaseModel):
    """聊天请求体 / chat request DTO（非法字段自动 422）。"""

    message: str = Field(min_length=1, max_length=4000, description="用户消息 / user msg")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, object]:
    """同步端点：把 req 喂给 agent，算完一次性返回 / blocking endpoint."""
    # 把这里换成你 Day 6~45 的真实 agent / swap in your real agent here
    resp = await clients["llm"].chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": req.message}],
        temperature=req.temperature,
    )
    return {"answer": resp.choices[0].message.content,
            "total_tokens": resp.usage.total_tokens}


async def sse_stream(req: ChatRequest) -> AsyncIterator[str]:
    """生成 SSE 数据帧 / yield SSE frames（每帧 `data: ...\\n\\n`）。"""
    stream = await clients["llm"].chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": req.message}],
        temperature=req.temperature, stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {delta}\n\n"  # SSE 协议帧格式 / SSE frame
    yield "data: [DONE]\n\n"  # 约定结束标记 / sentinel


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式端点：SSE 逐字返回 / streaming endpoint via SSE."""
    return StreamingResponse(sse_stream(req), media_type="text/event-stream")


@app.get("/healthz")  # 存活探针，永远轻量、不打外部依赖 / liveness, must stay cheap
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
```

**Step 3 — 本地起服务并自测**（`/docs` 还能看到自动生成的 Swagger UI）

```bash
uvicorn app.main:app --reload --port 8000   # 另开终端跑下面的 curl
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"hi"}'
curl -N -X POST localhost:8000/chat/stream \
  -H 'Content-Type: application/json' -d '{"message":"数到5"}'   # -N 关缓冲看流式
```

**Step 4 — `Dockerfile`：多阶段构建**（`requirements.txt` 即 Step 1 那四行）

```dockerfile
# 多阶段：builder 装依赖、final 只拷产物，镜像更小更安全 / multi-stage slim build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser   # 关键：非 root 用户运行 / non-root
COPY --from=builder /install /usr/local
COPY app ./app
USER appuser
EXPOSE 8000
# I/O 密集，worker 数可略多于核数 / I/O-bound: workers can exceed cores
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Step 5 — 构建并运行**（key 运行时 `-e` 注入，绝不写进镜像 / inject, never bake）

```bash
docker build -t agent-service:dev .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" agent-service:dev
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"hi"}'
```

## 3. 今日任务

1. 把**你自己 Day 6~45 做的某个 agent**（多工具 agent 或 LangGraph 图）替换掉 `/chat` 里的占位逻辑，让真实 agent 跑在服务里。
2. 跑通三件事：`/chat` 返回 JSON、`/chat/stream` 在 `curl -N` 下逐字冒、`/healthz` 返回 ok。
3. **容器化验收**：`docker build` + `docker run` 后宿主机 `curl` 能调通；`docker images` 看镜像大小（slim + 多阶段应在 ~200MB 量级）；再加个 `.dockerignore`（排除 `.venv/__pycache__/.git`）重 build 对比体积。

**验收标准**：容器内服务三个端点全部可用；镜像以非 root 用户运行（`docker exec` 进去 `whoami` 显示 `appuser`）；API key 是运行时 `-e` 注入而非写死在镜像里。

## 4. 自测清单

- [ ] 我能说清为什么 agent 服务该用 async / ASGI，而不是"一请求一线程"。
- [ ] 我知道流式为什么要用 SSE，对应 Java 的什么（`SseEmitter`/`Flux`）。
- [ ] 我理解多阶段 Dockerfile 为什么镜像更小、为什么要用非 root 用户。
- [ ] 我能解释 API key 为什么必须运行时注入、`lifespan` 对应 Spring 的哪个生命周期钩子。

## 5. 延伸 & 关联

- 本仓库 API 服务开发（FastAPI 更系统）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)｜Docker 部署细节：[../08-llm-engineering/02-model-serving/03-docker-deployment.md](../08-llm-engineering/02-model-serving/03-docker-deployment.md)
- 明天 Day 57 给这个服务补上**生产关注点**（限流/超时/并发/降级/健康检查），把"能跑"升级成"扛得住"。
- 本系列总计划：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
