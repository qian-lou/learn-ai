# API 服务开发 / API Service Development

## 1. 背景（Background）

> **为什么要学这个？**
>
> 将大模型封装为 **RESTful API** 是生产落地的标准方式。FastAPI 是 Python 的高性能 Web 框架，支持异步、自动文档、类型校验——对 Java 工程师来说，它就是 **Python 版 Spring Boot**。

## 2. 知识点（Key Concepts）

| FastAPI 概念 | Spring Boot 对应 |
|-------------|-----------------|
| `@app.post` | `@PostMapping` |
| `BaseModel` | DTO (Pydantic) |
| `Depends` | `@Autowired` |
| `middleware` | `Filter` |
| `uvicorn` | Tomcat |

## 3. 内容（Content）

### 3.1 完整 LLM API 服务

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import time

app = FastAPI(title="LLM API Service", version="1.0")

# CORS 配置
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# ============================================================
# 请求/响应模型（类似 Java DTO）
# ============================================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: system/user/assistant")
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=4096)
    stream: bool = False

class ChatResponse(BaseModel):
    content: str
    usage: dict

# ============================================================
# API 端点
# ============================================================
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(
            stream_generate(req), media_type="text/event-stream"
        )
    # response = await model.generate(req.messages)
    response = "LLM 回复内容"
    return ChatResponse(content=response, usage={"total_tokens": 100})

async def stream_generate(req: ChatRequest):
    """SSE 流式输出 / Server-Sent Events streaming."""
    tokens = "这是一个流式输出的示例回复".split()
    for token in tokens:
        yield f"data: {token}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"

# ============================================================
# 中间件：限流 + 日志
# ============================================================
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} - {duration:.3f}s")
    return response

# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4. 详细推理（Deep Dive）

### 4.1 生产环境要点

```
生产 API 必备组件：
  ✅ 限流（防 DDoS）      → slowapi / Redis
  ✅ 认证（API Key）       → FastAPI Depends
  ✅ 监控（延迟/错误率）   → Prometheus + Grafana
  ✅ 日志（结构化日志）    → structlog
  ✅ 缓存（相同问题缓存）  → Redis
  ✅ 负载均衡              → Nginx / Traefik
```

## 5. 例题（Worked Examples）

### 例题 1：基于 FastAPI 实现与 OpenAI 兼容的流式响应接口 / OpenAI-compatible Streaming API

大模型生成耗时较长，因此“打字机式”的流式输出（Streaming）是生产环境必不可少的。以下展示如何用 FastAPI 搭建此类高并发接口。

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

async def generate_mock_stream(prompt: str) -> AsyncGenerator[str, None]:
    """模拟大模型逐字生成回复的异步生成器 / Mock LLM generation.
    
    Time: O(T * N) - T 为生成 token 数 / T is token count.
    Space: O(1)
    """
    reply = f"已收到提示词: '{prompt}'。以下是我的逐步回答：开发高并发 API 必须使用异步库..."
    for char in reply:
        yield f"data: {char}\n\n"  # 遵守 SSE (Server-Sent Events) 格式约定 / SSE format.
        await asyncio.sleep(0.05)   # 模拟生成等待 / Simulate delay.
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    return StreamingResponse(
        generate_mock_stream(request.prompt),
        media_type="text/event-stream"
    )

# 启动命令 / Start with: uvicorn api_service:app --reload
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在 FastAPI 接口开发中，`def` 函数与 `async def` 异步函数在使用场景上有什么本质区别？
*参考答案*：
- `async def`：适用于内部调用了异步阻塞 I/O 操作（例如用 `httpx` 调用第三方大模型 API，或者使用异步数据库库），FastAPI 会在主事件循环中调度，非常节约线程资源。
- `def`：适用于纯 CPU 密集运算或只有同步阻塞 I/O 的方法，FastAPI 会将其抛入内部自带的专属线程池中执行，防止阻塞主事件循环。

### 进阶题
**练习 2**：在生产环境的流式大模型 API 中，请为上面的 FastAPI 服务设计并编写一个中间件，用来统计首包延迟（TTFT - Time to First Token）以及整体推理生成吞吐量（Tokens per Second），并将数据记录到日志中。
*参考答案*：
```python
import time
from fastapi import Request

# 中间件逻辑伪代码实现
async def log_llm_metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    
    # 模拟在 SSE 流中挂钩子拦截首个 token 弹出的时间
    # TTFT = time_first_token_sent - start_time
    # total_duration = time.perf_counter() - start_time
    # log(f"TTFT: {TTFT}s, Throughput: {total_tokens / total_duration} tokens/s")
    return response
```\n