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

## 5-6. 例题/习题

**练习 1：** 构建 OpenAI 兼容的 API，支持流式输出。

**练习 2：** 添加 API Key 认证和限流中间件。

**练习 3：** 用 wrk/locust 做压力测试，分析 QPS 和 P99 延迟。
