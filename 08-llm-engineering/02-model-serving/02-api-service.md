# API 服务开发 / API Service Development

## 1. 背景（Background）
> 用 FastAPI 将大模型封装为 RESTful API 服务，类似 Java Spring Boot 构建微服务。

## 2-3. 知识点与内容
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LLM API")

class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    content: str
    usage: dict

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 调用 vLLM/本地模型
    response = await model.generate(req.messages, req.temperature)
    return ChatResponse(content=response, usage={"tokens": len(response)})

# 流式响应 / Streaming response (SSE)
from fastapi.responses import StreamingResponse

@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        async for chunk in model.stream(req.messages):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 4-6. 推理/例题/习题
**练习：** 构建完整的 LLM API，支持流式输出和并发控制。
