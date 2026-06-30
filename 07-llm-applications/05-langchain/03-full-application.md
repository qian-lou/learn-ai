# 完整应用开发 / Full Application Development

## 1. 背景（Background）

> **为什么要学这个？**
>
> 将 LLM 能力封装为**可用的产品**——API 服务、Web 界面、对话流程管理——是从"技术 Demo"到"生产系统"的最后一步。本节学习用 FastAPI + Gradio/Streamlit 构建完整 LLM 应用。
>
> 对于 Java 工程师来说，这就像用 **Spring Boot + Thymeleaf** 构建 Web 应用——只不过核心逻辑换成了 LLM。

## 2. 知识点（Key Concepts）

| 工具 | 用途 | Java 类比 |
|------|------|----------|
| FastAPI | REST API 服务 | Spring Controller |
| Gradio | 快速 UI 原型 | Swagger UI |
| Streamlit | 数据应用 | Vaadin |
| SSE | 流式响应 | WebSocket |

## 3. 内容（Content）

### 3.1 FastAPI 构建 LLM API

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="LLM API Service")

class ChatRequest(BaseModel):
    message: str
    history: list = []
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int = 0

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """基础聊天接口 / Basic chat endpoint."""
    # response = await llm.ainvoke(req.message)
    response = "这是 LLM 的回复"
    return ChatResponse(reply=response)


# ============================================================
# 流式输出（SSE）— 像 ChatGPT 一样逐字输出
# ============================================================
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        # async for chunk in llm.astream(req.message):
        #     yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 3.2 Gradio 快速 UI

```python
import gradio as gr

# ============================================================
# Gradio: 5 行代码构建聊天界面
# ============================================================
def chatbot(message, history):
    """聊天回调函数."""
    # response = llm.invoke(message).content
    response = f"收到消息: {message}"
    return response

demo = gr.ChatInterface(
    chatbot,
    type="messages",                # Gradio 4/5 推荐用 messages 格式（history 为 OpenAI 风格 dict 列表）
    title="🤖 AI 助手",
    description="基于大模型的智能对话",
    examples=["介绍一下机器学习", "写一首关于春天的诗"],
    theme="soft",
)
# demo.launch(server_port=7860, share=True)

# 版本提示：Gradio 4/5 的 ChatInterface 推荐 type="messages"（旧默认 "tuples" 已弃用）；
# 回调返回字符串或字典（如 {"role": "assistant", "content": ...}）均可。
# Note: on Gradio 4/5 prefer type="messages"; the callback may return a str or a dict.
```

### 3.3 Streamlit 数据应用

```python
import streamlit as st

st.set_page_config(page_title="AI 助手", page_icon="🤖")
st.title("🤖 AI 文档问答")

# 侧边栏配置
with st.sidebar:
    api_key = st.text_input("API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

# 文件上传
uploaded_file = st.file_uploader("上传文档", type=["pdf", "txt"])

# 对话界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("请输入问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = f"回答: {prompt}"
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### 3.4 生产部署清单

```
LLM 应用上线检查清单：

□ API 限流（防止滥用）
□ 输入/输出安全过滤
□ Token 使用量监控
□ 错误处理和重试
□ 日志记录
□ 缓存策略（相同问题缓存回答）
□ A/B 测试框架
□ 成本监控
```

## 4. 详细推理（Deep Dive）

### 4.1 架构选型

```
原型阶段:    Gradio / Streamlit (快速验证)
内部工具:    Streamlit + FastAPI
生产 API:    FastAPI + Docker + K8s
企业级应用:  FastAPI + React/Vue 前端
```

## 5. 例题（Worked Examples）

```python
# 完整 RAG 应用 = FastAPI + LangChain + Gradio
# 架构: 用户 → Gradio UI → FastAPI → LangChain RAG → LLM
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 Gradio 构建一个翻译应用（中英互译）。

*参考答案*：用一个 LCEL 翻译链做回调，`gr.Interface` 提供输入框与下拉方向选择。

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (ChatPromptTemplate.from_template("把下面的文本翻译成{target}，只输出译文：\n{text}")
         | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser())

def translate(text: str, direction: str) -> str:
    target = "英文" if direction == "中→英" else "中文"
    return chain.invoke({"text": text, "target": target})

demo = gr.Interface(translate,
    inputs=[gr.Textbox(label="原文"), gr.Radio(["中→英", "英→中"], value="中→英")],
    outputs=gr.Textbox(label="译文"), title="🌐 翻译助手")
# demo.launch()
```

**练习 2：** 用 FastAPI 构建 LLM API，实现流式输出。

*参考答案*：用 `llm.astream` 异步逐块产出，包进 `StreamingResponse` 以 SSE 推送。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        # astream 异步逐块产出，按 SSE 协议封装 / async token stream as SSE
        async for chunk in llm.astream(req.message):
            yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### 进阶题

**练习 3：** 构建完整 RAG 应用：FastAPI 后端 + Gradio 前端 + Chroma 向量库。

*参考答案*：后端把 LCEL 检索链暴露为 `/ask` 接口，前端 Gradio `ChatInterface` 调用该接口。

```python
# ---- backend.py：FastAPI + Chroma + LCEL ----
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

vs = Chroma(persist_directory="./db",
            embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5"))
prompt = ChatPromptTemplate.from_template("依据上下文回答。\n\n上下文：{context}\n\n问题：{input}")
chain = create_retrieval_chain(
    vs.as_retriever(search_kwargs={"k": 3}),
    create_stuff_documents_chain(ChatOpenAI(model="gpt-4o-mini"), prompt))

app = FastAPI()
class Q(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Q):
    return {"answer": chain.invoke({"input": q.question})["answer"]}

# ---- frontend.py：Gradio 调用后端 / Gradio calls the backend ----
# import gradio as gr, requests
# def chat(msg, history):
#     return requests.post("http://localhost:8000/ask", json={"question": msg}).json()["answer"]
# gr.ChatInterface(chat, title="📄 文档问答").launch()
```

**练习 4：** 添加 Token 计费、限流和日志功能，使其可生产部署。

*参考答案*：用 `get_openai_callback` 统计 token/成本，`slowapi` 做限流，标准 `logging` 记录请求。

```python
import logging
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from langchain_community.callbacks import get_openai_callback

logging.basicConfig(level=logging.INFO)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")        # 每 IP 每分钟 10 次 / 10 req/min per IP
async def chat(request: Request, message: str):
    # 回调统计 token 与美元成本 / track tokens and USD cost
    with get_openai_callback() as cb:
        reply = chain.invoke({"input": message})["answer"]
    logging.info("tokens=%d cost=$%.4f", cb.total_tokens, cb.total_cost)
    return {"reply": reply, "tokens": cb.total_tokens}
```
