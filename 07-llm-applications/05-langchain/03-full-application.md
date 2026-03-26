# 完整应用开发 / Full Application Development

## 1. 背景（Background）
> 将 LLM 能力封装为可用的应用产品，包括 API 服务、Web 界面、对话流程管理等。

## 2-3. 知识点与内容
```python
# FastAPI 构建 LLM API 服务（Java 背景：类似 Spring Controller）
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    response = await llm.ainvoke(req.message)
    return {"reply": response.content}

# Gradio 快速构建 UI
import gradio as gr
def chatbot(message, history):
    return llm.invoke(message).content

demo = gr.ChatInterface(chatbot)
demo.launch()

# Streamlit 构建数据应用
import streamlit as st
st.title("AI 助手")
user_input = st.text_input("请输入问题：")
if user_input:
    st.write(llm.invoke(user_input).content)
```

## 4-6. 推理/例题/习题
**练习：** 用 FastAPI + Gradio 构建一个完整的 RAG 问答应用。
