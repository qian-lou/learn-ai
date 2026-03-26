# LangChain 基础 / LangChain Basics

## 1. 背景（Background）
> LangChain 是构建 LLM 应用的主流框架，提供 Chain、Agent、Memory 等抽象。类似 Java 的 Spring 框架之于 Web 开发。

## 2-3. 知识点与内容
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 基础 Chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("翻译成{language}: {text}")
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(language="英文", text="你好世界")

# LCEL (LangChain Expression Language) — 新推荐方式
chain = prompt | llm
result = chain.invoke({"language": "英文", "text": "你好世界"})
```

## 4-6. 推理/例题/习题
**练习：** 用 LangChain 构建一个带记忆的对话机器人。
