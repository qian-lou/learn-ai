# LangChain 基础 / LangChain Basics

## 1. 背景（Background）

> **为什么要学这个？**
>
> LangChain 是构建 LLM 应用的**主流框架**，提供了 Chain、Agent、Memory、Retriever 等高级抽象。它让你用几十行代码就能构建复杂的 AI 应用。
>
> 对于 Java 工程师来说，LangChain 就是 **AI 版的 Spring Framework**——Chain 类似 Filter Chain，Agent 类似 Controller，Memory 类似 Session，Retriever 类似 Repository。

## 2. 知识点（Key Concepts）

| LangChain 概念 | 功能 | Java 类比 |
|---------------|------|----------|
| Chain | 组合多个步骤 | Filter Chain |
| Prompt Template | 提示词模板 | Thymeleaf |
| Memory | 对话上下文 | HttpSession |
| Retriever | 检索器 | Repository |
| Agent | 自主决策 | Controller |
| Tool | 外部工具 | Service |

## 3. 内容（Content）

### 3.1 基础 Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# LCEL (LangChain Expression Language) — 推荐方式
# ============================================================

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 简单 Chain
prompt = ChatPromptTemplate.from_template("用简洁的语言解释什么是{concept}")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"concept": "机器学习"})
print(result)


# ============================================================
# 多步 Chain（管道组合）
# ============================================================
# 第一步：生成概念解释
explain_prompt = ChatPromptTemplate.from_template("解释{concept}的核心原理，限100字")
# 第二步：生成代码示例
code_prompt = ChatPromptTemplate.from_template("根据以下解释，给出Python代码示例：\n{explanation}")

chain = (
    {"explanation": explain_prompt | llm | StrOutputParser()}
    | code_prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({"concept": "快速排序"})
```

### 3.2 对话记忆（Memory）

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ============================================================
# 带记忆的对话 / Conversation with memory
# ============================================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手。"),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

chain = prompt | llm

with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 多轮对话
config = {"configurable": {"session_id": "user123"}}
response1 = with_memory.invoke({"input": "我叫小明"}, config=config)
response2 = with_memory.invoke({"input": "我叫什么？"}, config=config)
# → "你叫小明"（记住了上下文）
```

### 3.3 结构化输出

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分 1-10")
    summary: str = Field(description="一句话总结")

parser = JsonOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_template(
    "分析以下电影评论，{format_instructions}\n\n评论：{review}"
)

chain = prompt | llm | parser
result = chain.invoke({
    "review": "《星际穿越》太震撼了，诺兰的想象力无与伦比！",
    "format_instructions": parser.get_format_instructions(),
})
# result = {"title": "星际穿越", "rating": 9, "summary": "..."}
```

## 4. 详细推理（Deep Dive）

### 4.1 LCEL 的设计哲学

```
LCEL 用 | (管道) 运算符组合组件:

  prompt | llm | parser

等价于:
  parser(llm(prompt(input)))

优势:
  1. 声明式编程（像 Unix 管道）
  2. 自动支持流式输出
  3. 自动支持批量处理
  4. 自动支持异步
```

## 5. 例题（Worked Examples）

```python
# 批量处理
results = chain.batch([
    {"concept": "机器学习"},
    {"concept": "深度学习"},
    {"concept": "强化学习"},
])

# 流式输出
for chunk in chain.stream({"concept": "Transformer"}):
    print(chunk, end="", flush=True)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 LCEL 构建一个翻译 Chain：中文 → 英文 → 法文。

**练习 2：** 构建一个带记忆的多轮对话机器人。

### 进阶题

**练习 3：** 用 Pydantic + JsonOutputParser 实现结构化信息提取。

**练习 4：** 构建一个"链式推理"系统：分析问题 → 拆分子问题 → 逐一回答 → 汇总。
