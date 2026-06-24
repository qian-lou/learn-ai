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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # gpt-3.5-turbo 已下线

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

*参考答案*：两段翻译 prompt 用 `|` 串联，前一步输出喂给后一步。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
to_en = ChatPromptTemplate.from_template("把下面的中文翻译成英文，只输出译文：\n{text}")
to_fr = ChatPromptTemplate.from_template("Translate the following English into French, output only the translation:\n{english}")

# LCEL 管道串联：中文 → 英文 → 法文 / pipe two translation steps
chain = (
    {"english": to_en | llm | StrOutputParser()}
    | to_fr | llm | StrOutputParser()
)
print(chain.invoke({"text": "机器学习正在改变世界"}))
```

**练习 2：** 构建一个带记忆的多轮对话机器人。

*参考答案*：用 `RunnableWithMessageHistory` 包裹 chain，按 session_id 维护历史。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_history(session_id: str):  # 按会话隔离历史 / per-session history
    return store.setdefault(session_id, ChatMessageHistory())

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的助手。"), ("placeholder", "{history}"), ("human", "{input}")])
chat = RunnableWithMessageHistory(
    prompt | ChatOpenAI(model="gpt-4o-mini"), get_history,
    input_messages_key="input", history_messages_key="history")

cfg = {"configurable": {"session_id": "u1"}}
chat.invoke({"input": "我叫小明"}, config=cfg)
print(chat.invoke({"input": "我叫什么？"}, config=cfg).content)  # → 小明
```

### 进阶题

**练习 3：** 用 Pydantic + JsonOutputParser 实现结构化信息提取。

*参考答案*：用 Pydantic 定义 schema，`JsonOutputParser` 把模型输出解析成字典。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    city: str = Field(description="所在城市")

parser = JsonOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_template(
    "从文本中提取信息。{format_instructions}\n\n文本：{text}")
# get_format_instructions 自动生成格式说明 / auto-injects schema instructions
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser
print(chain.invoke({"text": "张三今年 28 岁，住在上海。",
                    "format_instructions": parser.get_format_instructions()}))
```

**练习 4：** 构建一个"链式推理"系统：分析问题 → 拆分子问题 → 逐一回答 → 汇总。

*参考答案*：分解步用 LLM 产出子问题列表，`.batch()` 并行回答，再汇总成最终答案。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
decompose = ChatPromptTemplate.from_template(
    '将问题拆成 2-4 个子问题，输出 JSON 字符串数组：{question}') | llm | JsonOutputParser()
answer = ChatPromptTemplate.from_template("简洁回答：{q}") | llm | StrOutputParser()
synthesize = ChatPromptTemplate.from_template(
    "综合以下子问答，给出完整结论：\n{qa}") | llm | StrOutputParser()

def solve(question: str) -> str:
    subs = decompose.invoke({"question": question})
    qas = answer.batch([{"q": s} for s in subs])      # 并行回答子问题 / answer in parallel
    qa_text = "\n".join(f"Q:{q} A:{a}" for q, a in zip(subs, qas))
    return synthesize.invoke({"qa": qa_text})

print(solve("如何从零搭建一个 RAG 系统？"))
```
