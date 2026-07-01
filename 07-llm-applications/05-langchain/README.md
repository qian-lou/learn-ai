# 05-langchain — LangChain 框架

> **所属阶段**：阶段七 · 大模型应用实战
> **学习目标**：用 LangChain/LangGraph 把 Prompt + RAG + 工具编排成可交付的 LLM 应用
> **预估时长**：5-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [langchain-basics](./01-langchain-basics.md) | LangChain 核心概念 | LCEL 管道 `prompt\|llm\|parser`、多步 Chain、Memory 会话记忆、Pydantic + JsonOutputParser 结构化输出、batch/stream |
| 02 | [agents-and-tools](./02-agents-and-tools.md) | Agents 与工具集成 | `@tool` 自定义工具（AST 安全求值）、`create_react_agent`、OpenAI Function Calling、SQL Agent、Human-in-the-loop |
| 03 | [full-application](./03-full-application.md) | 完整应用开发实战 | FastAPI + SSE 流式接口、Gradio/Streamlit 快速 UI、生产上线清单（限流/计费/日志）、架构选型 |

---

## 🔑 知识点详解

### 01 · LangChain 核心概念（LCEL）

- **核心概念**：LCEL（LangChain Expression Language）用 `|` 管道把 prompt、模型、解析器声明式地串起来，天然支持流式、批量、异步；Memory/Retriever 等抽象让复杂应用几十行搞定。
- **关键 API**：
  - `chain = prompt | llm | StrOutputParser()`，`chain.invoke({...})`；`.batch([...])` 批量、`.stream({...})` 流式。
  - `RunnableWithMessageHistory(chain, get_session_history, input_messages_key=, history_messages_key=)` — 按 session_id 维护对话记忆。
  - `JsonOutputParser(pydantic_object=Model)` + `parser.get_format_instructions()` — 结构化输出。
- **易错点**：
  - 多步 Chain 里前一步输出的键名要和后一步 prompt 变量名对上（用 `{"key": subchain}` 显式命名）。
  - Memory 不隔离 session_id 会串会话；`get_session_history` 需按 id 返回独立历史。
- **Java 视角**：Chain ≈ Servlet Filter Chain（管道式处理），Memory ≈ HttpSession，Retriever ≈ Repository，LCEL 的 `|` ≈ Unix 管道 / Stream 的链式 `map`。
- **前置**：模块 02（Prompt）；结构化输出承接 02 章的 Pydantic。

### 02 · Agents 与工具

- **核心概念**：Agent 让 LLM 不止能"说"还能"做"——模型按 ReAct 循环自主决定调哪个工具、传什么参数、看结果后是否继续，直到给出最终答案。
- **关键 API**：
  - `@tool` 装饰器把普通函数（带 docstring 说明）变成工具；docstring 就是模型选工具的依据。
  - `create_react_agent(llm, tools)`（LangGraph 预制）**替代已废弃的 `initialize_agent`**；`agent.invoke({"messages": [("user", ...)]})`。
  - Function Calling：`tools=[{"type":"function","function":{...schema...}}]`，解析 `tool_calls` → 执行 → 把结果作为 `role:"tool"` 消息回传。
  - HITL：`create_react_agent(..., checkpointer=MemorySaver(), interrupt_before=["tools"])` 在工具执行前暂停等人工确认。
- **易错点**：
  - **计算器/代码类工具绝不能 `eval()` 不可信输入**（RCE 风险）——用 AST 白名单求值。
  - SQL Agent 务必连**只读账号**，防模型误执行写/删。
  - 工具描述含糊会导致选错工具或参数解析失败；死循环用迭代上限兜底。
- **Java 视角**：Agent ≈ Controller（做决策）+ Service（工具执行操作）；ReAct 循环 ≈ 请求处理循环里"调服务→看返回→再决策"。
- **前置**：01（Chain/工具接入）、02 章 ReAct 理论。

### 03 · 完整应用开发

- **核心概念**：把 LLM 能力封装成可用产品——REST API、流式响应、Web UI、上线保障，是从"Demo"到"生产系统"的最后一步。
- **关键 API/清单**：
  - FastAPI 流式：`async for chunk in llm.astream(...)` → `yield f"data: {chunk.content}\n\n"`，包进 `StreamingResponse(media_type="text/event-stream")`（SSE）。
  - Gradio：`gr.ChatInterface(fn, type="messages")` 几行起一个聊天界面（Gradio 4/5 推荐 `type="messages"`，旧 `"tuples"` 已弃用）。
  - 上线清单：限流（slowapi）、Token 计费（`get_openai_callback`）、日志、缓存、输入输出安全过滤、成本监控。
- **易错点**：
  - SSE 每条消息格式必须是 `data: ...\n\n`，漏掉双换行前端收不到。
  - 生产接口不做限流/成本监控极易被刷爆预算。
- **Java 视角**：FastAPI ≈ Spring Controller，Gradio ≈ Swagger UI（快速可交互），SSE ≈ 单向 WebSocket；整套 ≈ Spring Boot + 前端构建 Web 应用，只是核心逻辑换成 LLM。
- **前置**：01、02，以及模块 03 的 RAG 链（常作为后端核心）。

---

## 🎯 学习要点

- **LCEL 是新时代默认写法**：优先用 `prompt | llm | parser` 组合，而非老式 `LLMChain`；它免费带来流式、批量、异步能力。
- **Agent 用 LangGraph 预制件**：`create_react_agent` 取代 `initialize_agent`；需要断点续跑/人工确认时叠 `checkpointer` + `interrupt_before`。
- **工具安全是硬要求**：AST 白名单替代 `eval`、只读 DB 账号、危险动作 HITL——这些不是可选项而是默认项。
- **结构化输出优先 Pydantic**：给下游用的数据用 `JsonOutputParser(pydantic_object=...)` 或模型原生 `parse()` 拿强类型对象，别手写正则。
- **原型与生产分层选型**：验证想法用 Gradio/Streamlit，内部工具 Streamlit+FastAPI，生产 API 上 FastAPI+Docker+K8s。
- **上线前过一遍清单**：限流、计费、日志、缓存、安全过滤一个都不能少，这是 Demo 与产品的分水岭。

---

## 🔗 关联

- **上一模块**：[04-fine-tuning](../04-fine-tuning/) — 微调好的模型可直接接入本模块的 Chain/Agent。
- **下一步**：本模块是阶段七收口；深入 Agent 工程化（LangGraph 状态机、多 Agent、可观测性、部署）见配套的 [agent-course/](../../agent-course/)。
- **本阶段总览**：[阶段七 README](../README.md)
- **相关 Day**：[Day 6 Tool Calling 原理](../../agent-course/Day-06-tool-calling-basics.md) · [Day 9 ReAct 循环](../../agent-course/Day-09-react-loop.md) · [Day 26 LangGraph 入门](../../agent-course/Day-26-langgraph-intro.md) · [Day 30 Human-in-the-loop](../../agent-course/Day-30-human-in-the-loop.md) · [Day 56 部署 FastAPI + Docker](../../agent-course/Day-56-deploy-fastapi-docker.md)。
