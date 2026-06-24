# Agent 与工具调用 / Agents and Tool Calling

## 1. 背景（Background）

> **为什么要学这个？**
>
> AI Agent 是大模型落地的**前沿方向**——让 LLM 不仅能"说"，还能"做"。Agent 可以调用搜索引擎、执行代码、查询数据库。
>
> 对于 Java 工程师来说，Agent 就像 **Controller + Service**——LLM 做决策，Tool 执行操作。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| ReAct | 思考→行动→观察 循环 |
| Function Calling | LLM 结构化调用函数 |
| Tool | 外部功能封装 |

## 3. 内容（Content）

### 3.1 自定义 Tool

```python
import ast
import operator
from langchain_core.tools import tool

# 安全算术求值：用 AST 白名单替代 eval()，避免任意代码执行 (RCE)
# Safe arithmetic eval via AST whitelist — never eval() untrusted input (RCE risk)
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):      # 数字字面量 / numeric literal
        return node.value
    if isinstance(node, ast.BinOp):         # 二元运算 / binary op
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):       # 一元运算 / unary op
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式 / unsupported expression")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式（仅 + - * / ** 与括号）/ Evaluate a safe arithmetic expression."""
    return str(_safe_eval(ast.parse(expression, mode="eval").body))

@tool
def get_current_time() -> str:
    """获取当前时间 / Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]
```

### 3.2 创建 Agent

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 用 LangGraph 预制 ReAct Agent（替代已废弃的 initialize_agent）
# Use LangGraph's prebuilt ReAct agent (replaces deprecated initialize_agent)
agent = create_react_agent(llm, tools)

# ReAct 循环：Thought → Action → Observation → ... → Final Answer
result = agent.invoke({"messages": [("user", "计算 15+27 然后查北京天气")]})
print(result["messages"][-1].content)
```

### 3.3 Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()
functions = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "北京天气怎样？"}],
    tools=functions,
)
tool_call = response.choices[0].message.tool_calls[0]
print(f"调用: {tool_call.function.name}({tool_call.function.arguments})")
```

### 3.4 Agent 执行流程

```
ReAct Agent 循环：

用户: "计算 15+27 然后查天气"

Thought 1: 先计算
Action 1: calculator("15+27")  → 42

Thought 2: 再查天气
Action 2: search("北京天气")  → 晴，25°C

Final Answer: 15+27=42，北京晴25°C。
```

## 4. 详细推理（Deep Dive）

### 4.1 Agent 的挑战

```
常见问题与解决：
  工具选择错误 → 清晰的工具描述
  参数解析错误 → 严格的参数 schema
  循环调用     → max_iterations 限制
  安全风险     → Human-in-the-loop
```

## 5. 例题（Worked Examples）

```python
# 多工具 Agent / Multi-tool agent
result = agent.invoke({"messages": [("user", "搜索 Python 3.12 新特性并总结")]})
print(result["messages"][-1].content)  # Agent 自动：选择工具 → 调用 → 总结
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 创建一个计算器+时间查询的 Agent。

*参考答案*：复用 3.1 节的安全 `calculator` 与 `get_current_time` 两个 tool，交给 LangGraph 预制 ReAct Agent。

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# calculator / get_current_time 见 3.1（AST 白名单，杜绝 eval）/ from §3.1, AST-safe

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# create_react_agent 替代已废弃的 initialize_agent / replaces deprecated initialize_agent
agent = create_react_agent(llm, [calculator, get_current_time])
res = agent.invoke({"messages": [("user", "现在几点？再算一下 (3+5)*2")]})
print(res["messages"][-1].content)
```

**练习 2：** 用 Function Calling 实现天气查询。

*参考答案*：声明 tool schema 让模型决定调用，解析 tool_call 后执行真实函数，再把结果回传模型。

```python
import json
from openai import OpenAI
client = OpenAI()

def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"  # 实际接入天气 API / call a real weather API here

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "获取城市天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
msgs = [{"role": "user", "content": "北京天气怎样？"}]
resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, tools=tools)
call = resp.choices[0].message.tool_calls[0]
# 解析参数→执行→把结果作为 tool 消息回传 / parse args, run, feed result back
args = json.loads(call.function.arguments)
msgs += [resp.choices[0].message,
         {"role": "tool", "tool_call_id": call.id, "content": get_weather(**args)}]
print(client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
      .choices[0].message.content)
```

### 进阶题

**练习 3：** 构建能读数据库、生成 SQL 的 Agent。

*参考答案*：用 LangChain 的 `SQLDatabaseToolkit` 把"看 schema / 生成 SQL / 执行查询"封装成工具集，交给 ReAct Agent。生产中务必用只读账号防止误写。

```python
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

# 建议连接只读账号，避免 Agent 误执行写操作 / use a READ-ONLY DB user
db = SQLDatabase.from_uri("sqlite:///company.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# toolkit 提供 list_tables / schema / query 等工具 / schema-aware SQL tools
tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

agent = create_react_agent(llm, tools)
res = agent.invoke({"messages": [("user", "销售额最高的 3 个产品是什么？")]})
print(res["messages"][-1].content)
```

**练习 4：** 实现 Human-in-the-loop 确认机制。

*参考答案*：用 LangGraph 的 checkpointer + `interrupt_before` 在工具执行前暂停，人工确认后再 resume。

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# interrupt_before=["tools"]：调用工具前暂停等待人工 / pause before any tool call
agent = create_react_agent(llm, tools, checkpointer=MemorySaver(),
                           interrupt_before=["tools"])
cfg = {"configurable": {"thread_id": "t1"}}

state = agent.invoke({"messages": [("user", "删除 id=5 的订单")]}, config=cfg)
pending = state["messages"][-1].tool_calls       # 查看待执行工具 / inspect pending action
if input(f"确认执行 {pending}? (y/n) ") == "y":
    # 传 None 表示批准并继续 / resume by passing None
    final = agent.invoke(None, config=cfg)
    print(final["messages"][-1].content)
```
