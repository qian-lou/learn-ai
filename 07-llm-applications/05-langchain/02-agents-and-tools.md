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
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]
```

### 3.2 创建 Agent

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ReAct 执行循环:
# Thought → Action → Observation → ... → Final Answer
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
    model="gpt-4-turbo",
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
# 多工具 Agent
# executor.invoke({"input": "搜索Python 3.12新特性并总结"})
# Agent 自动: search → 总结结果
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 创建一个计算器+时间查询的 Agent。

**练习 2：** 用 Function Calling 实现天气查询。

### 进阶题

**练习 3：** 构建能读数据库、生成 SQL 的 Agent。

**练习 4：** 实现 Human-in-the-loop 确认机制。
