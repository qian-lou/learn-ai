# Agent 与工具调用 / Agents and Tool Calling

## 1. 背景（Background）
> Agent 让 LLM 能调用外部工具（搜索、计算、API），实现复杂任务自动化。这是大模型落地的前沿方向。

## 2-3. 知识点与内容
```python
from langchain.agents import create_react_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# 定义工具
search = DuckDuckGoSearchRun()
tools = [Tool(name="搜索", func=search.run, description="搜索互联网信息")]

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# Function Calling (OpenAI 原生)
functions = [{
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}]
response = llm.invoke("今天北京天气如何？", functions=functions)
```

## 4-6. 推理/例题/习题
**练习：** 构建一个能搜索+计算+查数据库的多工具 Agent。
