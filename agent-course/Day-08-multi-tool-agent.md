# Day 8 · 多工具 Agent：模型如何在 4 个工具里选对的那个

> **今日目标**：给 Agent 装 4 个工具（算术 / 查外部数据 / 读文件 / 查数据库），观察模型如何**自主路由**到正确的工具，甚至并行调用。
> **时长**：~2h ｜ **前置**：Day 6~7
> **今日产出**：一个 `day08_multi_tool.py`，一个 Agent 持有 4 个工具，能按问题自动选工具并回答。

## 1. 为什么 & 是什么

单工具是玩具。真实 Agent 手里有一**工具箱**，价值在于模型能根据问题**自己选对工具**——这叫 **tool routing（工具路由）**。今天重点不是写工具，而是**观察模型的选择行为**：何时选算术、何时选数据库、何时一次并行调两个。给 Java 工程师的类比：这像 Spring 的**策略模式 + 自动装配**——容器里注册一堆 `Strategy` Bean，运行时挑一个执行；区别是"挑哪个"的决策者是 **LLM**，不是你写的 `if-else`。模型靠什么选对？**全靠每个工具的 `description` 和参数 schema**——描述越互斥路由越准。今天最重要的直觉：**工具描述 = 给模型的路由表**。

四类典型工具，各有不可逾越的**安全红线**：`calculator`（算术，**绝不 `eval()`**，用 AST 白名单）、`get_stock_price`（查外部 API，真实场景要超时+重试）、`read_file`（读本地文件，**限定目录**防路径穿越）、`query_users`（查库，**只读账号**+参数化）。

## 2. 跟着做（Hands-on）

**Step 1 — 四个工具的本地实现**（依赖同前 `pip install "openai>=1.0"`，注意安全写法）

```python
"""Day 8: 多工具 Agent / a multi-tool agent with safe implementations."""

import ast
import json
import operator
from pathlib import Path
from typing import Any

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

# 允许的算术运算符白名单 / whitelist of arithmetic operators
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def calculator(expression: str) -> float:
    """安全计算算术表达式（AST 白名单，杜绝 eval/RCE）/ AST-safe arithmetic.

    Args:
        expression: 仅含 + - * / ** 和数字 / arithmetic only.

    Returns:
        计算结果 / the numeric result. 非法节点抛 ValueError / raises on bad nodes.
    """
    # 时间 O(N) 空间 O(H)，N=节点数 H=树高 / safe recursive eval
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"非法表达式 / illegal expr: {ast.dump(node)}")

    return _eval(ast.parse(expression, mode="eval").body)


def get_stock_price(symbol: str) -> dict[str, Any]:
    """查股价（模拟外部 API）/ fake external stock API."""
    fake = {"AAPL": 228.5, "MSFT": 467.2, "NVDA": 131.8}
    return {"symbol": symbol, "price_usd": fake.get(symbol.upper(), 0.0)}


_DATA_DIR = Path("./data").resolve()  # 安全沙箱目录 / sandbox dir, blocks traversal


def read_file(filename: str) -> str:
    """读取沙箱目录内的文本文件（防路径穿越）/ read a file inside the sandbox."""
    target = (_DATA_DIR / filename).resolve()
    # 注意：裸 str.startswith 前缀匹配是常见漏洞写法，会被同前缀兄弟目录（如 ./data-evil）绕过
    if not target.is_relative_to(_DATA_DIR):  # 语义化判断仍在沙箱内 / still inside?
        raise ValueError("拒绝越权访问 / path traversal blocked")
    return target.read_text(encoding="utf-8")[:2000]  # 截断防爆上下文 / cap length


_USERS = [{"id": 1, "name": "张三", "city": "北京"}, {"id": 2, "name": "李四", "city": "上海"}]


def query_users(city: str) -> list[dict[str, Any]]:
    """按城市查用户（模拟只读数据库）/ fake read-only DB query."""
    return [u for u in _USERS if u["city"] == city]
```

**Step 2 — 工具 schema 清单 + 统一执行循环**

```python
# 小工厂：少写样板，批量生成单参数工具 schema（description 越互斥路由越准）
def tool(name: str, desc: str, arg: str, arg_desc: str) -> dict[str, Any]:
    """构造一个单参数工具的 function schema / build a 1-arg tool schema."""
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "additionalProperties": False,
            "required": [arg],
            "properties": {arg: {"type": "string", "description": arg_desc}}}}}


TOOLS = [
    tool("calculator", "做纯算术计算（加减乘除幂）。涉及数字运算时用。", "expression", "如 (3+4)*2"),
    tool("get_stock_price", "查询某股票代码的当前美元价格。", "symbol", "股票代码，如 AAPL"),
    tool("read_file", "读取本地 data 目录下的文本文件内容。", "filename", "文件名，如 notes.txt"),
    tool("query_users", "按城市查询用户列表（数据库）。", "city", "城市名，如 北京"),
]
IMPL = {"calculator": calculator, "get_stock_price": get_stock_price,
        "read_file": read_file, "query_users": query_users}


def run(question: str) -> str:
    """多工具一轮：模型选工具→执行(可并行)→回填→最终答复。"""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    first = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = first.choices[0].message
    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:  # 可能一次多个=并行调用 / may be parallel calls
        name, args = call.function.name, json.loads(call.function.arguments)
        print(f"  → 模型选了工具 / picked: {name}({args})")  # 观察路由 / watch routing
        try:
            result = IMPL[name](**args)
        except Exception as e:           # 工具失败先兜住，回填给模型（Day 10 细化）
            result = {"error": str(e)}   # feed the error back, don't crash
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps(result, ensure_ascii=False, default=str)})

    second = client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content


if __name__ == "__main__":
    print(run("帮我算一下 (128 + 256) * 3 等于多少"))   # → calculator
    print(run("北京有哪些用户？"))                       # → query_users
    print(run("AAPL 现在多少钱，顺便算下买 10 股要多少"))  # → 多步: 查价 + 算总价
```

重点看那行 `→ 模型选了工具`（见上方注释）：你没写一句 `if 问题包含"算"`——**路由全是模型做的**。

## 3. 今日任务

1. 准备 `./data/notes.txt` 随便写两行，跑 4 类问题各一遍，确认每类都路由正确。
2. **制造路由歧义**：把两个工具的 description 改得很相似，观察模型是否选错——体会"描述互斥性"的作用。
3. **触发并行**：问"北京和上海分别有谁？"，看 `tool_calls` 是否一次返回两个 `query_users` 调用。
4. **验证安全**：诱导模型 `read_file("../../etc/passwd")` 和 `read_file("../data-evil/x.txt")`（同前缀兄弟目录，可先在 `data` 旁边建个 `data-evil/` 试验——裸 `startswith` 前缀匹配就会被它绕过），确认两者都被路径穿越防护拦下、Agent 优雅报错而非崩溃。

**验收标准**：4 类问题各自路由正确；能复现一次并行/多步工具调用；越权读文件被拦截且 Agent 不崩；你能解释"为什么模型选对了工具"。

## 4. 自测清单

- [ ] 我理解 tool routing 由模型决定，依据是工具的 description + schema。
- [ ] 我能写出 AST 白名单的安全 `calculator`，并说清为何不能 `eval()`。
- [ ] 我的 `read_file` 有沙箱校验、能挡路径穿越；见过模型一次返回多个 `tool_calls`（并行）。
- [ ] 我会在执行工具时 `try/except` 把异常兜成结构化错误回填（不让 Agent 崩）。

## 5. 延伸 & 关联

- **路由调优**：工具一多（>10）模型易选错。手段：精炼 description、用 `tool_choice` 强制/禁用某工具、按场景动态裁剪工具清单。今天的 `try/except` 只是雏形，Day 10 做成完整的"错误处理 + 优雅降级"。
- 关联章节：
  - 单工具回顾：[./Day-07-agents-sdk-first-tool.md](./Day-07-agents-sdk-first-tool.md)
  - LangChain 的多工具 Agent（含 SQL 工具箱示例）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - 完整应用串联多组件：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
