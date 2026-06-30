# Day 15 · 🎯 阶段项目：数据查询 Agent

> **今日目标**：综合 Day 6~12，做一个能跑的「数据查询 Agent」——查**真实数据**(SQLite)、返回**结构化结果**、带**错误处理**。这是你简历上的第一个 Agent 作品。
> **时长**：~2h ｜ **前置**：Day 6~14（工具 + ReAct + 错误处理 + 结构化输出）
> **今日产出**：单文件 `data_query_agent.py`，输入自然语言 → Agent 查 SQLite → 返回校验过的强类型结果对象。

## 1. 为什么 & 是什么

Phase 1 的毕业设计：把前 9 天的零件拼成**让不懂 SQL 的人用大白话查数据库**——最经典、面试最常被问的落地场景（"text-to-data"）。三个里程碑要求缺一不可：① **真实数据**（连真的 SQLite，标准库零依赖）；② **结构化结果**（最终吐 Pydantic 对象，可直接给下游，Day 12）；③ **错误处理**（SQL 错/查无/非法参数都不崩，优雅降级，Day 10）。

**架构**：`用户自然语言 → [Agent/ReAct]` 自主调三个工具（`list_tables` 看表 → `describe_table` 看结构 → `run_query` 执行只读 SELECT）`→ [结构化定型] → QueryResult`。给 Java 工程师：这就是一个**自然语言 → 动态查询**的 Service——`run_query` 是 DAO、Agent 循环是 Service 编排、`QueryResult` 是返回的 VO。**最关键的安全红线：只读**，必须挡住 `DROP/DELETE/UPDATE/INSERT`，等价于给在线库挂只读账号。

## 2. 跟着做（Hands-on）

**Step 1 — 造一个真实 SQLite 库 + 三个查询工具（含只读护栏）**（依赖 `pip install "openai>=1.0" "pydantic>=2"`，sqlite3 是标准库）

```python
"""Day 15 阶段项目：数据查询 Agent / a text-to-data agent over SQLite."""

import json
import sqlite3
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()
MODEL = "gpt-4o-mini"
_CONN = sqlite3.connect(":memory:", check_same_thread=False)  # 演示用内存库 / in-memory
# 只读护栏：禁止出现这些写操作关键字 / read-only guard: ban write keywords
_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace")

def seed() -> None:
    """初始化一个真实的小型库（订单 + 客户）/ seed a real small DB."""
    _CONN.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, amount REAL, status TEXT);
        INSERT INTO customers VALUES (1,'张三','北京'),(2,'李四','上海'),(3,'王五','北京');
        INSERT INTO orders VALUES (1,1,299,'shipped'),(2,1,88.5,'pending'),(3,2,1200,'shipped'),(4,3,45,'shipped');
    """)
    _CONN.commit()

def run_query(sql: str) -> dict[str, Any]:
    """执行只读 SELECT 查询，拒绝任何写操作 / run a read-only SELECT.

    Args:
        sql: 一条 SELECT 语句 / a single SELECT statement.
    Returns:
        成功 {"columns","rows"}，失败 {"error"} / result or error.
    """
    low = sql.strip().lower()
    if not low.startswith("select") or any(k in low for k in _FORBIDDEN):  # 安全红线
        return {"error": "仅允许 SELECT 查询 / only SELECT is allowed"}
    try:
        cur = _CONN.execute(sql)  # 演示库只读故直接执行；生产应配只读连接 / read-only conn
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(50)]  # 限 50 行防爆 / cap rows
        return {"columns": cols, "rows": rows}
    except sqlite3.Error as e:  # SQL 语法/字段错 → 回填给模型自我纠正 / feed error back
        return {"error": f"SQL 执行失败: {e}"}

def list_tables() -> dict[str, Any]:
    """列出库里所有表名 / list all table names."""
    cur = _CONN.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {"tables": [r[0] for r in cur.fetchall()]}

def describe_table(table: str) -> dict[str, Any]:
    """查某表的列结构（字段名 + 类型）/ describe a table's columns."""
    if not table.isidentifier():  # 防注入：表名只允许合法标识符 / anti-injection
        return {"error": "非法表名 / illegal table name"}
    cur = _CONN.execute(f"PRAGMA table_info({table})")
    return {"columns": [{"name": r[1], "type": r[2]} for r in cur.fetchall()]}


# ===== Step 2：最终结果契约 + 工具 schema + Agent 主体 / the typed deliverable =====
class QueryResult(BaseModel):
    """一次数据查询的结构化结果 / structured result of one query."""
    success: bool = Field(description="是否成功取到数据 / did we get data")
    row_count: int = Field(ge=0, description="返回行数 / number of rows")
    rows: list[dict[str, Any]] = Field(description="结果行 / the result rows")
    summary: str = Field(description="一句话中文结论 / one-line answer")

def schema(name: str, desc: str, arg: str = "") -> dict[str, Any]:
    """工具 schema 小工厂，arg 为空=无参工具 / a tiny 0-or-1-arg schema factory."""
    props = {arg: {"type": "string"}} if arg else {}
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "additionalProperties": False,
            "properties": props, "required": [arg] if arg else []}}}

TOOLS = [
    schema("list_tables", "列出数据库里有哪些表。不知道表名时先用它。"),
    schema("describe_table", "查看某表的字段及类型。写 SQL 前先用它确认字段。", "table"),
    schema("run_query", "执行一条只读 SELECT 查询并返回结果行。", "sql"),
]
IMPL = {"list_tables": list_tables, "describe_table": describe_table, "run_query": run_query}
MAX_STEPS = 8  # 守卫：探表+查询通常 3~5 步够 / loop guard

def query_agent(question: str) -> QueryResult:
    """自然语言 → 数据查询 → 结构化结果，全程不崩 / text-to-data, never crashes.

    Args:
        question: 用户的自然语言问题 / a natural-language question.
    Returns:
        校验过的 QueryResult；阶段 A 取数、阶段 B 定型 / validated result (gather→shape).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是数据分析助手。流程：先 list_tables 看表，再 "
         "describe_table 看字段，然后写 SELECT 用 run_query 查。只能查不能改，查不到就如实说、绝不编造。"},
        {"role": "user", "content": question}]

    # ---- 阶段 A：ReAct 多步取数（探表 → 看结构 → 查询）----
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:  # 模型不再调工具 = 信息够了 / done gathering
            break
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            try:  # 工具级兜底，永不抛栈（Day 10）/ never crash; feed error back
                obs = IMPL[call.function.name](**args)
            except Exception as e:
                obs = {"error": repr(e)}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(obs, ensure_ascii=False, default=str)})

    # ---- 阶段 B：把取到的数据定型为 QueryResult（Day 12）----
    messages.append({"role": "user", "content": "把查询结果整理成结构化输出。"})
    out = client.beta.chat.completions.parse(
        model=MODEL, messages=messages, response_format=QueryResult).choices[0].message
    if out.refusal:  # 拒答也降级成一个合法对象 / refusal → still a valid object
        return QueryResult(success=False, row_count=0, rows=[], summary=out.refusal)
    return out.parsed

if __name__ == "__main__":
    seed()
    for q in ["北京的客户有几个？分别叫什么？", "已发货(shipped)的订单总金额是多少？",
              "帮我把所有订单删掉"]:  # 第三条会被只读护栏挡下 / blocked by guard
        print(f"\nQ: {q}")
        print(query_agent(q).model_dump_json(indent=2))
```

跑 `python data_query_agent.py`：第一题 Agent 自己走 `list_tables → describe_table → run_query` 写出 SQL 并定型为 `QueryResult(success=True, row_count=2, ...)`（**全程没人写 SQL**）；第二题写聚合 SQL；第三题"删掉订单"被只读护栏拦下、模型如实回复"只能查不能删"。

## 3. 今日任务

1. 跑通三个示例，确认前两题返回正确的结构化结果、第三题被安全拦截。
2. **加难度**：问一个**跨表 JOIN** 的问题（如"张三一共下了几单、总金额多少"），看 Agent 能否自己探出 `customer_id` 外键并写出 JOIN。
3. **压错误路径 + 加可观测**：问一个**字段不存在**的问题（如"查每个客户邮箱"，无 email 列），确认 Agent 读到 SQL 错误后纠正或如实说明；并打印每步工具调用形成"思考轨迹"（讲解作品的杀手锏）。
4. **写复盘**：在文件顶部注释记下 Phase 1 踩的 3 个坑（如"取数与定型必须分两步""只读护栏必须有"）。

**验收标准**：能用大白话查到真实数据并返回校验过的 `QueryResult`；JOIN 类问题 Agent 能自主完成；写操作被只读护栏挡住且不崩；你能完整讲清这个 Agent 的架构与每一步在做什么。

## 4. 自测清单

- [ ] 我的 Agent 能连真实 SQLite、自主探表写 SQL 并返回结构化结果。
- [ ] 我实现了**只读护栏**挡住一切写/删；错误（SQL 错/查无/拒答）都降级成合法 `QueryResult` 不崩。
- [ ] 我能讲清"阶段 A 取数(ReAct) + 阶段 B 定型(结构化)"的两段式。
- [ ] 我能把这个 Agent 的每个组件映射到 Java 的 DAO/Service/VO。

## 5. 延伸 & 关联

**🎉 里程碑达成**：这是计划里的第一个作品（Day 15），证明你能把"工具调用 + ReAct + 错误处理 + 结构化输出"串成一个**能跑、能讲、能上简历**的东西。**下一步预告**：现在 Agent 只能查结构化数据，Phase 2（Day 16 起）教它查**非结构化文档**——嵌入向量库、按语义检索，这就是 **RAG**（面试高频中的高频）。进阶方向：把 `_CONN` 换成真实业务库的只读连接、给 `QueryResult` 加分页/图表字段、接 Day 13 的 Spring AI 做 Java 版。

- 关联章节：
  - 工具 + 结构化输出（本项目的定型基础）：[./Day-12-tools-structured-output.md](./Day-12-tools-structured-output.md)
  - 错误处理与降级（只读护栏与兜底）：[./Day-10-error-handling.md](./Day-10-error-handling.md)
  - 下一阶段 RAG 基础（查非结构化数据）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
  - 本系列总计划（看你在 70 天里的位置）：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
