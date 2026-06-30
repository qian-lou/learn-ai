# Day 12 · 工具 + 结构化输出结合：端到端强类型可校验

> **今日目标**：把"工具调用"和"结构化输出"拼起来——Agent 用工具拿到原始数据，最终**吐出一个强类型 Pydantic 对象**，全链路类型安全、可校验。
> **时长**：~2h ｜ **前置**：Day 4（结构化输出）、Day 6~10（工具）
> **今日产出**：一个 `day12_typed_agent.py`，输入自然语言 → Agent 查工具 → 返回一个**校验过的业务对象**（可直接入库/给下游 API）。

## 1. 为什么 & 是什么

Day 4 让模型吐强类型对象，但数据是模型"编"的；Day 6+ 让 Agent 用工具拿**真实数据**，但最终答复是**自然语言字符串**。生产里你两个都要：**用真实工具数据 + 最终强类型输出**。这样 Agent 的产物才能**直接喂给下游系统**（写库、调 API、触发流程），而不是让下游再去解析一段话。

给 Java 工程师，这正是你最熟的分层：

| 本日链路 | Java 三层架构对应 |
|---|---|
| 工具（查 DB/API） | DAO / 外部 RPC，拿原始 `DO` |
| ReAct 循环 | Service 层编排（多次取数 + 业务逻辑） |
| 最终 Pydantic 对象 | 组装成 `VO`/`DTO` 返回给 Controller |
| Pydantic 字段校验 | Bean Validation 卡在出口 |

**关键工程点：分两阶段，别混用。** **阶段 A（取数）**用 `tools=[...]` 跑 ReAct 让模型自由调工具收集事实，此阶段**不要**加 `response_format`（会和工具调用打架）；**阶段 B（成型）**循环结束、信息齐了再**发一次**带 `response_format=PydanticModel` 的请求，把事实定型成业务对象。一句话：**先用工具把事实凑齐，再用结构化输出把事实定型**——两步，不是一步。

## 2. 跟着做（Hands-on）

依赖：`pip install "openai>=1.0" "pydantic>=2"`。

**Step 1 — 定义工具（取数）+ 目标对象（定型契约）**

```python
"""Day 12: 工具 + 结构化输出 / tools then typed output, end-to-end."""

import json
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()
MODEL = "gpt-4o-mini"

# --- 工具：查订单 + 查物流（模拟真实数据源）/ data-source tools ---
_ORDERS = {"ORD-1": {"amount": 299.0, "items": 2, "status": "shipped"}}
_LOGISTICS = {"ORD-1": {"carrier": "顺丰", "eta_days": 2}}


def get_order(order_id: str) -> dict[str, Any]:
    """查订单金额、件数、状态 / order amount, item count, status."""
    return _ORDERS.get(order_id, {"error": "订单不存在"})


def get_logistics(order_id: str) -> dict[str, Any]:
    """查物流承运商与预计天数 / carrier and ETA."""
    return _LOGISTICS.get(order_id, {"error": "无物流信息"})


# --- 定型契约：Agent 最终必须吐出的强类型业务对象 / the output contract ---
class OrderReport(BaseModel):
    """一份订单概览，可直接入库/给下游 / a typed order summary."""

    order_id: str
    amount: float = Field(ge=0, description="订单金额 / total amount")
    status: Literal["shipped", "pending", "unknown"]  # 受限取值 / enum
    carrier: str = Field(description="承运商，未知填 未知 / carrier")
    eta_days: int = Field(ge=0, description="预计天数 / ETA in days")
    note: str = Field(description="一句话备注 / a one-line note")
```

**Step 2 — 阶段 A 跑工具循环，阶段 B 定型**

```python
# 两个工具都只收 order_id，用小工厂少写样板 / both take order_id; a tiny factory
def order_tool(name: str, desc: str) -> dict[str, Any]:
    """构造一个收 order_id 的工具 schema / a schema for an order_id tool."""
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "required": ["order_id"],
            "additionalProperties": False,
            "properties": {"order_id": {"type": "string"}}}}}


TOOLS = [order_tool("get_order", "查询订单的金额、件数、状态。"),
         order_tool("get_logistics", "查询订单的物流承运商与预计到达天数。")]
IMPL = {"get_order": get_order, "get_logistics": get_logistics}
MAX_STEPS = 6


def build_report(question: str) -> OrderReport:
    """端到端：工具取数(阶段A) → 结构化定型(阶段B)，返回校验过的对象。

    Args:
        question: 自然语言请求 / a natural-language request.

    Returns:
        校验过的 OrderReport；拒答抛 ValueError / a validated object, raises on refusal.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "你是订单助手。先用工具查清订单与物流，信息齐了再作答。"},
        {"role": "user", "content": question},
    ]

    # ---- 阶段 A：自由调工具收集事实（不加 response_format）----
    # phase A: gather facts via tools; do NOT set response_format here
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:        # 模型认为信息够了 / facts gathered
            messages.append(msg)
            break
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            obs = IMPL[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(obs, ensure_ascii=False)})

    # ---- 阶段 B：把对话里的事实定型为强类型对象（加 response_format）----
    # phase B: shape the gathered facts into a typed object
    messages.append({"role": "user",
                     "content": "把上面查到的信息整理成结构化订单概览。"})
    final = client.beta.chat.completions.parse(
        model=MODEL, messages=messages, response_format=OrderReport,
    )
    out = final.choices[0].message
    if out.refusal:
        raise ValueError(f"模型拒答 / refused: {out.refusal}")
    return out.parsed  # 直接就是校验过的 OrderReport / a validated object


if __name__ == "__main__":
    report = build_report("帮我看下订单 ORD-1 的整体情况")
    # 强类型：可直接 .属性 取用、可入库、可给下游 API / typed & ready for downstream
    print(report.model_dump_json(indent=2, ))
    print("到货天数(int):", report.eta_days, type(report.eta_days).__name__)
```

跑 `python day12_typed_agent.py`：阶段 A 模型会先后调 `get_order` 和 `get_logistics` 拿到真实数据，阶段 B 把这些事实**定型**成一个 `OrderReport`——`eta_days` 是真正的 `int`、`status` 只能是受限枚举值、`amount` 越界会被 Pydantic 拦。这个对象可以**直接 `model_dump()` 写库或 POST 给下游**，下游再也不用解析自然语言。

## 3. 今日任务

1. 跑通 `build_report`，确认输出是一个 JSON 化的 `OrderReport`，且字段类型正确（`eta_days` 是 int 不是 str）。
2. **验证全链路校验**：把 `OrderReport.amount` 约束改成 `ge=10000`，跑一个金额不够的订单，观察 Pydantic 在**定型阶段**拦下脏数据。
3. **测缺数据路径**：查一个不存在的订单（如 `ORD-999`），看 Agent 如何把 `{"error":...}` 体面地落到 `status="unknown"` / `note` 里——而不是编造数据。
4. **思考为何分两步**：试着在阶段 A 的 `create` 里同时加 `tools` 和 `response_format`，观察/记录冲突或行为异常，理解"取数与定型必须分离"。

**验收标准**：能稳定拿到强类型 `OrderReport`；改约束后定型阶段能拦下越界数据；缺数据时不编造而是落到"unknown"；你能讲清阶段 A（取数）与阶段 B（定型）为何要分开。

## 4. 自测清单

- [ ] 我理解"工具取真实数据 + 结构化定型最终对象"的端到端价值。
- [ ] 我知道取数阶段用 `tools`、定型阶段才用 `response_format`，两者分开。
- [ ] 我能把这条链路类比成 Java 的 DAO→Service→VO + 出口校验。
- [ ] 我会用 `parse(response_format=Model)` 把对话事实定型成校验过的对象。
- [ ] 我的 Agent 在缺数据时不编造，而是落到受限的 "unknown" 取值。

## 5. 延伸 & 关联

- 这个"强类型产物"正是 Day 15 阶段项目的核心交付物——数据查询 Agent 必须返回结构化、可校验的结果，而非一段话。
- 进阶：可给 `OrderReport` 加 `@field_validator` 做跨字段一致性校验（如 status=shipped 时 carrier 不能为空），等价于 Java 的自定义 `@AssertTrue`。
- 关联章节：
  - 结构化输出原理：[../agent-course/Day-04-structured-output.md](./Day-04-structured-output.md)
  - 工具错误处理（缺数据/失败如何落到对象里）：[../agent-course/Day-10-error-handling.md](./Day-10-error-handling.md)
  - 评估与监控（强类型输出便于自动评测）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
