# Day 10 · 工具错误处理：让 Agent 优雅降级，而不是崩

> **今日目标**：把工具的异常、非法参数、超时都"喂回"给模型，让它**自我修复或优雅降级**，而不是让整个 Agent 抛栈崩掉。
> **时长**：~2h ｜ **前置**：Day 6~9
> **今日产出**：一个 `day10_robust.py`，工具会失败（异常/超时/坏参数），但 Agent 能感知错误、重试或换路、最终给用户一个体面的回应。

## 1. 为什么 & 是什么

真实工具一定会失败：外部 API 超时、数据库连不上、模型传了非法参数。**新手 Agent 一遇异常就整段崩**——这在生产里不可接受。成熟 Agent 的标志是：**把错误当成一种 Observation 喂回模型**，让它有机会改参数重试、换个工具、或诚实地告诉用户"这步失败了"。核心心法一句话：**工具的失败不是程序的失败，而是给模型的一条新信息。**

给 Java 工程师，这套你太熟了，只是落点不同：

| Agent 错误处理 | Java 世界对应 | 关键差异 |
|---|---|---|
| `try/except` 包住工具调用 | Service 层 `try/catch` | catch 后**不抛给上层，而是回填给模型** |
| 超时 `timeout` | HTTP/事务超时 | 必设，否则一个慢工具拖死整轮 |
| 重试 + 退避 | Spring Retry / Resilience4j | 但**别让模型无限重试**，要计数封顶 |

**三道防线**（从里到外）：① **工具内**：捕获异常 → 返回**结构化错误**（含可读 message），而非 `raise` 到天上；② **循环里**：给错误计数，同一工具连错 N 次就强制收尾，防"重试死循环"；③ **对模型**：错误信息要**对模型友好**，写清"为什么失败、该怎么改"，模型才可能自己修。

## 2. 跟着做（Hands-on）

**Step 1 — 会"受控失败"的工具 + 安全执行包装**（依赖 `pip install "openai>=1.0"`）

```python
"""Day 10: 工具错误处理与优雅降级 / robust tool execution & graceful fallback."""

import json
import random
from typing import Any

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"


def get_order(order_id: str) -> dict[str, Any]:
    """查订单（模拟：偶发超时 + 非法 id 校验）/ flaky order lookup.

    Args:
        order_id: 形如 ORD-123 的订单号 / order id like ORD-123.

    Returns:
        订单信息；非法 id 抛 ValueError、超时抛 TimeoutError / raises on bad id / timeout.
    """
    # 参数校验：把非法输入挡在边界，给模型可读的纠错信息 / actionable error
    if not order_id.startswith("ORD-"):
        raise ValueError("订单号必须以 ORD- 开头，例如 ORD-123")
    if random.random() < 0.4:  # 40% 概率模拟超时 / simulate flakiness
        raise TimeoutError("订单服务响应超时")
    return {"order_id": order_id, "status": "已发货", "eta_days": 2}


def safe_execute(name: str, impl_map: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """统一执行工具：永不抛栈，失败转成结构化错误回填给模型。

    Args:
        name: 工具名 / tool name. impl_map: 工具名→实现的注册表 / tool registry. args: 参数 / kwargs.

    Returns:
        成功为结果 dict，失败为 {"error", "hint"} / result or error.
    """
    # 工具名查找也纳入防线：模型可能幻觉出未注册的工具名 / guard against hallucinated names
    impl = impl_map.get(name)
    if impl is None:
        return {"ok": False, "error": f"未知工具：{name}", "retryable": False,
                "hint": "该工具不存在，请只用已提供的工具 / no such tool, use provided ones only"}
    try:
        return {"ok": True, "data": impl(**args)}
    except ValueError as e:        # 可纠正：参数错 → 提示模型改参数重试
        return {"ok": False, "error": str(e), "retryable": True,
                "hint": "请修正参数后重试 / fix the argument and retry"}
    except TimeoutError as e:      # 暂时性：超时 → 提示模型可重试一次
        return {"ok": False, "error": str(e), "retryable": True,
                "hint": "上游超时，可重试一次 / transient, retry once"}
    except Exception as e:         # 兜底：未知错 → 不可重试，建议降级
        return {"ok": False, "error": repr(e), "retryable": False,
                "hint": "不可恢复，请向用户说明 / unrecoverable, tell the user"}
```

**Step 2 — 带"错误计数 + 重试封顶"的循环**

```python
TOOLS = [{"type": "function", "function": {"name": "get_order",
    "description": "按订单号查询订单状态，订单号形如 ORD-123。",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string", "description": "订单号，必须 ORD- 开头"}},
        "required": ["order_id"], "additionalProperties": False}}}]
IMPL = {"get_order": get_order}

MAX_STEPS = 6
MAX_TOOL_ERRORS = 3  # 重试封顶：累计错 3 次就强制收尾 / cap retries to avoid loops


def run(question: str) -> str:
    """带错误处理与优雅降级的 Agent 循环。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content":
         "你是客服助手。工具失败时：可纠正就改参数重试，超时可重试一次，"
         "若多次失败就如实向用户说明并给出已知信息，不要假装成功。"},
        {"role": "user", "content": question},
    ]
    error_count = 0

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = safe_execute(call.function.name, IMPL, args)
            if not result["ok"]:
                error_count += 1
                print(f"  [错误 {error_count}] {result['error']}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

        # 第 2 道防线：错误太多 → 注入"停止重试，请降级作答"的指令
        # too many errors → force graceful degradation instead of looping
        if error_count >= MAX_TOOL_ERRORS:
            messages.append({"role": "system",
                "content": "工具已多次失败，请停止重试，向用户如实说明并给出能给的信息。"})
            final = client.chat.completions.create(model=MODEL, messages=messages)
            return final.choices[0].message.content

    return "（系统繁忙，请稍后再试 / service busy, please retry later）"


if __name__ == "__main__":
    print(run("帮我查下订单 ORD-888 到哪了？"))      # 可能超时→模型重试→成功
    print(run("查下订单 888 的状态"))                # 缺 ORD- → 模型改参数重试
```

跑几次（因为有随机超时，多跑几遍能看到不同路径）：第一题里若工具超时，模型会读到 `retryable: True` 然后**自己再调一次**；第二题里模型读到"必须 ORD- 开头"，会**自动把参数改成 `ORD-888`** 重试。全程**没崩**——错误被转化成了模型的决策输入。

## 3. 今日任务

1. 跑通并多跑几遍 `run("...ORD-888...")`，观察"超时→模型自动重试→成功"的路径出现。
2. **验证自我纠错**：跑 `run("查下订单 888")`，确认模型读到错误提示后把参数改对重试。
3. **逼出降级**：把 `get_order` 的超时概率改成 `1.0`（必超时），确认错误累计到 `MAX_TOOL_ERRORS` 后 Agent **如实告知失败**，而不是死循环或假装成功。
4. **对比"裸奔"**：临时去掉 `safe_execute` 直接调工具，跑一次必超时的场景，亲眼看到整段抛栈崩掉——对比出 `safe_execute` 的价值。

**验收标准**：超时能被模型自动重试且最终不崩；非法参数能被模型自我纠正；必失败时 Agent 优雅降级（如实说明 + 不死循环）；你能讲清"三道防线"各拦住了什么。

## 4. 自测清单

- [ ] 我理解"工具失败 = 给模型的一条 Observation"，而非程序崩溃。
- [ ] 我会把异常转成结构化错误（含 `retryable` / `hint`）回填，让模型可自我修复。
- [ ] 我在循环里做了**错误计数 + 重试封顶**，防"重试死循环"。
- [ ] 我知道超时必须设上限，否则一个慢工具拖死整轮。
- [ ] 我能让 Agent 在多次失败后**优雅降级**（如实说明）而非假装成功。

## 5. 延伸 & 关联

- **错误信息要"对模型友好"**：`"ValueError"` 没用，`"订单号必须 ORD- 开头，例如 ORD-123"` 才能让模型自己改对——把错误当成写给模型看的提示词。真实外部调用还要叠加 HTTP 级 `timeout`、指数退避重试、熔断（Java 侧用 Resilience4j，Python 侧用 `tenacity`）。
- 关联章节：
  - ReAct 循环（错误重试就发生在循环里）：[./Day-09-react-loop.md](./Day-09-react-loop.md)
  - 评估与监控（线上要统计工具失败率/重试率）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - API 服务化（部署时的超时/降级/健康检查）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
