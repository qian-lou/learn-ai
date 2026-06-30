# Day 4 · 结构化输出：Pydantic + parse 拿强类型对象

> **今日目标**：用 Pydantic + `client.beta.chat.completions.parse` 稳定拿到强类型对象，而不是手解析字符串。
> **时长**：~2h ｜ **前置**：Day 1~3
> **今日产出**：一个把自由文本解析成 `Pydantic` 模型实例的脚本，字段类型安全、可直接 `.属性` 取用。

## 1. 为什么 & 是什么

Demo 阶段，模型回个字符串就行。但**进了系统**，下游代码要的是**结构化、强类型**的数据——要遍历、要入库、要校验。让模型"尽量返回 JSON"然后自己 `json.loads` + 各种 `if` 兜底，既脆弱又啰嗦。

给 Java 工程师，这件事你太熟了：

| LLM 结构化输出 | Java 对应 | 说明 |
|---|---|---|
| Pydantic `BaseModel` | DTO / POJO | 定义字段、类型、默认值 |
| `parse(response_format=Model)` | Jackson 反序列化到 POJO | 直接把模型输出变成对象，而非裸 JSON 串 |
| Pydantic 字段约束/校验 | Bean Validation（`@NotNull`/`@Min`） | 类型不对/缺字段直接报错，**把错误挡在边界** |

**企业为什么"必须"强类型**：

- **可靠**：契约由 schema 约束，模型不能想给啥给啥；下游不必为"万一格式变了"写防御代码。
- **可维护**：字段即文档，IDE 有补全和类型检查。
- **可校验**：解析阶段就能拦下脏数据（如评分必须 1~5），等价于 Java 边界上的参数校验。
- **可演进**：加字段就是改 model，调用处类型立刻报错提醒你改全——**编译期/解析期暴露问题，而不是线上炸**。

**2026 现代做法**：OpenAI SDK 的 `client.beta.chat.completions.parse(...)`，传入一个 Pydantic 模型作为 `response_format`，SDK 会让模型按该 schema 输出，并**直接反序列化成你的模型实例**（`.message.parsed` 就是对象）。这叫 **Structured Outputs**，底层靠约束解码保证字段齐全、类型正确。

## 2. 跟着做（Hands-on）

```bash
pip install "openai>=1.0" "pydantic>=2"
```

```python
"""Day 4: 结构化输出 / structured outputs with Pydantic."""

from typing import List, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


# --- 定义“契约”：一个商品评论的解析结果 / the parsing contract ---
class ActionItem(BaseModel):
    """一条待办 / a single follow-up action."""

    owner: str = Field(description="负责人 / who should handle it")
    task: str = Field(description="要做的事 / what to do")


class ReviewAnalysis(BaseModel):
    """对一段用户评论的结构化分析 / structured analysis of a review."""

    sentiment: Literal["positive", "neutral", "negative"]  # 受限取值 / constrained enum
    score: int = Field(ge=1, le=5, description="1~5 星 / star rating")  # 校验 1..5
    summary: str = Field(description="一句话摘要 / one-line summary")
    actions: List[ActionItem]  # 嵌套结构 / nested structure


def analyze_review(text: str) -> ReviewAnalysis:
    """把一段评论解析为强类型对象 / parse a review into a typed object.

    Args:
        text: 原始评论文本 / raw review text.

    Returns:
        ReviewAnalysis 实例，字段已校验 / a validated model instance.

    Raises:
        ValueError: 模型拒答或解析失败时 / if the model refuses or parsing fails.
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是客服分析助手，严格按给定结构输出。"},
            {"role": "user", "content": text},
        ],
        response_format=ReviewAnalysis,  # 传入 Pydantic 模型作为契约 / pass the model as schema
    )

    message = completion.choices[0].message
    # 安全检查：模型可能因安全策略拒答 / the model may refuse
    if message.refusal:
        raise ValueError(f"模型拒答 / refused: {message.refusal}")

    # .parsed 直接就是 ReviewAnalysis 实例，无需手动 json.loads
    # .parsed is already a ReviewAnalysis instance — no manual json.loads
    return message.parsed


if __name__ == "__main__":
    sample = "买了三天就掉漆，客服还爱答不理，强烈不推荐。但物流挺快。"
    result: ReviewAnalysis = analyze_review(sample)

    # 直接点属性取用，IDE 有补全、类型安全 / dot-access, typed & autocompleted
    print("情感 / sentiment :", result.sentiment)
    print("评分 / score     :", result.score)
    print("摘要 / summary   :", result.summary)
    for a in result.actions:
        print(f"  - [{a.owner}] {a.task}")
```

运行后，`result` 是一个**真正的对象**：`result.score` 是 `int`，越界会在校验时报错；`result.actions` 是 `list`，可直接遍历。

## 对比：旧的 `json_object` 模式（了解即可，别再新写）

```python
# 旧方式：只能保证“是合法 JSON”，但 **不保证字段和类型**，还得手动校验
# legacy: guarantees valid JSON only — NOT the fields/types; you must validate by hand
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "...必须返回 JSON..."}],
    response_format={"type": "json_object"},
)
import json
data = json.loads(resp.choices[0].message.content)  # 拿到的是 dict，字段全靠自觉
# 还要自己写：data.get("score") 在不在？是不是 int？范围对不对？……
```

**结论**：能用 `parse(response_format=PydanticModel)` 就别退回 `json_object`。前者把"格式 + 字段 + 类型 + 校验"一次性交给 schema，等价于 Java 里"直接反序列化进带校验注解的 DTO"。

## 3. 今日任务

1. 跑通 `analyze_review`，确认拿到的是对象、能 `.score` 直接取值。
2. **加字段**：给 `ReviewAnalysis` 增一个 `topics: List[str]`（评论涉及的话题），重跑，体会"改 schema 即改契约"。
3. **触发校验**：把某条输入构造得很极端，或临时把 `score` 约束改成 `ge=10`，观察 Pydantic 的校验行为。
4. **对照旧法**：用 `json_object` 模式实现同一任务，数一数你为"兜底"多写了几行——这就是强类型省下的成本。

**验收标准**：能稳定拿到 `ReviewAnalysis` 实例并点属性取值；新增字段后输出随之带上该字段；能讲清 `parse` 相比 `json_object` 好在哪。

## 4. 自测清单

- [ ] 我能把 Pydantic 模型类比成 Java 的 DTO + Bean Validation。
- [ ] 我会用 `client.beta.chat.completions.parse(response_format=Model)` 拿对象。
- [ ] 我知道用 `message.parsed` 取结果，并会先判 `message.refusal`。
- [ ] 我能说出企业为何"必须"强类型输出（可靠/可维护/可校验/可演进）。
- [ ] 我理解 `json_object` 与 Structured Outputs 的本质差距，且优先用后者。

## 5. 延伸 & 关联

- 结构化输出是后面**工具调用（function calling）**的同源能力——工具的"参数 schema"也是用同样方式约束的（Day 6 起会大量用到）。
- Anthropic 侧：用 tool / `response_format` 或在系统提示中给 JSON schema 约束，配合 SDK 解析，思路一致。
- 关联章节：
  - 提示工程进阶（含输出约束）：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
  - 评估与监控（结构化输出便于自动评测）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
