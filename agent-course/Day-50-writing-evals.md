# Day 50 · 写 eval：准确率、幻觉率、回归测试与 LLM-as-judge

> **今日目标**：给昨天的规则 eval 加上**幻觉率**和 **LLM-as-judge**，并把整套接成**回归测试**——改一处不再担心崩别处。
> **时长**：~2h ｜ **前置**：Day 49（eval 入门、测试集）、Day 47（tracing 可选）
> **今日产出**：一个 `evals.py`，对一个 RAG/问答 Agent 同时跑「精确匹配 + 幻觉检测 + LLM 打分」，输出多维分数，并能作为回归套件重复跑。

## 1. 为什么 & 是什么

昨天的精确匹配，只能评"答案=标准答案"这种封闭题。但真实 Agent 大量是**开放式**的——问答、摘要、解释。这类输出**措辞千变万化，没法 `==`**。今天补三件事：

1. **幻觉率（hallucination rate）**：答案有没有"编造"——尤其 RAG 场景，回答必须**有据可依**，不能瞎说。这是 LLM 最致命的质量问题。
2. **LLM-as-judge**：让**另一个模型**当裁判，按 rubric（评分标准）给开放式回答打分。这是把"人工评审"自动化。
3. **回归测试**：把多维 eval 接进流程，每次改动**自动重跑**，分数掉了就拦住。

给 Java 工程师的对照：

| 今日概念 | Java / 测试世界 | 说明 |
|---|---|---|
| LLM-as-judge | 复杂断言 / 自定义 Matcher | 没法 `==` 时，写个"判定器"来裁决 |
| 幻觉率 | 契约校验 / 一致性断言 | 断言"输出只能来自给定上下文" |
| 多维评分 | 多个测试维度 + 覆盖率 | 准确率、幻觉率、有用性分开看 |
| 回归套件接 CI | CI 里的回归测试 | 每次提交自动跑，红了就 block |

**LLM-as-judge 的关键纪律**（否则裁判本身就不可信）：

- **给明确 rubric**：别问"好不好"，要问"是否事实正确 / 是否只用了给定上下文 / 1~5 分各代表什么"。
- **让它先说理由再打分**（CoT），并**输出结构化分数**（Day 4 的强类型），便于聚合。
- **裁判可以用更强的模型**：被测用便宜模型，裁判用强模型，性价比高。
- **裁判也要被校准**：抽样人工复核裁判的判罚，别让"裁判幻觉"污染指标。

## 2. 跟着做（Hands-on）

### 方式一：纯手写多维 eval（不依赖 eval 框架，看清原理）

```bash
pip install "openai>=1.0" "pydantic>=2"
```

```python
"""Day 50: 多维 eval —— 幻觉检测 + LLM-as-judge / multi-dim evals."""

from typing import List, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()
SUT_MODEL = "gpt-4o-mini"     # 被测系统用便宜模型 / system under test (cheap)
JUDGE_MODEL = "gpt-4o"        # 裁判用强模型 / judge uses a stronger model


# ---- 被测：一个基于给定上下文作答的 RAG 问答 / a context-grounded QA ----
def answer_with_context(question: str, context: str) -> str:
    """只依据 context 回答，不知道就说不知道 / answer ONLY from context."""
    resp = client.chat.completions.create(
        model=SUT_MODEL,
        messages=[
            {"role": "system", "content": "只根据【资料】回答；资料没有就回答「资料中未提及」。"},
            {"role": "user", "content": f"【资料】{context}\n【问题】{question}"},
        ],
    )
    return resp.choices[0].message.content


# ---- 裁判输出契约：先讲理由再给分，结构化 / judge contract: reason then score ----
class Judgement(BaseModel):
    """LLM 裁判的一次判罚 / one verdict from the LLM judge."""

    reasoning: str = Field(description="判定理由，先想后打分 / reason before scoring")
    grounded: bool = Field(description="回答是否完全有据于资料（无幻觉）/ fully grounded?")
    helpfulness: int = Field(ge=1, le=5, description="有用性 1~5 / helpfulness")


def judge(question: str, context: str, answer: str) -> Judgement:
    """LLM-as-judge：按 rubric 评一条开放式回答 / grade an open-ended answer."""
    # rubric 要具体：明确 grounded(有据=无幻觉) 与 helpfulness 各自含义，且先想后判
    rubric = (
        "你是严格的评审。判定标准：\n"
        "1) grounded：回答的每个事实是否都能在【资料】中找到依据，编造则为 false（幻觉）。\n"
        "2) helpfulness：是否切题、清晰、完整（1~5）。先在 reasoning 里逐条分析再判定。"
    )
    completion = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"【资料】{context}\n【问题】{question}\n【回答】{answer}"},
        ],
        response_format=Judgement,
    )
    return completion.choices[0].message.parsed


# ---- 回归套件：对测试集跑多维评估并聚合 / regression suite ----
def run_suite(cases: List[dict]) -> None:
    """对测试集跑批，输出幻觉率 + 平均有用性 / run multi-dim evals over a dataset."""
    hallucinated = 0
    help_scores: List[int] = []
    for c in cases:
        ans = answer_with_context(c["question"], c["context"])
        v = judge(c["question"], c["context"], ans)
        if not v.grounded:
            hallucinated += 1
        help_scores.append(v.helpfulness)
        mark = "✅" if v.grounded else "🔴幻觉"
        print(f"{mark} help={v.helpfulness} | Q: {c['question']}")

    n = len(cases)
    print(f"\n幻觉率 / hallucination rate: {hallucinated}/{n} = {hallucinated / n:.0%}")
    print(f"平均有用性 / avg helpfulness: {sum(help_scores) / n:.2f} / 5")


if __name__ == "__main__":
    dataset = [
        {   # 资料里有答案——应 grounded
            "question": "这款电池容量多大？",
            "context": "X1 手机配备 5000mAh 电池，支持 67W 快充。",
        },
        {   # 资料里【没有】价格——若模型敢编价格就是幻觉
            "question": "这款手机多少钱？",
            "context": "X1 手机配备 5000mAh 电池，支持 67W 快充。",
        },
    ]
    run_suite(dataset)
```

第 2 条是**幻觉陷阱**：资料里没价格，好系统应回"资料中未提及"，裁判给 `grounded=True`；若模型瞎编一个价格，裁判应判 `grounded=False`，幻觉率随之上升。

### 方式二：用 LangSmith 跑托管 eval（生产做法，要点）

手写帮你理解原理；生产里常用 LangSmith 托管。要点（`pip install -U langsmith openevals`）：

- 数据集放在 LangSmith；用 `ls.evaluate(target, data="...", evaluators=[...], max_concurrency=4)` 跑批（异步用 `aevaluate`）。
- **2026 评估器签名**是关键字注入：`def my_eval(inputs, outputs, reference_outputs) -> bool|dict`（不再是旧的 `run`/`example` 位置参数）。
- LLM 裁判用官方 `openevals`：`from openevals.llm import create_llm_as_judge` + 预置 `CORRECTNESS_PROMPT`，把它当作一个 evaluator 传进 `evaluate` 即可，结果与逐条分数都在 Web 控制台聚合。

> **回归测试怎么用**：把上面任一套接进 CI（Day 56 起）。每次提交跑全集，**幻觉率上升或有用性下降就让流水线失败**——这就是"改一处不再担心崩别处"的底气。

## 3. 今日任务

1. 跑通方式一，确认能输出**幻觉率 + 平均有用性**，且幻觉陷阱样本被正确判罚。
2. **校准裁判**：人工复核 2~3 条裁判结果，看它判得对不对；若裁判误判，改进 rubric（更具体、给正反例）。
3. **做回归**：故意把被测的 system prompt 改差（如删掉"不知道就说不知道"），重跑，**观察幻觉率上升**——证明 eval 能抓住质量回退。
4. **加一维**（任选）：加 `conciseness`（简洁度）或 `format_ok`（是否符合要求格式）评分，扩成三维评估。

**验收标准**：能跑出多维分数；幻觉陷阱被正确识别；改差 prompt 后幻觉率确实上升（回归被抓住）；至少做过一次裁判校准。

## 4. 自测清单

- [ ] 我理解开放式任务为何要 LLM-as-judge，而非精确匹配。
- [ ] 我会给裁判写明确 rubric、让它先讲理由再结构化打分。
- [ ] 我知道"裁判也会幻觉"，要抽样人工校准。
- [ ] 我能定义并测量幻觉率（grounded=False 的比例）。
- [ ] 我能把多维 eval 当回归套件，用分数变化抓住质量回退。

## 5. 延伸 & 关联

- 评估的输入最好来自真实 trace（[Day-48](./Day-48-trace-debugging.md)）；接下来两天解决"太贵 / 选型"：[Day-51](./Day-51-cost-optimization.md)、[Day-52](./Day-52-model-routing.md)。
- 本仓库相关章节：
  - 评估与监控（BLEU/ROUGE/BERTScore 等自动指标 + 幻觉/有害内容监控）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - RAG 实践（引用与溯源是压低幻觉的根本手段）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
