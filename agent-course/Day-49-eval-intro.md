# Day 49 · 评估（Eval）入门：把"它答得好不好"变成数字

> **今日目标**：理解 Agent 为什么**必须有 eval**，分清几类评估方法，亲手构建一个小测试集并跑出第一个量化分数。
> **时长**：~2h ｜ **前置**：Day 4（结构化输出）、Day 46~48（可观测性）
> **今日产出**：一个 `eval_intro.py` + 一个 `dataset.jsonl`，用代码对你的 Agent 跑一遍测试集，输出"通过率 / 准确率"这种可对比的数字。

## 1. 为什么 & 是什么

到目前为止,你判断 Agent 好不好,靠的是**手动多聊几句、凭感觉**。这在 demo 阶段够用,**一旦要迭代就崩**:

- 你改了个 prompt,**修好了 A 问题,但有没有弄坏 B 问题?** 不知道——你不可能每次都手动回归所有场景。
- 换个模型(便宜的那个)能不能扛住?**没有分数就没法对比。**
- 老板问"准确率多少",你答不上来。

这正是软件工程的老问题,你太熟了——**没有测试,就不敢重构**。给 Java 工程师的对照:

| Agent 评估 | Java / 测试世界 | 说明 |
|---|---|---|
| **测试集(dataset)** | JUnit 测试用例集 | 一组(输入, 期望)样本 |
| **评估器(evaluator)** | `assertEquals` / 断言 | 判定一次输出算不算"对" |
| **eval 跑批** | `mvn test` 跑全套 | 一次性对整个集合打分 |
| **回归测试** | CI 里的回归套件 | 改动后重跑,防止"修一处崩另一处" |
| **准确率 / 通过率** | 测试通过数 / 总数 | 一个能横向对比的数字 |

**核心心智:eval 就是 LLM 时代的单元测试 + 回归测试。** 区别在于——传统断言是 `==` 精确匹配,但 LLM 输出**不确定、措辞千变万化**,"对不对"往往没法 `==`。所以评估方法分几档,**按"判定难度"选**:

| 方法 | 怎么判对 | 适合 | 类比 |
|---|---|---|---|
| **精确匹配 / 规则** | `==`、正则、JSON 字段比对 | 结构化输出、分类、抽取 | 传统断言 |
| **指标计算** | 相似度、包含关键事实 | 问答、摘要 | 字符串相似度断言 |
| **LLM-as-judge** | 让另一个 LLM 按 rubric 打分 | 开放式生成、"有没有帮上忙" | 人工评审的自动化(Day 50 深入) |
| **人工评估** | 真人打标 | 黄金标准、抽样兜底 | 手动 QA |

**今天先打地基:构建测试集 + 跑能用规则判定的那档。** 测试集是 eval 的**核心资产**——它比代码更值钱,因为它沉淀了"什么叫做对"。怎么来?(1) 手写典型场景,(2) 从 Day 48 的真实 trace / 线上日志里捞真实 case,(3) 故意构造边界与失败样本。**好测试集要覆盖:正常 + 边界 + 已知会错的坑。**

### 2026 补充：eval 数据集工程

上面"测试集怎么来"三句话,业界已沉淀成一套标准打法,值得单独记一节:

- **三来源,各司其职**:① **人工种子**——领域专家手写几十条典型 + 边界样本,冷启动用,质量最高也最贵;② **生产 trace 回流**——从线上日志里捞真实失败 case(用户点踩、转人工、异常中断的会话)回灌进数据集,最贴真实分布,是数据集持续生长的主引擎(Day 59 的数据飞轮);③ **合成扩充**——用 LLM 对种子样本造变体(换措辞、换语言、加干扰信息),廉价上量,但合成样本必须**抽样人审**,否则会放大种子里的偏差。
- **标注 rubric 要写成文档**:开放式任务没法 `==`,"对不对"的标准要落成 rubric——明确评什么维度、每一档长什么样、配正反例,让"什么叫 3 分"不因标注人而异。同一份 rubric 既给人工标注用,也喂给 LLM-as-judge 当打分说明。
- **LLM-as-judge 先校准、再信任**:judge 自己也是个 LLM,有已知偏置(偏爱长回答、偏爱与自己同源的输出、受选项位置影响)。上岗前先在一小批**人工标注的子集**上对齐:算 judge 与人工判定的一致率,达标才放行,之后定期抽样复核。经验共识:**pairwise 比较("A 和 B 哪个更好")比绝对打分("给这条打 1~5 分")更稳定**——无论人还是模型,都更擅长"比较"而不是"定标"。Day 50 会动手实践。
- **数据集要版本化**:dataset 跟代码一样进 git(或用 eval 平台自带的 dataset versioning)。每个准确率数字都必须能回答"这是在哪一版数据集上测的",否则改动前后的分数根本不可比。

一句话点睛:模型会换、prompt 会重写、框架会过时,但沉淀了"什么叫做对"的 **eval 数据集是你换什么都带得走的资产——eval 数据集就是护城河**。

## 2. 跟着做（Hands-on）

我们评估一个简单的"情感分类 Agent"(输出限定 positive/neutral/negative,正好能用**精确匹配**判定,先把闭环跑通)。

### Step 1 — 写测试集（`dataset.jsonl`，一行一个样本）

```jsonl
{"id": 1, "input": "这手机太好用了，强烈推荐！", "expected": "positive"}
{"id": 2, "input": "客服爱答不理，再也不买了。", "expected": "negative"}
{"id": 3, "input": "包装一般，东西还行吧。", "expected": "neutral"}
{"id": 4, "input": "虽然贵了点，但质量是真的好。", "expected": "positive"}
{"id": 5, "input": "不功能不算坏，但也谈不上惊喜。", "expected": "neutral"}
{"id": 6, "input": "物流快是快，可惜到货就坏了。", "expected": "negative"}
```

> 这里特意放了第 4、6 条"转折句"(有褒有贬),它们是最容易分错的**坑样本**——好测试集就该专挑这种。

### Step 2 — 被测系统 + eval 跑批

```python
"""Day 49: 最小 eval 框架 / a minimal eval harness (rule-based).

把“答得好不好”变成一个通过率数字，并打印逐条结果。
Turns "is it good?" into a pass-rate number with per-case breakdown.
"""

import json
from pathlib import Path
from typing import Callable, Literal

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()
MODEL = "gpt-4o-mini"

Label = Literal["positive", "neutral", "negative"]


class Sentiment(BaseModel):
    """受限输出，便于精确匹配判定 / constrained output for exact-match eval."""

    label: Label


# ---- 被测系统(System Under Test)：一次情感分类 / the system under test ----
def classify(text: str) -> str:
    """对一段文本做情感分类，返回标签字符串 / classify sentiment."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "判断这条评论的情感，只输出 positive/neutral/negative 之一。"},
            {"role": "user", "content": text},
        ],
        response_format=Sentiment,
    )
    return completion.choices[0].message.parsed.label


# ---- 评估器：精确匹配(规则) / the evaluator: exact match ----
def exact_match(prediction: str, expected: str) -> bool:
    """判定一次预测是否正确 / one assertion. 时间 O(1)。"""
    return prediction.strip().lower() == expected.strip().lower()


def run_eval(dataset_path: str, system: Callable[[str], str]) -> None:
    """对整个测试集跑批，打印逐条结果 + 汇总准确率 / run dataset & report.

    Args:
        dataset_path: jsonl 测试集路径 / path to the jsonl dataset.
        system: 被测系统函数，输入文本→输出标签 / the SUT callable.
    """
    cases = [json.loads(ln) for ln in Path(dataset_path).read_text("utf-8").splitlines() if ln.strip()]
    results = [(c, system(c["input"])) for c in cases]  # 逐条跑被测系统 / call the SUT
    passed = sum(exact_match(pred, c["expected"]) for c, pred in results)
    print(f"\n准确率 / accuracy: {passed}/{len(cases)} = {passed / len(cases):.0%}\n")
    for c, pred in results:
        mark = "✅" if exact_match(pred, c["expected"]) else "❌"
        print(f"{mark} #{c['id']} 期望={c['expected']:<8} 实际={pred:<8} | {c['input']}")


if __name__ == "__main__":
    # 失败样本（❌）正是下一轮要改 prompt 的靶子 / failures = next iteration's targets
    run_eval("dataset.jsonl", classify)
```

运行后你会得到一行**准确率**和逐条 ✅/❌。现在你有了**数字**:改 prompt、换模型后**再跑一次**,数字涨了就是真的改好了,而不是"感觉好了"。

> **这就是回归测试的雏形。** 把这个脚本接进 CI(Day 56 起),每次提交自动跑,准确率掉了就拦住——你就敢放手迭代了。Day 50 会给它加上**幻觉率**和 **LLM-as-judge**,覆盖那些没法 `==` 判定的开放式任务。

## 3. 今日任务

1. 跑通 `eval_intro.py`,拿到第一个准确率数字和逐条结果。
2. **扩充测试集**到 12+ 条,刻意加难样本(双重否定、反讽、夹杂表情)。观察准确率怎么变。
3. **做一次对比实验**:把 `MODEL` 换成一个更小/更便宜的模型(或改 system prompt),重跑,**用准确率数字对比**两版优劣——体会"有 eval 才敢改"。
4. **从真实数据建集**:回到 Day 48 的 trace 或随便造几条线上风格的输入,挑 2~3 条加进测试集,并标注期望——体会"测试集来自真实流量"。

**验收标准**:能输出准确率 + 逐条结果;测试集≥12 条且含坑样本;完成至少一次"改动前后用数字对比";测试集里有来自真实/拟真场景的样本。

## 4. 自测清单

- [ ] 我能把 eval 类比成"LLM 时代的单元测试 + 回归测试"。
- [ ] 我能说出四档评估方法,以及各自适合什么任务。
- [ ] 我理解"为什么 LLM 输出常常没法用 `==` 判定"。
- [ ] 我会构建测试集,并知道好测试集要覆盖正常 / 边界 / 坑样本。
- [ ] 我能用准确率数字,做一次"改动前后"的客观对比。
- [ ] 我能说出 eval 数据集的三来源(人工种子 / trace 回流 / 合成扩充),并解释为什么 LLM-as-judge 要先与人工标注校准、为什么 pairwise 优于绝对打分。

## 5. 延伸 & 关联

- 明天把评估升级:**幻觉率 + LLM-as-judge + 回归测试**,覆盖开放式任务:[Day-50-writing-evals.md](./Day-50-writing-evals.md)。
- 测试集的输入,最好来自 Day 48 的真实 trace——可观测性与评估是一对:[Day-48-trace-debugging.md](./Day-48-trace-debugging.md)。
- 本仓库相关章节：
  - 评估与监控（BLEU/ROUGE/BERTScore 等指标 + 生产监控全景）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - 结构化输出（受限输出让"精确匹配评估"成为可能）：[Day-04-structured-output.md](./Day-04-structured-output.md)
