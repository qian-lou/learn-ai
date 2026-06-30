# Day 48 · 看 trace 调试：用 token / 延迟 / 成本定位真实 bug

> **今日目标**：学会**读** trace——逐 span 看 token、延迟、成本、工具入出参，用它定位一个你之前"看不见"的真实 bug。
> **时长**：~2h ｜ **前置**：Day 47（已接通 tracing）
> **今日产出**：一个**故意埋了 bug** 的 `buggy_agent.py`，你通过看 trace 把 bug 找出来并修掉，附一段"我是怎么从 trace 看出来的"复盘。

## 1. 为什么 & 是什么

接通 tracing 只是第一步；**会读 trace** 才是真本事。今天练的是一种新调试姿势：**不靠 print、不靠断点，靠看链路**。

给 Java 工程师的对照：线上一个接口慢了，你不会去本地单步调试，而是打开 **APM 看调用链**——哪个 RPC 耗时最长、哪个 SQL 慢、哪次重试了。Agent 调试一模一样：**打开 trace，看哪个 span 异常**。

Agent 里**只有 trace 能看见**的四类 bug：

| Bug 类型 | trace 上的症状 | 传统手段为什么看不见 |
|---|---|---|
| **死循环 / 反复调工具** | span 数量爆炸，同一工具被调 N 次 | 程序没崩，只是慢且贵 |
| **上下文膨胀** | 后面的 LLM span，`input_tokens` 越滚越大 | 输出看着正常，钱悄悄涨 |
| **工具被喂错参数** | 某 tool span 的入参里有脏值 / 类型不对 | 模型"自圆其说"地用了错数据 |
| **隐藏的高成本步骤** | 某一个 span 的 cost 占了大头 | 总账单涨了但不知道凶手 |

**核心心智：trace 是 Agent 的"飞行记录仪"。** 出问题时，你回放这盒磁带，逐帧看模型每一步**输入了什么、决策了什么、花了多少**。Day 46 你已知道一次 trace 长什么样；今天学会**从异常形状反推 bug**。

## 2. 跟着做（Hands-on）

下面这个"研究 Agent"埋了一个**真实又隐蔽**的 bug：它在多轮里**把工具的完整原始结果反复塞回 history**，导致上下文随轮数**线性膨胀**——输出看着没毛病，但 token 和成本悄悄爆炸。我们用 trace 抓它。

```bash
pip install "langfuse>=3" openai          # 复用 Day 47 的 Langfuse / reuse Day 47 setup
```

```python
"""Day 48: 一个埋了'上下文膨胀'bug 的 Agent / an agent with a context-bloat bug.

跑完去 Langfuse 看 trace：你会发现后面几轮的 input_tokens 异常地大。
After running, inspect the trace — later turns have suspiciously large input_tokens.
"""

from typing import Dict, List

from langfuse import get_client, observe
from langfuse.openai import openai

langfuse = get_client()
MODEL = "gpt-4o-mini"


@observe()
def fake_search(query: str) -> str:
    """模拟一次返回'大块原文'的检索工具 / a tool returning a big blob."""
    # 真实场景这里是网页正文/数据库行——很长 / pretend this is a long web page
    return f"[关于「{query}」的检索结果] " + ("内容 " * 400)  # 故意很长 / intentionally huge


@observe()
def buggy_research(topics: List[str]) -> str:
    """多轮研究：每轮检索一个主题再让模型小结 / multi-round research.

    BUG：把每轮 fake_search 的**完整原文**都 append 进 history，
    导致后续每轮的输入 token 线性膨胀。
    BUG: appends each full raw search blob into history → input grows every round.
    """
    history: List[Dict[str, str]] = [
        {"role": "system", "content": "你是研究助手，基于已知信息逐步总结。"}
    ]
    summary = ""
    for topic in topics:
        raw = fake_search(topic)
        # ↓↓↓ 这就是 bug：把超长原文整块塞回对话历史，且永不清理
        # ↓↓↓ the bug: dumping the full raw blob into history, never trimmed
        history.append({"role": "user", "content": f"资料：{raw}\n请把它纳入总结。"})
        resp = openai.chat.completions.create(
            model=MODEL, messages=history, name=f"summarize-{topic}",
        )
        summary = resp.choices[0].message.content
        history.append({"role": "assistant", "content": summary})
    return summary


if __name__ == "__main__":
    buggy_research(["向量数据库", "重排序", "混合检索", "评估指标"])
    langfuse.flush()
    print("去 Langfuse 看每个 summarize-* span 的 input_tokens 怎么涨的。")
```

**怎么用 trace 抓它**（在 Langfuse 控制台）：

1. 打开这条 trace，按顺序看 4 个 `summarize-*` span 的 **input tokens**。
2. 你会看到一条**单调上升**的曲线：第 1 轮可能 500 tok，第 4 轮飙到 6000+——这就是**上下文膨胀**的指纹。
3. 看 **cost** 列：后面几轮的成本远高于前面，元凶一目了然。
4. 点开某个 span 的 **input**，会看到 history 里堆了多份超长原文——**根因确认**。

**修复**（把原始大块换成"先压缩再入历史"）：

```python
@observe()
def fixed_research(topics: List[str]) -> str:
    """修复版：只把'精炼摘要'入历史，原文用完即弃 / keep summaries, drop raw blobs."""
    running_summary = ""
    for topic in topics:
        raw = fake_search(topic)
        resp = openai.chat.completions.create(
            model=MODEL,
            # 关键修复：每轮上下文 = 滚动摘要 + 本轮原文，而非无限堆叠
            # fix: context = rolling summary + THIS round's blob only
            messages=[
                {"role": "system", "content": "把新资料并入已有总结，保持简洁。"},
                {"role": "user", "content": f"已有总结：{running_summary}\n新资料：{raw}"},
            ],
            name=f"summarize-{topic}",
        )
        running_summary = resp.choices[0].message.content  # 只留摘要 / keep only the summary
    return running_summary
```

再跑一次 `fixed_research`，回 trace 看：4 个 span 的 input token **基本持平**，总成本明显下降。**你用眼睛（trace）而不是猜测，证明了 bug 已修。**

> 这正是 Day 51 成本优化的预演——**先观测、再优化**。没有 trace，你根本不知道钱花在"反复重发的原文"上。

## 3. 今日任务

1. 跑 `buggy_research`，在 trace 里**画出/记录** 4 轮的 input_tokens，确认单调上升。
2. **写出诊断**：用一句话说出"从哪个信号、看出是什么 bug"（如"summarize-* 的 input token 每轮翻倍 → 上下文膨胀"）。
3. 跑 `fixed_research`，在 trace 里对比修复前后的 token / 成本曲线，**用数字证明修好了**。
4. **换个 bug 练手**（任选其一）：制造"死循环"——让某工具在条件不满足时被反复调用；或制造"喂错参数"——把上一步结果的字段取错。然后**只靠 trace** 把它揪出来。

**验收标准**：能在 trace 里指出膨胀曲线并定位根因；修复后 token/成本曲线变平；额外完成一个"自造 bug→看 trace→定位"的练习，并写下诊断依据。

## 4. 自测清单

- [ ] 我会逐 span 读 token / 延迟 / 成本 / 入出参，而不是只看最终输出。
- [ ] 我能从"input token 单调上升"识别出上下文膨胀。
- [ ] 我能从"span 数量爆炸 / 同工具重复"识别出死循环。
- [ ] 我能从"某 span cost 占大头"定位高成本步骤。
- [ ] 我理解"先观测后优化"，并能用 trace 数字证明一次修复有效。

## 5. 延伸 & 关联

- 把"它答得对不对"也变成数字——评估入门：[Day-49-eval-intro.md](./Day-49-eval-intro.md)。
- 今天看到的"上下文膨胀"，正是后天成本优化的主战场：[Day-51-cost-optimization.md](./Day-51-cost-optimization.md)。
- 本仓库相关章节：
  - 评估与监控（延迟 P99 / 成本 / 质量的监控指标体系）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
