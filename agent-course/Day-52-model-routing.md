# Day 52 · 智能路由：简单任务用小模型，难任务用大模型

> **今日目标**：实现按"任务难度"分流——便宜小模型扛大多数请求，难题才升级到大模型，在保质量的前提下大幅降本。
> **时长**：~2h ｜ **前置**：Day 51（成本优化）、Day 49（eval 守质量）
> **今日产出**：一个 `router.py`，含一个"难度分类→选模型"的路由器，并用 LiteLLM `Router` 加上失败回退；用 eval 验证"路由后质量不掉、成本下降"。

## 1. 为什么 & 是什么

Day 51 在"单次调用内"省钱；今天在"调用之间"省钱：**不是每个请求都配得上最贵的模型**。"今天几号"用大模型纯属浪费；"分析这份财报的风险点"才值得上大模型。智能路由就是**按需分配算力**。

给 Java 工程师的对照：这是**负载分级 / 服务降级**的老思路——

| 智能路由 | Java / 架构世界 | 说明 |
|---|---|---|
| 难度分类器 | 请求分类 / 优先级队列 | 先判这个请求"多难" |
| 小模型/大模型路由 | 普通实例 vs 高配实例 | 难的走强模型，简单的走便宜模型 |
| 失败回退（fallback） | 熔断后降级 / 重试到备用 | 小模型答不好 → 升级到大模型 |
| 成本/延迟权衡 | SLA 分级 | 用对模型，省钱又不牺牲体验 |

**三种路由策略，从简到繁：**

1. **规则路由**：按输入长度、是否含代码、关键词等硬规则选模型。简单、零额外延迟、可解释。
2. **分类器路由（LLM 判难度）**：用一个**便宜模型**先给请求打"难度分"，再据此选模型。更准，但多一次小调用。
3. **专用路由库**：如 **RouteLLM**（LMSYS，用学习到的路由器在"质量/成本"间找最优点），或 **LiteLLM Router**（多模型 + 自动 fallback + 重试）。

**核心心智：路由是"用一点点判断成本，换一大笔生成成本"。** 一次几百 token 的难度判断（或一条规则，零成本），可能为你省下一次昂贵的大模型调用。**但必须用 eval 兜底**——路由错了会掉质量，所以"路由后质量是否守住"要能量化（Day 49~50）。

### 2026 补充：推理模型当 Agent 大脑——路由从"成本单轴"升级为"能力/推理 × 成本"两轴

上面的 SMALL/BIG 只有**成本一条轴**；2026 年选型还有一条正交的**推理轴**：推理模型（OpenAI o 系列 / GPT-5 的 reasoning、Claude 的 extended thinking、DeepSeek-R1 等）在作答前先生成内部思考，多步规划与自我纠错能力远强于同档常规模型，代价是**思考 token（按输出计费）+ 明显延迟**。于是路由决策从"小/大"二选一变成一张 2×2 矩阵：

| | 常规模型 | 推理模型 |
|---|---|---|
| **便宜档** | 闲聊、抽取、格式化（本文 SMALL） | 中等难度推理（o4-mini 这类小杯推理模型 + 低 effort） |
| **旗舰档** | 长文生成、宽知识面问题（本文 BIG） | 规划、数学/代码证明、疑难 debug |

- **何时开推理**：任务需要多步规划、代码/数学推理、自我检查纠错时才开；抽取、闲聊、格式转换开了纯属浪费——思考 token 照样计费，延迟可能翻数倍。粒度旋钮：OpenAI 系用 `reasoning_effort`（low/medium/high），Anthropic 系用 `thinking={"type": "enabled", "budget_tokens": N}` 限思考预算——**effort/budget 本身就是路由器的一个输出**，工程问题从"用哪个模型"变成"这个任务值得为推理付多少延迟和 token"。
- **推理模型的 prompt 写法不同**：**给目标与约束，不给步骤**。别再写 "Let's think step by step"、别喂手写 few-shot CoT——模型自己会推理，外挂思考链反而干扰甚至降低表现（OpenAI 对 o 系列的官方建议）；指令简洁直接、零样本优先，把省下的篇幅用来写清成功标准。
- **架构建议（Agent 内按节点路由）**：planning / reflection 节点（要"想清楚"的地方，如 Day 41 研究 Agent 的规划与反思环节）路由到推理模型；工具调用、格式化输出、简单抽取等执行节点路由到快模型。一个 Agent 内部就是一张节点粒度的路由表——两轴选型最终落到每个节点上。

## 2. 跟着做（Hands-on）

### Step 1 — 规则 + 分类器路由（看清原理）

```bash
pip install "openai>=1.0" "pydantic>=2"
```

```python
"""Day 52: 难度路由 —— 规则 + LLM 分类器 / difficulty routing."""

from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()
SMALL = "gpt-4o-mini"   # 便宜、扛大多数 / cheap workhorse
BIG = "gpt-4o"          # 贵、留给难题 / pricey, hard tasks only


# ---- 策略一：规则路由（零额外成本，最先用）/ rule-based routing ----
def route_by_rule(question: str) -> str:
    """按硬规则选模型：长/含代码/含'分析'→大模型 / pick model by rules. 时间 O(n) 空间 O(1)。"""
    q = question.lower()
    looks_hard = (
        len(question) > 200                       # 长输入往往更复杂 / long → likely complex
        or any(k in q for k in ["代码", "code", "证明", "分析", "推理", "debug"])
    )
    return BIG if looks_hard else SMALL


# ---- 策略二：LLM 分类器路由（更准，多一次小调用）/ classifier routing ----
class Difficulty(BaseModel):
    """难度判定结果 / difficulty verdict."""

    level: Literal["easy", "hard"] = Field(description="easy=事实/闲聊, hard=推理/分析/代码")


def route_by_classifier(question: str) -> str:
    """用便宜模型先判难度，再据此选模型 / classify difficulty with the cheap model."""
    completion = client.beta.chat.completions.parse(
        model=SMALL,  # 用便宜模型当分类器 / the classifier itself is cheap
        messages=[
            {"role": "system", "content": "判断这个问题的难度：easy 还是 hard。"},
            {"role": "user", "content": question},
        ],
        response_format=Difficulty,
    )
    return BIG if completion.choices[0].message.parsed.level == "hard" else SMALL


def answer(question: str, route=route_by_classifier) -> tuple[str, str]:
    """路由后作答，返回 (所选模型, 回答) / route then answer."""
    model = route(question)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}],
    )
    return model, resp.choices[0].message.content


if __name__ == "__main__":
    for q in ["现在几点的概念是什么？", "推导一下快速排序的平均时间复杂度并证明。"]:
        model, ans = answer(q)
        print(f"[路由→{model}] {q}\n  → {ans[:60]}...\n")
    # 观察：闲聊走 SMALL，推理题走 BIG —— 大多数流量落在便宜模型上
```

### Step 2 — 用 LiteLLM Router 做多模型 + 自动回退（生产做法）

```python
# pip install litellm
"""Day 52 Step2: LiteLLM Router 统一多模型 + fallback / unified routing + fallback."""
from litellm import Router

router = Router(
    model_list=[
        {   # 默认首选：便宜小模型 / default cheap
            "model_name": "cheap",
            "litellm_params": {"model": "gpt-4o-mini", "api_key": "os.environ/OPENAI_API_KEY"},
        },
        {   # 兜底强模型 / strong fallback
            "model_name": "strong",
            "litellm_params": {"model": "gpt-4o", "api_key": "os.environ/OPENAI_API_KEY"},
        },
    ],
    # cheap 失败(报错/超时/内容策略)时自动回退到 strong / auto-fallback
    fallbacks=[{"cheap": ["strong"]}],
    num_retries=2,  # 失败重试 / retries before fallback
)

if __name__ == "__main__":
    # 统一入口：先打便宜模型，出问题自动升级 / one entry point, auto-escalation
    resp = router.completion(
        model="cheap", messages=[{"role": "user", "content": "用一句话解释智能路由。"}],
    )
    print(resp.choices[0].message.content)
```

> **进阶：RouteLLM（LMSYS）**。它用在 Chatbot Arena 数据上训练的路由器，给定一个"成本预算阈值"，自动把请求在强/弱模型间分配以**逼近大模型质量、只花一部分钱**。用法是 `from routellm.controller import Controller`，传 `strong_model` / `weak_model` 和一个 `router-mf-<阈值>` 模型名。适合"想要数据驱动的最优分流"的场景。

**路由的纪律（别翻车）：**

- **先规则，后分类器**：规则零成本、可解释，能用规则解决就别加 LLM 判断。
- **必须 eval 兜底**：用 Day 49~50 的测试集，对"全大模型 / 全小模型 / 路由"三档各跑一次，**对比质量与成本**，证明路由是帕累托更优（省钱且质量没明显掉）。
- **留逃生通道**：分类器判错时，靠 fallback（小模型答得差 → 升级重答）或人工兜底。
- **观测命中分布**：回 Day 47 trace，看多少流量落在便宜模型——这直接对应省了多少钱。

## 3. 今日任务

1. 跑通 Step 1，确认闲聊走 `SMALL`、推理/代码题走 `BIG`，且分类器本身用的是便宜模型。
2. 跑通 Step 2 的 LiteLLM Router，**手动制造一次 cheap 失败**（如填个无效模型名触发回退），确认自动 fallback 到 strong。
3. **三档对比实验**：用 Day 49 的测试集，对"全 BIG / 全 SMALL / 路由"各跑一遍，做一张表对比**准确率 + 总成本**——证明路由是更优解。
4. **统计命中率**：跑一批混合难度的问题，统计有多少比例落在 `SMALL`，估算相对"全 BIG"省了多少钱。

**验收标准**：路由能按难度正确分流；fallback 在 cheap 失败时生效；三档对比表显示"路由质量≈全大模型，但成本显著更低"；能报出小模型命中比例与省钱估算。

## 4. 自测清单

- [ ] 我理解"不是每个请求都配得上最贵模型"，路由是按需分配算力。
- [ ] 我能写规则路由，也能用便宜模型做分类器路由，并知道二者取舍。
- [ ] 我会用 LiteLLM Router 配多模型 + fallback + 重试。
- [ ] 我知道路由必须用 eval 兜底，避免省钱却掉质量。
- [ ] 我会用 trace 看小模型命中分布，把"省了多少"量化出来。
- [ ] （2026）我理解"能力/推理 × 成本"两轴选型：知道何时开 extended thinking / 调 reasoning effort，推理模型 prompt 要"给目标不给步骤、少喂手写 CoT"，并会把 planning/reflection 节点路由到推理模型、执行节点用快模型。

## 5. 延伸 & 关联

- 路由 + 缓存 + 裁剪三件套合起来就是完整降本方案（[Day-51](./Day-51-cost-optimization.md)）；其安全有效性靠 Day 49~50 的 eval 与 Day 47 的 trace 共同保障。
- 本仓库相关章节：
  - 知识蒸馏（训练一个更小但够用的模型，是"小模型分流"的上游手段）：[../08-llm-engineering/01-model-optimization/03-knowledge-distillation.md](../08-llm-engineering/01-model-optimization/03-knowledge-distillation.md)
  - Scaling Laws（理解模型大小与能力的关系，指导选型）：[../06-llm-core-technology/02-pretrained-models/04-scaling-laws.md](../06-llm-core-technology/02-pretrained-models/04-scaling-laws.md)
