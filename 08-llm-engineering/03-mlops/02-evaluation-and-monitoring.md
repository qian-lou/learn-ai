# 模型评估与监控 / Model Evaluation and Monitoring

## 1. 背景（Background）

> **为什么要学这个？**
>
> 大模型上线后需要持续监控——评估输出质量、检测幻觉、跟踪延迟和成本。"模型在 benchmark 上很好" 不代表 "模型在生产中很好"。
>
> 对于 Java 工程师来说，这就像 **APM（应用性能监控）**——Prometheus + Grafana 监控 QPS 和延迟，加上 LLM 特有的质量指标。

## 2. 知识点（Key Concepts）

| 评估维度 | 指标 | 工具 |
|----------|------|------|
| 质量 | BLEU, ROUGE, BERTScore | evaluate |
| 安全 | 幻觉率，有害内容率 | Guardrails |
| 性能 | 延迟 P99, 吞吐量 | Prometheus |
| 成本 | Token 消耗, GPU 利用率 | W&B |

## 3. 内容（Content）

### 3.1 自动评估指标

```python
# ============================================================
# 文本生成评估 / Text generation evaluation
# ============================================================

# ROUGE（摘要评估）
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score("reference summary text", "generated summary text")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# BERTScore（语义相似度）
# from bert_score import score
# P, R, F1 = score(["generated"], ["reference"], lang="en")

# LLM-as-Judge（用 GPT-4 评估）
judge_prompt = """
请评估以下 AI 回答的质量（1-10 分）：
问题：{question}
回答：{answer}
评分标准：准确性、完整性、清晰度
"""
```

### 3.2 幻觉检测

```
幻觉检测策略：

1. 基于检索的验证
   回答 → 提取事实声明 → 与知识库核对

2. 自我一致性
   同一问题生成 5 次 → 交集 = 可靠信息

3. 不确定性量化
   Token logprob < threshold → 标记为不确定

4. 引用验证
   要求模型给出引用 → 验证引用是否存在
```

### 3.3 线上监控

```python
# Prometheus 指标收集
from prometheus_client import Counter, Histogram, start_http_server

request_count = Counter('llm_requests_total', 'Total LLM requests')
latency = Histogram('llm_latency_seconds', 'LLM response latency')
token_usage = Counter('llm_tokens_total', 'Total tokens used')

@app.post("/chat")
async def chat(req):
    request_count.inc()
    with latency.time():
        response = await model.generate(req.message)
    token_usage.inc(response.usage.total_tokens)
    return response
```

### 3.4 LLM-as-judge 与现代评测基准（2024-2025）

> 2024-2025 主线：生成质量评测已从 BLEU/ROUGE 字面匹配，转向 **LLM-as-judge**（强模型打分）+ **防污染基准**。
> LLM-as-judge / Strong-model grading is now the de-facto standard for generation quality.

**(1) LLM-as-judge（事实标准 / de-facto standard）**
两种范式：**pointwise**（单条 1-10 打分）与 **pairwise**（A/B 对比，更稳）。
已知偏差与缓解 / Known biases & mitigations：
- **Position bias**（偏好靠前答案）→ 交换 A/B 顺序各跑一次取一致结果 / swap order, run twice。
- **Verbosity bias**（偏好长答案）→ rubric 中明确"简洁不扣分" / penalize padding in rubric。
- **Self-preference**（偏好同源模型）→ 用异源裁判 + few-shot 锚点样例 / cross-family judge + anchors。

```python
# ============================================================
# pairwise LLM-as-judge 最小可运行示例 / Minimal runnable pairwise judge
# 依赖 / Deps: pip install openai>=1.92 pydantic>=2 ; export OPENAI_API_KEY=...
# ============================================================
from openai import OpenAI
from pydantic import BaseModel, Field

class Verdict(BaseModel):
    """结构化裁判结果 / Structured judge verdict."""
    winner: str = Field(description="'A' | 'B' | 'tie'")  # 胜者 / winner
    reason: str  # 简要理由 / short rationale

client = OpenAI()  # 读取环境变量 OPENAI_API_KEY / reads OPENAI_API_KEY

def judge_pairwise(question: str, ans_a: str, ans_b: str) -> Verdict:
    """对比两条回答，缓解 position bias（交换顺序两次取一致）。
    Compare two answers; mitigate position bias by swapping order.

    Time: O(1) LLM calls (×2), Space: O(1)
    """
    rubric = ("评估准确性、完整性、简洁性；长答案不因长度加分。"
              "Judge accuracy/completeness/conciseness; do NOT reward verbosity.")
    def _ask(first: str, second: str) -> str:
        rsp = client.chat.completions.parse(  # 结构化输出 / structured output
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"你是严格的评测裁判。{rubric}"},
                {"role": "user",
                 "content": f"问题: {question}\n[A]: {first}\n[B]: {second}\n谁更好?"},
            ],
            response_format=Verdict,  # SDK 自动注入 JSON Schema / auto JSON Schema
        )
        return rsp.choices[0].message.parsed.winner

    fwd = _ask(ans_a, ans_b)                       # 正序 A=a / forward
    rev = {"A": "B", "B": "A", "tie": "tie"}[_ask(ans_b, ans_a)]  # 逆序回映 / remap
    winner = fwd if fwd == rev else "tie"          # 不一致判平局 / disagree -> tie
    return Verdict(winner=winner, reason=f"forward={fwd}, swapped={rev}")

# print(judge_pairwise("用一句话解释 RAG", "检索增强生成。", "RAG 是一种很复杂的技术，它……"))
```

**(2) 评测基准与 harness / Benchmarks & harnesses**
- **Harness**：`lm-evaluation-harness`（EleutherAI，事实标准）、`OpenCompass`（中文友好）。
- **防污染 / Contamination-resistant**：`LiveBench`（按月更新题目，避免训练集泄漏）。
- **常见 benchmark**：MMLU-Pro、GPQA（研究生级推理）、HumanEval+ / MBPP+（代码）、
  GSM8K / MATH（数学）、IFEval（指令遵循）、Arena-Hard 与 Chatbot Arena Elo（人类偏好）。

**(3) Ragas 真实调用 / Real Ragas usage（替代前文手写 mock）**
正文例题用的是手写规则；生产中用 `ragas.evaluate(...)` 调强模型自动评检索/生成质量。

```python
# ============================================================
# Ragas 检索+生成自动评测 / Retrieval+generation auto-eval
# 依赖 / Deps: pip install ragas>=0.2 langchain-openai datasets
# 注意 / NOTE: 0.2.x API（EvaluationDataset + 指标对象）；旧版 0.1 用 dataset=Dataset。
# ============================================================
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))  # 裁判模型 / judge

dataset = EvaluationDataset.from_list([{
    "user_input": "Redis 为什么能抗高并发?",                # 问题 / query
    "retrieved_contexts": ["Redis 基于内存且单线程事件循环，避免锁竞争。"],  # 召回 / contexts
    "response": "因为 Redis 把数据放内存并用单线程模型规避锁竞争。",          # 生成 / answer
    "reference": "Redis 使用内存存储与单线程事件模型，因此高并发性能强。",    # 参考 / ground truth
}])

result = evaluate(  # 内部对每条样本调用 LLM 打分 / LLM-graded per sample
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
)
print(result)  # -> {'faithfulness': 1.0, 'answer_relevancy': 0.97, 'context_precision': 1.0, ...}
```

## 4. 详细推理（Deep Dive）

```
监控告警规则示例：
  P99 延迟 > 5s     → 告警（模型过载）
  错误率 > 1%       → 告警（模型异常）
  幻觉率 > 10%      → 告警（质量下降）
  GPU 利用率 < 30%  → 告警（资源浪费）
```

## 5. 例题（Worked Examples）

### 例题 1：使用 Ragas 计算 RAG 检索生成系统的可信度与答案相关性 / Evaluating RAG system using Ragas

大模型应用的在线评估不能单纯依赖精准比对。以下例题使用 Ragas 框架评估 RAG 系统答复质量。

```python
# ============================================================
# RAG 在线评估指标定义
# ============================================================
# 1. Faithfulness (忠实度): 生成的回答是否能从检索到的上下文中推导出来
# 2. Answer Relevance (回答相关性): 生成的回答是否切中用户的提问核心

import numpy as np

def estimate_faithfulness(context: str, answer: str) -> float:
    """模拟评估系统根据上下文对生成回答的召回比对 / Simulate faithfulness.
    
    Time: O(Len_C * Len_A), Space: O(1)
    """
    # 简化规则判断：如果上下文包含了回答里的核心实体词，则得分高
    keywords = ["高并发", "异步", "缓存", "数据库"]
    matches = [w in context and w in answer for w in keywords]
    return float(sum(matches) / len(keywords))

context_doc = "为了处理高并发场景，系统设计必须引入异步消息中间件，并且将热数据放入 Redis 缓存数据库中。"
answer_text = "开发高性能系统应当选用高并发的异步模型，并且集成缓存数据库。"

score = estimate_faithfulness(context_doc, answer_text)
print(f"RAG  Faithfulness (忠实度评分): {score:.2f}")  # 得分 0.75
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在 LLM 应用中，为什么传统 NLP 的字面匹配评估指标（如 ROUGE、BLEU）不再适合直接用来评估大模型的文本生成质量？
*参考答案*：
大模型的文本生成具有极高的多样性。同一个问题，大模型可以使用完全不同的词汇和句式进行正确答复，但 ROUGE 和 BLEU 只是基于字词在 N-gram 上的硬匹配，会导致虽然模型生成了正确且完美的回答，但匹配分值却极低。因此现代评估更侧重于基于 LLM-as-a-Judge 或者向量相似度去衡量语义上的正确性。

### 进阶题
**练习 2**：在生产环境中，大模型 API 的延迟往往受首字延迟（TTFT）与后续流式字符生成速度（ITL - Inter-Token Latency）的共同制约。如果我们要用 Prometheus 对在线推理服务的这两个指标进行实时监控和报警，应该定义什么类型的 Metric，并编写统计代码？
*参考答案*：
应该定义 `Histogram` 类型的指标（例如 `llm_ttft_seconds` 和 `llm_inter_token_latency_seconds`），因为这两个指标的分布是连续的，且需要分析分位数（如 P95、P99 延迟）。
```python
# 使用 prometheus_client SDK
# from prometheus_client import Histogram
# ttft_histogram = Histogram('llm_ttft_seconds', 'Time to first token in seconds')
# ttft_histogram.observe(time_first_token - start_time)
```\n