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

## 4. 详细推理（Deep Dive）

```
监控告警规则示例：
  P99 延迟 > 5s     → 告警（模型过载）
  错误率 > 1%       → 告警（模型异常）
  幻觉率 > 10%      → 告警（质量下降）
  GPU 利用率 < 30%  → 告警（资源浪费）
```

## 5-6. 例题/习题

**练习 1：** 用 ROUGE 评估一个摘要模型的输出质量。

**练习 2：** 实现 LLM-as-Judge 评估管线。

**练习 3：** 搭建 Prometheus + Grafana 监控面板，追踪 LLM 服务指标。
