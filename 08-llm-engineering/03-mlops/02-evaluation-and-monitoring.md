# 模型评估与监控 / Model Evaluation and Monitoring

## 1. 背景（Background）
> 大模型上线后需要持续监控——评估输出质量、检测幻觉、跟踪延迟和成本。

## 2-3. 知识点与内容
```python
# 大模型评测框架
# 1. 人工评测：人类标注质量（金标准）
# 2. 自动评测：BLEU/ROUGE/BERTScore
# 3. LLM-as-Judge：用 GPT-4 评估其他模型的输出

# 常用评估指标
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score("reference text", "generated text")

# 幻觉检测（Hallucination Detection）
# 1. 基于检索的验证（RAG + 事实核查）
# 2. 自我一致性检查（多次生成取交集）
# 3. 不确定性量化（token 概率分析）

# 线上监控指标
# - 响应延迟 (P50/P95/P99)
# - 吞吐量 (requests/sec)
# - Token 用量和成本
# - 用户满意度反馈
```

## 4-6. 推理/例题/习题
**练习：** 构建一个 LLM 输出质量评估 Pipeline。
