# 高级提示技巧 / Advanced Prompt Techniques

## 1. 背景（Background）
> 高级 Prompt 技巧包括 CoT、ReAct、Self-Consistency 等，能显著提升模型推理能力。

## 2-3. 知识点与内容
```
高级技巧：
1. Chain-of-Thought (CoT): 分步推理
2. Tree-of-Thought (ToT): 搜索多条推理路径
3. ReAct: 思考+行动交替（Agent 基础）
4. Self-Consistency: 多次采样取多数投票
5. Structured Output: 约束输出为 JSON/XML

ReAct 模式示例：
Thought: 我需要查询今天的天气
Action: search("今天北京天气")
Observation: 晴天，25度
Thought: 我现在知道答案了
Answer: 今天北京天气晴朗，温度25度
```

## 4-6. 推理/例题/习题
**练习：** 实现 ReAct 模式的简单 Agent（搜索+推理）。
