# Prompt Engineering 基础 / Prompt Engineering Basics

## 1. 背景（Background）
> Prompt Engineering 是不改变模型参数，通过设计输入提示来引导模型输出的技术。它是使用大模型最重要的实用技能。

## 2-3. 知识点与内容
```
核心技巧：
1. Zero-shot: 直接提问，不给示例
2. Few-shot: 给 2-5 个示例，引导输出格式
3. Chain-of-Thought (CoT): "让我们一步步思考"
4. System Prompt: 设定角色和规则
5. 输出格式约束: "以 JSON 格式输出"

Few-shot 示例：
"将以下文本分类为正面/负面：
输入：这个产品太棒了！ 输出：正面
输入：服务态度很差。 输出：负面
输入：今天天气不错。 输出："
```

## 4-6. 推理/例题/习题
**练习：** 设计 Prompt 让 LLM 做情感分析，对比 zero-shot/few-shot/CoT 效果。
