# 02-prompt-engineering — Prompt 工程

> **所属阶段**：阶段七 · 大模型应用实战
> **学习目标**：不改模型参数，仅靠设计输入把同一个模型调到最优——掌握从基础模式到 CoT/ReAct 的全套提示技巧
> **预估时长**：3-4 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [prompt-basics](./01-prompt-basics.md) | Prompt 基础与原则 | Zero-shot/Few-shot、System Prompt 角色设定、JSON 输出约束、Prompt 模板函数、六大设计原则与常见错误 |
| 02 | [advanced-techniques](./02-advanced-techniques.md) | 高级技巧 | Chain-of-Thought 思维链、Self-Consistency 多数投票、ReAct 思考-行动循环、推理模型与 Structured Outputs（json_schema）|

---

## 🔑 知识点详解

### 01 · Prompt 基础与原则

- **核心概念**：Prompt 工程的本质是**用输入激活模型预训练时学到的模式**——Few-shot 靠 In-Context Learning 补全"A→B, C→?"映射，System Prompt 靠角色词激活相关知识区，格式约束靠模型对 JSON/XML 规律的记忆。
- **关键写法**：Few-shot 几乎总优于 Zero-shot；`temperature=0` 让分类/抽取类任务可复现；输出要 JSON 就在 prompt 里明确给出目标结构。
- **易错点**：
  - Prompt 过长时模型易忽略**中间部分**（"lost in the middle"），关键指令放首尾。
  - 指令自相矛盾（"简短" + "详细解释"）会让输出不可控。
  - 没给示例却期待复杂格式输出，命中率低。
- **Java 视角**：Prompt 工程 ≈ 精心设计 API 的请求参数——接口（模型）不变，靠不同入参拿到完全不同的响应；System Prompt ≈ 全局拦截器里设定的上下文规则。
- **前置**：模块 01（能调起模型）。

### 02 · 高级技巧（CoT / Self-Consistency / ReAct）

- **核心概念**：让模型"多花计算"以提升复杂推理——CoT 用中间步骤当"草稿纸"，Self-Consistency 采样多条推理路径取众数，ReAct 把"思考→调工具→看结果"串成循环。
- **关键写法/公式**：
  - Zero-shot CoT：加一句 "Let's think step by step." 即可触发分步推理。
  - Self-Consistency：`temperature≈0.7` 采样 N 次 → `Counter(answers).most_common(1)` 取多数；N 越大越稳（5 次≈92%，10 次≈95%，示例数据）。
  - ReAct 模式：`Thought → Action → Observation → …→ Answer`。
  - Structured Outputs：`response_format=PydanticModel` / `{"type":"json_schema"}` 在**解码层强制**贴合 schema，比旧 `{"type":"json_object"}`（只保证合法 JSON）约束更强。
- **易错点**：
  - **对推理模型（o 系列 / DeepSeek-R1 等）再加 "Let's think step by step" 是反模式**——浪费 token 且干扰其内置长思维链。
  - CoT 在小模型上几乎不起作用（attention 容量不足以承载推理链）。
- **Java 视角**：Self-Consistency ≈ 集群多副本"多数表决"提高可靠性；ReAct ≈ Controller 的请求循环——调 Service(工具)→看返回→再决定下一步。
- **前置**：01（基础 Prompt）；ReAct 是模块 05 Agent 的理论基础。

---

## 🎯 学习要点

- **Few-shot 是默认起手**：几乎所有任务先给 2-4 个示例，边界样本会明显更稳；不够再上 CoT。
- **按任务难度选技巧**：简单分类/抽取用 Few-shot → 数学/逻辑用 CoT → 高准确率要求叠 Self-Consistency → 多步真实任务上 ReAct（即 Agent）。
- **结构化输出别再手拼**：需要给下游 API 用的输出，直接上 Pydantic + `parse()` / `json_schema`，拿到已校验的强类型对象，免去脆弱的正则解析。
- **推理模型换一套用法**：o 系列/R1 这类模型只给"任务 + 约束"，把推理交给它自己，别再手写 CoT；能一步答出的任务别上推理模型（贵且慢）。
- **量化评估你的 Prompt**：建一组带标签的测试集，把候选 prompt 都跑一遍统计准确率——凭感觉调 prompt 不如用数据说话。
- **2026 模型对齐**：`gpt-3.5-turbo` 已下线，示例统一用 `gpt-4o-mini` 等现役模型；推理任务用 `o4-mini` 并配 `reasoning_effort`。

---

## 🔗 关联

- **上一模块**：[01-huggingface](../01-huggingface/) — 先能加载/调起模型，再谈怎么提示它。
- **下一模块**：[03-rag](../03-rag/) — Prompt 是 RAG 生成环节的核心（"基于参考资料回答"就是一段 prompt）。
- **本阶段总览**：[阶段七 README](../README.md)
- **相关 Day**：[Day 3 Prompt 基础（角色/few-shot/CoT）](../../agent-course/Day-03-prompt-basics.md) · [Day 4 结构化输出（Pydantic + parse）](../../agent-course/Day-04-structured-output.md) · [Day 9 ReAct 循环](../../agent-course/Day-09-react-loop.md)。
