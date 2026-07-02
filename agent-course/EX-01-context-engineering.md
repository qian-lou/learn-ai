# EX-01 · 上下文工程（Context Engineering）

> **今日目标**：建立"把上下文窗口当稀缺资源管理"的心智——会做预算分配、会排序编排、会压缩摘要、懂多 agent 场景的上下文隔离，并能说清 `lost in the middle` 与 `context rot`。
> **时长**：~2h ｜ **前置**：建议完成 Day 1–70 主线后学习（至少 Day 9 / Day 11 / Day 43）
> **今日产出**：一张**自己的上下文预算表**——用 tiktoken 统计一次 agent 调用里各组成部分的 token 占比。

> 对应每日计划「🧩 扩展 / 进阶主题」中的 **Day EX-CTX**（P0 应补），不占 Day 1–70 编号。

## 1. 为什么 & 是什么

主线学到 Day 70，你已经会写 prompt、会截断历史（Day 11）、会省钱（Day 51）。但这些是"零散招式"。**上下文工程（context engineering）**是把它们统一成一门学科：**在每一步调用前，决定上下文窗口这块定长缓冲区里到底装什么、装多少、按什么顺序装。** 2026 年业界的重心已经从 prompt engineering（措辞怎么写）转向 context engineering（窗口里放什么）——因为 agent 是多轮循环，每一轮工具结果、检索片段都在往窗口里灌，"写好一句 prompt"早已不是瓶颈，"管好一整个窗口"才是。

给 Java 工程师的类比：

| 上下文工程 | Java 世界类比 | 说明 |
|---|---|---|
| Context window | 固定大小的 JVM 堆（`-Xmx`） | 定长预算，塞满就 OOM（截断/报错） |
| 上下文预算表 | heap dump / 内存分析报告 | 先量化"谁占了多少"，才谈得上优化 |
| 历史压缩/摘要 | GC + 对象池化 | 旧对象（旧轮次）折叠成小摘要，腾出空间 |
| 子 agent 上下文隔离 | 线程栈隔离 / 微服务拆分 | 各干各的，只把**结果**汇报给主调用方 |

必须建立的五个概念：

- **上下文预算分配**：一次 agent 调用的窗口大致由这几部分瓜分——**系统指令、工具 schema、对话历史、检索片段（RAG）、工具执行结果、当前问题**，还要给输出预留 `max_tokens`。像做内存规划一样给每部分定配额（例如：历史 ≤30%、检索 ≤25%），超配额就触发压缩，而不是放任窗口被单一部分挤爆。
- **编排与排序（lost in the middle）**：研究（Liu et al., 2023, *Lost in the Middle*）发现模型对**开头和结尾**的内容利用得最好，**中间**的信息最容易被"看丢"。所以：系统指令放最前，当前问题/关键指令放最后，最重要的检索片段别埋在一堆片段的正中间。另一个排序理由是 **prompt 缓存**（Day 51）：稳定不变的部分（系统指令、工具 schema）放前面，变动的部分放后面，前缀才可复用缓存。
- **上下文压缩/摘要**：Day 11 的截断是"扔掉"，压缩是"变小不变义"——把久远轮次折叠成一段摘要（保留结论与关键事实，丢弃过程细节），工具结果只保留 agent 决策需要的字段而不是整个原始 JSON。
- **多 agent 的上下文隔离与中间步骤压缩**：Day 43 里子 agent 检索了 10 个网页，如果把全文都塞回主 agent，主 agent 窗口立刻爆炸。正确做法：子 agent 在**自己独立的上下文**里干脏活，只把**压缩后的结论**（几百 token 的摘要 + 引用）返回主 agent。这正是"研究型多 agent"能处理超大信息量的原因——总信息量远超单一窗口，但每个窗口各自可控。
- **Context rot（上下文腐化）**：2025 年 Chroma 的报告用这个词描述一个反直觉事实——**即使远没到窗口上限，输入越长，模型表现也会退化**，且退化不均匀（连"复述一句话"这种简单任务也会变差）。启示：**"塞得下"≠"该塞"**，窗口利用率不是越高越好，无关内容本身就是性能毒药。

一句话心智：**prompt engineering 优化"怎么说"，context engineering 优化"让模型看到什么"。后者是 2026 年 agent 工程师的核心手艺。**

## 2. 跟着做（Hands-on）

**Step 1 — 装 tiktoken（OpenAI 官方 tokenizer 库）**

```bash
pip install tiktoken   # gpt-4o 家族使用 o200k_base 编码 / gpt-4o family uses o200k_base
```

**Step 2 — 写上下文预算表脚本**

```python
"""EX-01: 一次 agent 调用的上下文预算表 / context budget table."""

import tiktoken

# encoding_for_model 会为 gpt-4o-mini 返回 o200k_base 编码器
# encoding_for_model returns the o200k_base encoder for gpt-4o-mini
ENC = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    """统计一段文本的 token 数（不含消息结构的少量开销）。

    Args:
        text: 待统计文本 / the text to count.

    Returns:
        token 数量 / number of tokens.
    """
    # 时间 O(N) 空间 O(N)，N 为文本长度 / N = text length
    return len(ENC.encode(text))


def budget_table(sections: dict[str, str], window: int = 128_000) -> None:
    """打印各组成部分的 token 数与占比，按大小降序。

    Args:
        sections: 组成部分名 -> 该部分拼入上下文的文本 / part name -> text.
        window: 模型上下文窗口大小 / model context window in tokens.
    """
    # 时间 O(T + K log K) 空间 O(K)：T 为总字符数，K 为部分数
    counts = {name: count_tokens(text) for name, text in sections.items()}
    total = sum(counts.values())
    print(f"{'组成部分 / part':<26}{'tokens':>8}{'占比':>9}")
    print("-" * 43)
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{name:<26}{n:>8}{n / total:>8.1%}")
    print("-" * 43)
    print(f"{'合计 / total':<26}{total:>8}{total / window:>8.1%} of {window // 1000}k 窗口")


if __name__ == "__main__":
    # 模拟一次 ReAct 第 5 轮调用的窗口组成（用你自己的真实内容替换）
    # Simulated round-5 ReAct call; replace with your real payloads
    demo = {
        "系统指令 system": "你是研究助理…（约定输出格式、引用规范等）" * 20,
        "工具schema tools": '{"name": "search", "parameters": {...}}' * 30,
        "对话历史 history": "user: … assistant: … （前 4 轮的完整消息）" * 200,
        "检索片段 retrieval": "【片段1】…【片段2】…【片段3】…" * 150,
        "工具结果 tool_result": '{"results": [{"title": …, "snippet": …}]}' * 100,
        "当前问题 question": "综合以上信息，总结三条核心结论并给出引用。",
    }
    budget_table(demo)
```

**Step 3 — 运行并校准**

```bash
python ex01_budget.py
```

预期看到一张降序表格：`history`、`retrieval`、`tool_result` 通常是三大头，`question` 占比小得可怜——这就是 agent 场景的典型形态：**真正的"问题"只占窗口的零头，绝大部分预算花在"给模型喂背景"上。**

校准提示：真实 API 每条消息还有少量结构开销（每条几个 token），工具 schema 的服务端序列化方式也与你本地拼接略有出入，所以这张表是**估算**。拿它跟真实调用返回的 `response.usage.prompt_tokens` 对一下，误差在个位数百分比内即可放心使用。

## 3. 今日任务

1. 跑通 Step 2 的脚本，看懂降序表格与"占窗口百分比"。
2. **做一张真实的预算表**：从你 Day 45 的研究 agent（或 Day 9 的 ReAct loop）里，取**循环中后期的某一轮**真实调用，把 messages 里各部分文本按类别填进 `sections`，生成你自己的上下文预算表。对照 `usage.prompt_tokens` 校准。
3. **做一次预算优化**：选占比最大的那一项动手——历史折叠成摘要、或工具结果只保留决策所需字段——打印优化前后两张表，对比总 token 降幅。
4. **口头演练**：不看笔记，向自己解释 `lost in the middle` 和 `context rot` 的区别（一个关于**位置**，一个关于**长度**）。

**验收标准**：你有一张来自真实 agent 调用的预算表（估算 vs `usage.prompt_tokens` 误差可解释）；优化后的第二张表总量明显下降且 agent 行为未劣化；能一句话说清两个概念的区别。

## 4. 自测清单

- [ ] 我能列出一次 agent 调用上下文的 5~6 个组成部分，并说出自己项目里哪部分最占预算。
- [ ] 我能解释为什么"系统指令放最前、当前问题放最后"，以及这个排序如何同时服务注意力与 prompt 缓存。
- [ ] 我能区分截断（扔掉）与压缩（变小不变义），并知道各自适用场景。
- [ ] 我能说清多 agent 场景里"子 agent 上下文隔离 + 只回传压缩结论"为什么能突破单窗口限制。
- [ ] 我能向面试官解释 context rot：为什么"窗口塞得下"不等于"应该塞"。
- [ ] 我能用一句话说出 prompt engineering 与 context engineering 的分工。

## 5. 延伸 & 关联

- 三份值得精读的原始材料（按标题搜索）：Liu et al. 的论文 *Lost in the Middle: How Language Models Use Long Contexts*（位置效应的出处）；Chroma 的技术报告 *Context Rot: How Increasing Input Tokens Impacts LLM Performance*（长度退化的出处）；Anthropic 工程博客 *Effective context engineering for AI agents*（工业界最系统的实践总结）。
- 想直观感受不同文本的 token 开销：用本讲的 `count_tokens` 对比同义的中文/英文/JSON 三种表达——你会发现原始 JSON 工具结果往往是最"费"的，这也是"工具结果瘦身"收益大的原因。
- 本课程的相关 Day 讲义：
  - 上下文截断/压缩策略的实现基础：[Day-11-memory-and-context.md](Day-11-memory-and-context.md)
  - prompt 缓存与"稳定前缀"排序的成本视角：[Day-51-cost-optimization.md](Day-51-cost-optimization.md)
  - 子 agent 编排（隔离的工程载体）：[Day-39-tool-orchestration.md](Day-39-tool-orchestration.md)
  - 多 agent 上下文隔离的实战项目：[Day-43-research-agent-multiagent.md](Day-43-research-agent-multiagent.md)
  - 检索片段这部分预算怎么来的：[Day-20-retrieval-topk.md](Day-20-retrieval-topk.md)、[Day-23-advanced-retrieval.md](Day-23-advanced-retrieval.md)
- 主课程相关章节：
  - 提示工程基础（context engineering 的前身）：[../07-llm-applications/02-prompt-engineering/01-prompt-basics.md](../07-llm-applications/02-prompt-engineering/01-prompt-basics.md)
  - 提示工程进阶技巧：[../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md](../07-llm-applications/02-prompt-engineering/02-advanced-techniques.md)
  - LLM 核心技术总览（attention 与长上下文的原理侧）：[../06-llm-core-technology/README.md](../06-llm-core-technology/README.md)
