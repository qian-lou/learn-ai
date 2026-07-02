# EX-02 · 长期 / 跨会话记忆

> **今日目标**：搞懂「短期记忆 vs 长期记忆」的分层，理解记忆的写入 / 检索 / 遗忘三件事，并**不依赖任何框架**手写一个最小跨会话记忆层。
> **时长**：~2h ｜ **前置**：建议完成 Day 1–70 主线后学习（至少：Day 11 上下文管理、Day 16 embedding、Day 20 top-k 检索、Day 29 checkpointer）
> **今日产出**：一个 `ex02_memory.py`——会话 1 从对话中抽取用户事实并落盘，会话 2（全新进程）检索 top-k 记忆注入 system prompt，模型"记得你"。

> 对应《AI-Agent-每日学习计划》🧩 扩展主题中的 **`Day EX-MEM`（P0 应补）**，不占 Day 1–70 编号。

## 1. 为什么 & 是什么

Day 11 教你管理**一次会话内**的消息列表，Day 29 用 checkpointer 把 graph 状态落盘续跑——但两者本质都是「同一个线程的对话记录」。用户明天换个新会话再来，Agent 依然是失忆的。**长期记忆**要解决的就是：跨会话地记住"这个用户是谁、偏好什么、上次聊到哪"。

**记忆分层**（类比 Java：局部变量 vs 数据库里的用户 profile）：

| 层 | 生命周期 | 存在哪 | 本课程对应 |
|---|---|---|---|
| 短期记忆 | 单次会话内 | `messages` 列表 / graph state | Day 5 / Day 11 / Day 29 |
| 长期记忆 | 跨会话、跨线程 | 外部存储（JSON / 向量库 / DB） | 今天 |

**长期记忆内部再分两类**（借自认知科学）：

- **语义记忆（semantic）**：与时间无关的事实与偏好——"用户是 Java 工程师""向量库选了 pgvector"。最常用，今天动手做的就是它。
- **情节记忆（episodic）**：具体事件——"6 月 30 日那次会话，我们调通了 checkpointer，但 SQLite 路径踩过坑"。带时间戳，常用于"接着上次继续"。

**记忆系统的三个动作**（缺一不可，也是面试考点）：

1. **写入（when to write）**：什么时候抽取事实？常见策略：每轮对话后即时抽取（贵但及时）/ 会话结束时批量总结（便宜，主流默认）。抽取本身就是一次 LLM 调用：把对话喂给模型，让它输出结构化的事实列表（Day 4 的结构化输出）。
2. **检索（how to read）**：把记忆全塞进 prompt 会撑爆上下文（Day 11 的老问题），所以和 RAG 一样走**嵌入相似度 top-k**：用当前用户输入做 query，取最相关的几条注入 system prompt。
3. **遗忘 / 更新（forget & update）**：用户说"我不用 Milvus 了，改 pgvector"——旧记忆必须被**更新或作废**，否则两条矛盾记忆同时被检索出来。常见做法：写入前先检索相似旧记忆，让 LLM 判定 新增 / 更新 / 删除（mem0 就是这个思路）。

**与 RAG 的本质区别**——技术手段（嵌入 + top-k）几乎一样，区别在**数据是什么**：

| | 记忆 Memory | RAG |
|---|---|---|
| 数据来源 | 运行时从对话中**产生** | 预先准备的外部文档库 |
| 内容本质 | 关于**用户 / 会话的状态** | 外部**知识** |
| 归属 | 按用户隔离，个性化 | 通常全用户共享 |
| 可变性 | 频繁增改删（需遗忘机制） | 基本只读，批量重建索引 |

**2026 主流框架**（先手写原理，再看框架就一眼看穿）：

- **mem0**（`pip install mem0ai`）：给 Agent 加记忆层的库，核心就两个操作——`add`（喂对话，内部用 LLM 抽事实并与旧记忆比对增改删）和 `search`（按相似度取回）。
- **Letta**（原 MemGPT，伯克利 MemGPT 论文孵化）：思路更激进——把内存管理做成"操作系统"：上下文窗口当 RAM（core memory），外部存储当磁盘（archival memory），**Agent 自己调用工具读写自己的记忆**。
- ⚠️ **版本漂移提醒**：这两个项目 API 迭代很快（MemGPT → Letta 连名字都换了），上面只描述稳定的**思想**；具体接口以你安装当天的官方文档为准，不要照抄网上的旧代码。

## 2. 跟着做（Hands-on）

不装任何记忆框架，用 Day 1 的 SDK + Day 16 的 embedding，几十行写通"抽取 → 落盘 → 跨会话检索注入"全链路。

```python
"""EX-2: 最小长期记忆层——抽取 → 落盘 → 跨会话检索注入。

Minimal long-term memory: extract -> persist -> recall across sessions.
"""

import json
import os

import numpy as np
from openai import OpenAI

client = OpenAI()
STORE = "memory_store.json"  # 记忆落盘文件 / on-disk memory store


def extract_facts(dialog: str) -> list[str]:
    """用 LLM 从对话中抽取值得长期记住的用户事实。

    Args:
        dialog: 本次会话的完整对话文本 / full dialog text.

    Returns:
        事实字符串列表 / a list of fact strings.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # 强制 JSON / force JSON mode
        messages=[
            {"role": "system", "content": (
                "从对话中抽取关于用户的、值得长期记住的事实（背景/偏好/目标），"
                '输出 JSON：{"facts": ["...", ...]}。没有可记的就返回空数组。'
            )},
            {"role": "user", "content": dialog},
        ],
    )
    return json.loads(resp.choices[0].message.content).get("facts", [])


def embed(texts: list[str]) -> list[list[float]]:
    """批量取 embedding / batch-embed texts."""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def remember(facts: list[str]) -> None:
    """事实 + 向量追加落盘（真实系统还需去重/合并，见任务 3）。"""
    if not facts:
        return
    store = json.load(open(STORE)) if os.path.exists(STORE) else []
    for text, vec in zip(facts, embed(facts)):
        store.append({"text": text, "embedding": vec})
    json.dump(store, open(STORE, "w"), ensure_ascii=False)


def recall(query: str, k: int = 3) -> list[str]:
    """按余弦相似度取 top-k 记忆。时间 O(N*D) 空间 O(N)。"""
    if not os.path.exists(STORE):
        return []
    store = json.load(open(STORE))
    mat = np.array([m["embedding"] for m in store])  # 形状 (N, D)
    q = np.array(embed([query])[0])                  # 形状 (D,)
    # 向量化算余弦相似度，不写 for 循环 / vectorized cosine similarity
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q))
    return [store[i]["text"] for i in np.argsort(-sims)[:k]]


if __name__ == "__main__":
    # ── 会话 1：只负责"记" / session 1: write only ──────────────
    dialog = (
        "user: 我是做 Java 后端的，正在转型 AI Agent 开发。\n"
        "user: 向量库我定了 pgvector，别再给我推荐别的了。\n"
        "assistant: 好，后续示例都按 pgvector 来。"
    )
    remember(extract_facts(dialog))

    # ── 会话 2：全新"进程"也记得你 / session 2: recall & inject ──
    question = "帮我设计一个文档问答系统的存储方案"
    memories = recall(question)
    system = "你是编程助教。已知该用户的长期信息：\n- " + "\n- ".join(memories)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
    )
    print("注入的记忆 / injected memories:", memories)
    print("\n回答 / answer:\n", resp.choices[0].message.content)
```

跑两遍看效果：第一遍生成 `memory_store.json`（打开看看抽出了什么事实）；回答里应主动使用 pgvector、并按"Java 工程师转型"的口吻给方案——**这些信息并不在第二次的 user 消息里**，全部来自注入的记忆。

## 3. 今日任务

1. 跑通 `ex02_memory.py`，确认 `memory_store.json` 里的事实合理、会话 2 的回答确实用上了记忆。
2. **验证"跨会话"**：把 `__main__` 拆成 `session1.py` / `session2.py` 两个文件分别运行——物理上两个进程，记忆仍然生效。
3. **实现最小"遗忘/更新"**：新增一段对话"我改主意了，向量库换成 Milvus"，在 `remember` 写入前先 `recall` 相似旧记忆，把新旧事实一起交给 LLM 判定 `ADD / UPDATE / DELETE`，据此改写 store。再问存储方案，确认答案已换成 Milvus 且不再提 pgvector。
4. （选做）给每条记忆加 `created_at` 时间戳，检索时打印出来——这就是最朴素的情节记忆雏形。

**验收标准**：任务 2 中两个独立进程共享记忆；任务 3 中矛盾的旧记忆被更新而不是与新记忆并存。

## 4. 自测清单

- [ ] 我能说清短期记忆和长期记忆分别存在哪、生命周期有何不同。
- [ ] 我能各举一例语义记忆和情节记忆，并说明哪类该带时间戳。
- [ ] 我能讲出记忆系统的三个动作（写入 / 检索 / 遗忘更新），以及漏掉"遗忘"会出什么 bug。
- [ ] 我能用一句话说清记忆和 RAG 的本质区别（状态 vs 知识），而不只是"都用向量检索"。
- [ ] 我知道 mem0 和 Letta 各自的核心思想，也知道它们的 API 需要以当天官方文档为准。

## 5. 延伸 & 关联

- **本系列相关讲义**：
  - 短期记忆与上下文裁剪：[Day-11-memory-and-context.md](Day-11-memory-and-context.md)
  - 同线程状态持久化（checkpointer ≠ 长期记忆）：[Day-29-checkpointer-persistence.md](Day-29-checkpointer-persistence.md)
  - embedding 与余弦相似度（今天检索的地基）：[Day-16-embedding-basics.md](Day-16-embedding-basics.md)
  - top-k 检索与完整 RAG（对照"记忆 vs RAG"表）：[Day-20-retrieval-topk.md](Day-20-retrieval-topk.md)、[Day-21-full-rag-chain.md](Day-21-full-rag-chain.md)
  - 记忆注入也会带来提示注入面（用户可"投毒"自己的记忆）：[Day-53-prompt-injection.md](Day-53-prompt-injection.md)
- **主课程相关章节**：
  - RAG 体系（外部知识一侧的完整版）：[../07-llm-applications/03-rag/](../07-llm-applications/03-rag/)
  - LangChain 基础（其 memory 抽象与今天的手写版对照）：[../07-llm-applications/05-langchain/01-langchain-basics.md](../07-llm-applications/05-langchain/01-langchain-basics.md)
- **进阶阅读方向**：MemGPT 论文（"LLM as OS"的记忆分层思想，Letta 的源头）；生产化时把今天的 JSON 换成 pgvector（Day 17）即可平滑升级，检索从 O(N) 线性扫描变成索引查询。
