# Day 22 · 引用与溯源：让答案带出处、不知道就说不知道

> **今日目标**：给 RAG 答案加上**可点的出处**，并用提示工程 + 结构化输出把"不知道就说不知道"做成硬约束，系统性压低幻觉。
> **时长**：~2h ｜ **前置**：Day 21（能用的 RAG）、Day 4（结构化输出/Pydantic）
> **今日产出**：一个 `day22_cite.py`，回答时返回 `{answer, citations[]}` 结构，每条 citation 指向具体来源；库外问题则明确拒答。

## 1. 为什么 & 是什么

能答还不够——**企业级 RAG 的命门是"可信"**。用户凭什么信你这段答案？答案来自哪份文档的哪一段？答错了能不能追溯？没有引用的 RAG 是"看起来像对的黑箱"，没人敢拿它做决策。今天解决两件事：**溯源（每句话指回原文）**和**降幻觉（没依据就别答）**。

给 Java 工程师的贴切类比：

| 引用/溯源世界 | Java/工程世界类比 | 说明 |
|---|---|---|
| citation（出处） | 日志里的 traceId / 审计字段 | 让结果**可追溯**，出问题能定位到源 |
| 带 metadata 的 chunk | 实体的外键 + 来源字段 | Day 19 存的 source/chunk_no 此刻派上用场 |
| "不知道就说不知道" | 查无结果返回 404 而非编造一条 | 宁可空，不可错 |
| 结构化引用输出 | 强类型 DTO（`{answer, citations}`） | 用 Pydantic 校验，杜绝"自由文本里夹出处" |
| grounding（接地） | 断言"输出必须可由输入推导" | 答案的每个论点都要能在召回资料里找到依据 |

三个核心手段：

- **溯源靠元数据**：Day 19 你把 `source/chunk_no` 一起入库不是白存的——检索召回时把这些元数据**带回来**，拼进提示让模型在答案里**标注用了哪条资料**，最后输出里附上 citations。这就是为什么"建库阶段就要存好元数据"。
- **降幻觉靠提示 + 阈值 + 结构**三件套：① 提示里**硬性要求**"只用资料、无依据就答不知道、每个论断标注来源"；② Day 20 的**相似度阈值**先在检索层拦掉库外问题（召回为空就直接拒答，根本不喂给模型）；③ 用**结构化输出**强制返回 `{answer, citations}`，citations 为空意味着"无据"。三层叠加，幻觉空间被挤得很小。
- **"无据可查"是合法答案**：训练有素的 RAG **敢说不知道**。这反直觉但极重要——一个会拒答的系统比一个永远给答案的系统**可信得多**。

> **2026 实践**：引用做法有粗有细。**粗粒度**：答案末尾列出参考的来源列表（够用、好实现）。**细粒度**：每个句子/论点后标 `[1][2]` 内联引用（体验好、实现复杂，需让模型对齐句子与来源）。今天先做**结构化的来源级引用**（answer + 来源列表），它是性价比最高的一档。

## 2. 跟着做（Hands-on）

**Step 1 — 装包**

```bash
pip install langchain langchain-openai langchain-huggingface \
            langchain-postgres "psycopg[binary]>=3.2" "sentence-transformers>=3.0" pydantic
export OPENAI_API_KEY="sk-你的key"
```

**Step 2 — 带引用 + 拒答的 RAG**

```python
"""Day 22: 带引用与溯源的 RAG / RAG with citations and grounding."""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

CONN = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


class CitedAnswer(BaseModel):
    """带引用的结构化答案 / structured answer with citations."""

    answer: str = Field(description="基于资料的回答；无依据则为'根据现有资料无法回答'")
    citations: list[str] = Field(
        default_factory=list, description="所引用资料的来源标识列表；无依据则为空",
    )


def build_store() -> PGVector:
    """灌入带 source 元数据的文档 / seed docs carrying source metadata."""
    store = PGVector(embeddings=emb, collection_name="cite_demo",
                     connection=CONN, use_jsonb=True)
    store.add_documents([
        Document(page_content="pgvector 通过 <=> 运算符做余弦距离检索。",
                 metadata={"source": "pgvector_guide.md#3"}),
        Document(page_content="RAG 用相似度阈值拦截库外问题以降低幻觉。",
                 metadata={"source": "rag_notes.md#7"}),
    ])
    return store


def answer(query: str, k: int = 3, min_sim: float = 0.4) -> CitedAnswer:
    """检索→（阈值拦截）→结构化生成带引用的答案 / retrieve, gate, then cite.

    Args:
        query: 用户问题 / user question.
        k: 召回块数 / number of chunks.
        min_sim: 相似度阈值，低于此值视为'库外' / below this = out-of-KB.

    Returns:
        CitedAnswer：含 answer 与 citations / answer plus its sources.
    """
    store = build_store()
    # 带分数检索，便于阈值判断 / retrieve with scores for thresholding
    hits = store.similarity_search_with_relevance_scores(query, k=k)
    kept = [(d, s) for d, s in hits if s >= min_sim]

    # 检索层拒答：没料就别喂给模型 / refuse at retrieval layer if nothing relevant
    if not kept:
        return CitedAnswer(answer="根据现有资料无法回答", citations=[])

    # 拼上下文，并把 source 一并交给模型 / build context with sources
    context = "\n".join(
        f"[{d.metadata['source']}] {d.page_content}" for d, _ in kept
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # with_structured_output 强制返回 CitedAnswer / force the typed schema
    structured = llm.with_structured_output(CitedAnswer)
    return structured.invoke(
        "只依据下列资料回答，并在 citations 中列出实际引用的 source；"
        "若资料不足以回答，answer 写'根据现有资料无法回答'且 citations 为空。\n\n"
        f"资料：\n{context}\n\n问题：{query}"
    )


def main() -> None:
    """对比库内问题与库外问题 / in-KB vs out-of-KB."""
    for q in ["pgvector 怎么算相似度？", "如何在 AWS 上配置 VPC？"]:
        r = answer(q)
        print(f"\nQ: {q}\nA: {r.answer}\n出处 / citations: {r.citations}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day22_cite.py
```

预期：第一个问题返回准确答案，且 `citations` 里出现 `pgvector_guide.md#3`；第二个**库外**问题返回 `根据现有资料无法回答`、citations 为空。**答案可溯源、无据则拒答**——RAG 从"能用"升级到"可信"。

## 3. 今日任务

1. 跑通 `day22_cite.py`，确认库内问题带出正确出处、库外问题被拒答且 citations 为空。
2. **验证溯源真实性**：检查返回的 citation 是否确实对应那条召回资料的 `source`（别让模型编出一个不存在的出处）——如发现不符，收紧提示词。
3. **调阈值看拒答边界**：把 `min_sim` 从 0.4 调到 0.7，观察更多"沾边但不够相关"的问题也被拒答——体会阈值是"敢说不知道"的旋钮。

**验收标准**：库内问答附带真实出处；库外问题稳定拒答；调高阈值后系统更"保守"（更倾向拒答）；你能讲清"提示+阈值+结构化"三层是如何协同压幻觉的。

## 4. 自测清单

- [ ] 我能说清为什么企业级 RAG 必须带引用/溯源。
- [ ] 我知道溯源依赖建库阶段存好的 source 元数据。
- [ ] 我能列出降幻觉的三件套（提示约束 + 检索阈值 + 结构化输出）。
- [ ] 我理解"敢说不知道"为什么让系统更可信。
- [ ] 我跑通了带引用的 RAG，并验证了出处真实、库外拒答。

## 5. 延伸 & 关联

- 内联引用（句末标 `[1][2]`）是进阶：需让模型把每个论点对齐到具体来源，可用更细的提示或后处理校验"答案里的每个声明是否有出处支撑"。
- 防幻觉还有"自检"一招：生成后再让模型核对"答案是否完全由资料支撑"，不支撑就降级为拒答——成本翻倍但更稳。
- 本仓库已有的相关章节：
  - 结构化输出基础（Pydantic 强约束）：[./Day-04-structured-output.md](./Day-04-structured-output.md)
  - RAG 实战（生成质量控制与引用）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
