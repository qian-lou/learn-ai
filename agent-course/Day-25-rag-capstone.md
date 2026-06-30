# Day 25 · 🎯 阶段项目：文档问答系统（带引用）+ RAG 常见坑清单

> **今日目标**：把 Day 16~24 攒下的零件组装成一个**能讲、能演**的里程碑项目——文档问答系统（带引用、会拒答），并沉淀一份 RAG 避坑清单。
> **时长**：~2h ｜ **前置**：Day 16~24（RAG 全链路 + Java 对照）
> **今日产出**：一个 `day25_qa.py`，对外暴露 `ask(question) -> {answer, citations}`：自动检索（含阈值拒答）→生成带出处的答案；附一份你自己的"RAG 踩坑笔记"。

## 1. 为什么 & 是什么

这是**简历高频的第二个里程碑项目**（继 Day 15 的数据查询 Agent）。面试官问"你做过 RAG 吗"，靠的就是今天这个能跑起来、能解释清楚的系统。它把九天所学收拢成一条干净的产品级链路：**建库（ETL）→ 检索（过滤+top-k+阈值）→ 生成（带引用、会拒答）**。

给 Java 工程师的项目分层类比：

| 文档问答系统 | Java 三层架构类比 | 本项目里的实现 |
|---|---|---|
| 嵌入 ETL | 数据初始化/导入层 | `ingest()`：文档→切分→嵌入→pgvector |
| 检索器 | Repository（语义版） | `retrieve()`：metadata 过滤 + top-k + 阈值 |
| 生成器 | Service（业务编排） | `answer()`：拼上下文 + 结构化引用输出 |
| `{answer, citations}` | 对外的 Response DTO | Pydantic 强类型，可校验、可溯源 |

一个合格 RAG 系统的**验收三问**（也是面试官会追问的）：

- **答得准吗**：库内问题能否给出基于资料的正确答案？（检索召回 + 生成约束）
- **可信吗**：每个答案带出处吗？错了能溯源吗？（citations + metadata）
- **稳吗**：库外问题会拒答而不是编造吗？（阈值 + 提示硬约束）

三问全过，才算"能用的 RAG"，而不是"看起来能用的 demo"。

## 2. 跟着做（Hands-on）

把前几天的能力收口成一个最小但完整的服务类。

```bash
pip install langchain langchain-openai langchain-huggingface \
            langchain-postgres "psycopg[binary]>=3.2" "sentence-transformers>=3.0" pydantic
export OPENAI_API_KEY="sk-你的key"
```

```python
"""Day 25: 文档问答系统（带引用、会拒答）/ doc-QA with citations & refusal."""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

CONN = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


class QaResult(BaseModel):
    """对外响应 DTO / response DTO."""

    answer: str = Field(description="基于资料的回答，无依据则为拒答语")
    citations: list[str] = Field(default_factory=list, description="引用来源列表")


class DocQA:
    """文档问答系统：建库 + 检索 + 带引用生成 / minimal end-to-end doc-QA."""

    def __init__(self, top_k: int = 4, min_sim: float = 0.45) -> None:
        """初始化向量库与检索参数 / init store and retrieval knobs."""
        self._top_k = top_k
        self._min_sim = min_sim
        self._store = PGVector(embeddings=_emb, collection_name="doc_qa",
                               connection=CONN, use_jsonb=True)
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def ingest(self, text: str, source: str) -> int:
        """切分→嵌入→入库，块均带 source 元数据 / ETL with source metadata.

        Returns:
            入库块数 / number of chunks stored.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "。", "\n", ""])
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c, metadata={"source": f"{source}#{i}"})
                for i, c in enumerate(chunks)]
        self._store.add_documents(docs)
        return len(docs)

    def ask(self, question: str) -> QaResult:
        """检索（含阈值拒答）→生成带引用答案 / retrieve, gate, then cite."""
        # 带分数检索，阈值拦截库外问题 / score-gated retrieval
        hits = self._store.similarity_search_with_relevance_scores(
            question, k=self._top_k)
        kept = [(d, s) for d, s in hits if s >= self._min_sim]
        if not kept:                                   # 无据 → 拒答 / refuse if empty
            return QaResult(answer="根据现有资料无法回答", citations=[])

        context = "\n".join(f"[{d.metadata['source']}] {d.page_content}"
                            for d, _ in kept)
        structured = self._llm.with_structured_output(QaResult)
        return structured.invoke(
            "只依据下列资料回答，并在 citations 列出实际引用的 source；"
            "资料不足以回答时 answer 写'根据现有资料无法回答'且 citations 为空。\n\n"
            f"资料：\n{context}\n\n问题：{question}")


def main() -> None:
    """演示：灌一篇文档，问库内/库外各一题 / demo: ingest then ask."""
    qa = DocQA()
    qa.ingest(
        "pgvector 是 Postgres 扩展，用 <=> 做余弦距离检索。"
        "RAG 通过相似度阈值拦截库外问题以降低幻觉。",
        source="rag_handbook.md")
    for q in ["pgvector 怎么做相似度检索？", "如何配置 Kafka 集群？"]:
        r = qa.ask(q)
        print(f"\nQ: {q}\nA: {r.answer}\n出处: {r.citations}")


if __name__ == "__main__":
    main()
```

预期：库内问题（pgvector）给出带出处的准确答案；库外问题（Kafka）稳定返回"根据现有资料无法回答"、citations 为空。**这就是你的里程碑 #2**——一个会引用、敢拒答的文档问答系统。

## 3. 今日任务

把上面的骨架**充实成你自己的项目**，并产出一份避坑清单。

1. **灌真实文档**：换成你手头的真材料（一篇技术文档/产品手册/几页 PDF，用 Day 19 的 loader），跑通"灌库→问答"，确认答案带正确出处。
2. **过验收三问**：自测"答得准 / 可信 / 稳"三关——准备 3 个库内问题（应答准且带出处）+ 2 个库外问题（应拒答），全部通过。
3. **写《RAG 常见坑清单》**：结合九天踩的坑，整理成一份笔记（下面给你一份起步清单，至少补充 2 条你自己遇到的）。

**RAG 常见坑清单（起步版，至少自己再加 2 条）：**

| # | 坑 | 症状 | 解法 | 对应 |
|---|---|---|---|---|
| 1 | query 与建库用了**不同 embedding 模型** | 召回全是噪声、相似度异常低 | 建库与检索必须同模型；换模型=全量重嵌 | Day 16 |
| 2 | chunk **过大** | 召回块塞多个主题，答案抓不准 | 调小 chunk_size、加 overlap | Day 18 |
| 3 | chunk **过小** | 单块缺上下文，模型看不懂 | 调大 chunk_size 或用父子块 | Day 18 |
| 4 | **没存元数据** | 答案无法溯源、不能过滤 | 入库就存 source/chunk_no | Day 19/22 |
| 5 | **不设阈值** | 库外问题强行召回→幻觉 | 设相似度阈值，召回空就拒答 | Day 20/22 |
| 6 | top-k **太大** | 噪声淹没信号、token 暴涨 | k=3~5 起步，或召回+rerank 两段式 | Day 20/23 |
| 7 | 提示**不约束** | 模型脱离资料凭记忆乱答 | "只依据资料/无据拒答/标注来源" | Day 21/22 |
| 8 | 只用**纯向量**检索 | 精确词项（型号/错误码）漏召 | 混合检索补关键词路 | Day 23 |

**验收标准**：系统能对真实文档问答、答案带真实出处、库外问题稳定拒答；你产出了一份至少 10 条的 RAG 避坑清单（含 2 条以上亲历坑）；能用 3 分钟把这个项目的架构（ETL/检索/生成）和"防幻觉三件套"讲清楚。

## 4. 自测清单

- [ ] 我的系统对真实文档能答准、带出处、库外会拒答（验收三问全过）。
- [ ] 我能讲清项目三层（ETL/检索/生成）各自的职责与对应 Java 分层。
- [ ] 我能说出"防幻觉三件套"（提示约束 + 检索阈值 + 结构化输出）。
- [ ] 我整理了 RAG 避坑清单，且包含我自己踩过的坑。
- [ ] 我能在 3 分钟内把这个里程碑项目向面试官讲明白。

## 5. 延伸 & 关联

- 下一步加固（Phase 4 会系统学）：给 RAG 加**评估**（RAGAS 测忠实度/相关性）、**可观测**（trace 每次检索与生成）、**缓存**（query/embedding 缓存降本）。
- 进阶检索没用够？回到 Day 23 把 metadata 过滤 + 混合检索 + rerank 接进本项目，对比召回质量提升。
- 本仓库已有的相关章节：
  - RAG 实战（生产级完整应用 + RAGAS 评估）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - LangChain 完整应用（端到端 LCEL 编排）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
  - 上一个里程碑（数据查询 Agent）：[./Day-15-data-query-agent.md](./Day-15-data-query-agent.md)
