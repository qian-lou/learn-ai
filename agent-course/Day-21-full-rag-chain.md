# Day 21 · 完整 RAG 链路：检索 → 拼 Prompt → 生成

> **今日目标**：把检索结果拼进提示词、交给 LLM 生成答案，跑出你**第一个能用的 RAG**，并用 LCEL 把整条链优雅地串起来。
> **时长**：~2h ｜ **前置**：Day 20（检索可用）、Day 1（LLM 调用）
> **今日产出**：一个 `day21_rag.py`，输入一句问题，自动检索→拼上下文→生成答案，端到端输出一段基于库内资料的回答。

## 1. 为什么 & 是什么

前五天你把"建库 + 检索"打通了。今天接上最后一棒：把召回的 chunk **拼进提示词**当作"参考资料"，让 LLM **基于这些资料**回答。这就是 RAG 的完整闭环：**Retrieve（检索）→ Augment（增强提示）→ Generate（生成）**。核心只有一句话——**别让模型凭记忆答，让它读着你给的资料答**。

给 Java 工程师的贴切类比：

| RAG 链路 | Java 世界类比 | 说明 |
|---|---|---|
| Retriever | `Repository`（语义版查询） | 按相似度查出相关资料 |
| 拼 context 进 prompt | 把查询结果组装进响应模板 | 检索结果 → 提示词的"参考资料"段 |
| LLM 生成 | `Service` 把数据加工成结果 | 基于上下文产出自然语言答案 |
| LCEL 的 `\|` 管道 | Stream/责任链 `.then().then()` | 用 `\|` 把"检索→格式化→提示→模型→解析"串成一条链 |
| RunnablePassthrough | 把原始入参透传到下游 | 让 question 既进检索、又进 prompt |

两个关键认知：

- **Prompt 模板是"行为契约"**：好的 RAG 提示会**明确约束**模型——"只依据下面的资料回答；资料中没有就说不知道；不要编造"。这几句话直接决定幻觉率（Day 22 深入）。提示词不是装饰，是控制器。
- **LCEL（LangChain Expression Language）是 2026 主流写法**：用 `|` 运算符把各环节串成 `Runnable` 链，天然支持流式、批处理、异步，还能整条 `.invoke()`。比起手写一堆中间变量，LCEL 让"数据怎么流"一目了然——很像 Java Stream 的 `map().filter().collect()`。

> **2026 选型**：检索仍走 Day 17 的 pgvector（这里用 `langchain_postgres.PGVector` 封装，省去手写 SQL）；嵌入 `bge-m3`；生成用任意 chat 模型（示例用 OpenAI，换 Claude/Qwen 只改一行）。整条链用 LCEL 组装。

## 2. 跟着做（Hands-on）

**Step 1 — 装包**

```bash
pip install langchain langchain-openai langchain-huggingface \
            langchain-postgres "psycopg[binary]>=3.2" "sentence-transformers>=3.0"
export OPENAI_API_KEY="sk-你的key"   # 生成模型用 / for the generator
```

**Step 2 — 用 LCEL 串完整 RAG 链**

```python
"""Day 21: 完整 RAG 链（LCEL）/ end-to-end RAG chain with LCEL."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

CONN = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


def build_store() -> PGVector:
    """初始化向量库并灌入示例文档 / init vector store with sample docs."""
    store = PGVector(
        embeddings=emb, collection_name="rag_demo",
        connection=CONN, use_jsonb=True,
    )
    store.add_documents([
        Document(page_content="pgvector 是 Postgres 的扩展，给它加上 vector 列类型，可直接做向量检索。"),
        Document(page_content="RAG 通过检索外部知识增强大模型回答，先检索相关资料再生成。"),
        Document(page_content="余弦相似度衡量两个向量夹角，值越接近 1 语义越相近。"),
    ])
    return store


def format_docs(docs: list[Document]) -> str:
    """把召回的多个块拼成一段参考资料 / join retrieved chunks into context."""
    return "\n\n".join(f"[资料{i + 1}] {d.page_content}" for i, d in enumerate(docs))


# 提示模板：明确"只依据资料作答" / prompt that constrains the model to the context
PROMPT = ChatPromptTemplate.from_template(
    "你是严谨的助手。只依据下面的参考资料回答问题；"
    "若资料中没有答案，就回答'根据现有资料无法回答'，不要编造。\n\n"
    "参考资料：\n{context}\n\n问题：{question}\n\n回答："
)


def main() -> None:
    """组装并运行 RAG 链 / assemble and run the RAG chain."""
    store = build_store()
    retriever = store.as_retriever(search_kwargs={"k": 3})  # top-3 检索 / top-3
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # 低温更稳 / deterministic

    # LCEL：question 既进检索（取 context）又透传进 prompt
    # LCEL: question feeds both the retriever (→context) and the prompt
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    for q in ["pgvector 是用来做什么的？", "RAG 的工作流程是怎样的？"]:
        print(f"\nQ: {q}\nA: {chain.invoke(q)}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python day21_rag.py
```

预期：两个问题都能得到**基于库内资料**的准确回答（pgvector 用途、RAG 流程）。如果你再问一个库里没有的问题（如"如何配置 Nginx？"），模型会按提示约束回答"根据现有资料无法回答"——这就是 RAG 在"压幻觉"。

> 看懂这条链：`{...}` 这个 dict 是并行步——`retriever | format_docs` 算出 `context`，`RunnablePassthrough()` 把原问题透传成 `question`，两者一起喂给 `PROMPT`，再 `| llm | StrOutputParser()` 生成并取出字符串。

## 3. 今日任务

1. 跑通 `day21_rag.py`，确认两个库内问题都能基于资料正确作答。
2. **测库外问题**：问一个库里完全没有的问题，确认模型按提示回答"根据现有资料无法回答"，而不是瞎编——感受提示词对幻觉的约束力。
3. **改提示对比**：把提示里"只依据资料、不要编造"那几句删掉，再问库外问题，观察模型是否开始凭记忆乱答——亲眼看到"提示词就是控制器"。

**验收标准**：库内问题答得准且贴合资料；库外问题在严格提示下被拒答；删除约束后模型行为明显变"放飞"，由此你能讲清提示约束的作用。

## 4. 自测清单

- [ ] 我能完整说出 RAG 的 Retrieve→Augment→Generate 三步。
- [ ] 我能读懂这条 LCEL 链里数据是怎么流的（context/question 如何并入 prompt）。
- [ ] 我理解提示词里"只依据资料/不编造"对幻觉率的直接影响。
- [ ] 我知道为什么生成环节常把 temperature 设低（求稳）。
- [ ] 我跑通了端到端 RAG，并验证了库外问题被拒答。

## 5. 延伸 & 关联

- 这条链还很"裸"：答案没带出处。明天 Day 22 就给它加**引用与溯源**，让每句话都能指回是哪条资料。
- LCEL 进阶：整条链支持 `.stream()` 流式输出、`.batch()` 批量、`.ainvoke()` 异步，生产里很有用。
- 本仓库已有的相关章节：
  - RAG 实战（完整可用的 RAG 应用）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - LangChain 完整应用（LCEL 链式编排）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
