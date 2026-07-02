# Day 18 · 文档切分（Chunking）：固定 vs 语义 + overlap

> **今日目标**：搞懂"为什么要切文档、怎么切、切多大、为什么要 overlap"，亲手对同一篇文档用两种策略切分并对比。
> **时长**：~2h ｜ **前置**：Day 16~17（embedding & 向量库）
> **今日产出**：一个 `day18_chunk.py`，对一段长文本分别用"固定大小+overlap"和"语义切分"两种方式切块，打印块数、每块长度与边界，直观对比。

## 1. 为什么 & 是什么

embedding 模型有输入上限，且**一个向量代表的语义越聚焦，检索越准**。把整篇 10 页文档压成一个向量，等于把一本书的主旨塞进一句话——查什么都"沾点边但都不准"。所以 RAG 必须先把文档**切成块（chunk）**，每块单独嵌入、单独检索。Chunking 是 RAG 里**最影响效果、又最被低估**的一步。

给 Java 工程师的贴切类比：

| Chunking 世界 | Java 世界类比 | 说明 |
|---|---|---|
| chunk（文档块） | 一条数据库记录 / 一个缓存条目 | 检索和召回的**最小单位**，粒度决定召回精度 |
| chunk size | 缓存条目的大小/字段长度设计 | 太大→噪声多、太小→语义碎，需权衡 |
| overlap（重叠） | 滑动窗口的步长 < 窗口宽度 | 相邻块共享一段文字，避免"答案正好被切断在边界" |
| 固定大小切分 | 按字节数 `split` 字符串 | 简单、快、不挑内容，但可能切断句子 |
| 语义切分 | 按业务语义（如段落/章节）拆分 | 沿自然边界切，块更"完整"，但更复杂 |

三个核心权衡：

- **chunk size（块大小）**：常见 200~800 token（中文约 300~1000 字）。**太大**：一个块塞进多个主题，向量被"平均"，检索召回一堆无关内容当陪衬，还浪费上下文窗口。**太小**：句子被切碎，单块缺乏足够上下文，模型看不懂。没有万能值，**取决于文档类型**（FAQ 短、技术手册中、法律条文长）。
- **overlap（重叠）**：相邻块重叠 10~20%（如块 500 字、overlap 80 字）。目的是**防止答案被边界切断**——如果关键句正好横跨两块，没有 overlap 就两块都只有半句。代价是存储和检索量略增。
- **固定 vs 语义**：
  - **固定大小**（`RecursiveCharacterTextSplitter`）：按一组分隔符（段落→句→词）递归切，尽量在自然边界断开，是**工程默认首选**——稳、快、够用。
  - **语义切分**（semantic chunking）：先对句子逐句嵌入，在"相邻句相似度骤降"处下刀，让每块主题内聚。效果常更好，但慢、依赖 embedding，适合对质量要求高的场景。

> **2026 实践**：绝大多数项目从 `RecursiveCharacterTextSplitter`(chunk_size≈500, overlap≈80) 起步，**先把链路跑通再调参**。语义切分作为"召回不理想时的进阶手段"。别一上来就上花活。

## 2. 跟着做（Hands-on）

**Step 1 — 装包**

```bash
pip install langchain-text-splitters langchain-experimental \
            langchain-huggingface "sentence-transformers>=3.0"
```

**Step 2 — 两种切分对比**

```python
"""Day 18: 固定 vs 语义切分对比 / fixed-size vs semantic chunking."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# 一段含两个明显主题切换的文本 / a text with two clear topic shifts
DOC = (
    "RAG 通过检索外部知识来增强大模型的回答能力。它先把文档切块、嵌入、入库，"
    "再在查询时召回相关块拼进提示词。这样模型就能基于真实资料作答，减少幻觉。"
    "另一方面，Postgres 是一款成熟的开源关系型数据库，支持事务、复杂查询和扩展。"
    "通过 pgvector 扩展，它还能直接存储和检索高维向量，省去单独运维向量库的成本。"
)


def show(name: str, chunks: list[str]) -> None:
    """打印切分结果概览 / print a summary of the chunks."""
    print(f"\n=== {name}：{len(chunks)} 块 / chunks ===")
    for i, c in enumerate(chunks):
        # 只打印长度和首尾，避免刷屏 / show length + head/tail only
        print(f"[{i}] len={len(c):3d} | {c[:18]}…{c[-12:]}")


def main() -> None:
    """对同一文档跑两种切分策略 / run both strategies on the same doc."""
    # 策略一：固定大小 + overlap（工程默认）/ fixed size + overlap (default)
    fixed = RecursiveCharacterTextSplitter(
        chunk_size=80,      # 演示用小块，真实项目常 300~800 / small for demo
        chunk_overlap=16,   # 约 20% overlap / ~20% overlap
        separators=["\n\n", "。", "，", ""],  # 优先在句号/逗号断 / prefer sentence breaks
    )
    show("固定大小+overlap", fixed.split_text(DOC))

    # 策略二：语义切分（在语义跳变处下刀）/ semantic chunking
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    semantic = SemanticChunker(
        emb,
        breakpoint_threshold_type="percentile",
        # 默认断句正则 (?<=[.?!])\s+ 只认英文标点+空白；中文无空格会被当成单句。
        # 显式传中文标点断句，SemanticChunker 才能逐句嵌入并在主题跳变处下刀。
        sentence_split_regex=r"(?<=[。！？])",
    )
    show("语义切分", semantic.split_text(DOC))


if __name__ == "__main__":
    main()
```

运行：

```bash
python day18_chunk.py
```

预期：**固定大小**会切出若干长度相近的块（约 80 字、相邻块尾首重叠）；**语义切分**通常切成 **2 块**——正好在"RAG 主题 → Postgres 主题"的跳变处断开。你能直观看到语义切分"沿主题边界下刀"的特点。

> **中文必踩的坑**：`SemanticChunker` 默认断句正则是 `(?<=[.?!])\s+`，只认**英文标点 + 空白**。中文用『。』分句且句间无空格，会被整段当成**单句**——单句输入时 `SemanticChunker` 直接原样返回（只有 1 块，甚至不会真正调 embedding），主题边界根本切不出来。所以上面**显式传了** `sentence_split_regex=r"(?<=[。！？])"`，让它按中文标点先拆句，再逐句嵌入、比相似度。处理中文文本时这一步不能省。

> 注意：语义切分会对每句调 embedding，比固定切分慢得多。小文档无所谓，大批量要掂量成本。

## 3. 今日任务

1. 跑通 `day18_chunk.py`，确认两种策略的块数和边界差异（语义切分应在主题切换处断开）。
2. **调 overlap 看效果**：把固定切分的 `chunk_overlap` 从 16 改到 0，再改到 40，观察相邻块的重叠文字变化——理解 overlap 在"防答案被切断"中的作用。
3. **调 chunk_size 找手感**：把 `chunk_size` 从 80 改到 40（更碎）再到 160（更整），观察块数与每块的完整度，写一句你对"这篇文档多大合适"的判断。

**验收标准**：两种策略都能切出块并打印长度/边界；overlap=0 时相邻块不再重叠、overlap=40 时重叠明显变长；你能说出语义切分为什么恰好切在主题边界，以及它为什么更慢。

## 4. 自测清单

- [ ] 我能解释"为什么不能整篇文档当一个 chunk"。
- [ ] 我知道 chunk size 太大/太小各自的坏处。
- [ ] 我能说清 overlap 解决什么问题、代价是什么。
- [ ] 我能区分固定大小切分和语义切分的原理与适用场景。
- [ ] 我跑通了对比，并对"这类文档该切多大"有了初步判断。

## 5. 延伸 & 关联

- chunking 没有银弹：表格、代码、Markdown 各有专用 splitter（如 `MarkdownHeaderTextSplitter` 按标题切，保留章节层级）。
- 进阶玩法："父子块"（small-to-big）——用小块检索保精度、召回后返回其所属大块给模型保上下文，Day 23 会提到类似思路。
- 本仓库已有的相关章节：
  - RAG 实战（含 chunking 策略优化对比）：[../07-llm-applications/03-rag/03-rag-practice.md](../07-llm-applications/03-rag/03-rag-practice.md)
  - RAG 基础概念（五步流程里的"分割"步）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
