# Day 24 · ☕ Java 对照日：Spring AI ETL + QuestionAnswerAdvisor 做 RAG

> **今日目标**：用你最熟的 Spring 生态把前八天的 RAG 重做一遍——Spring AI 的 ETL 三件套灌库、`QuestionAnswerAdvisor` 一行接通检索增强；并瞄一眼 LangChain4j 的 `EmbeddingStore` 写法。
> **时长**：~2h ｜ **前置**：Day 16~23（RAG 全链路）、会 Spring Boot
> **今日产出**：一个 Spring Boot 片段，跑通"文档→切分→嵌入→pgvector"的 ETL，并用 `ChatClient + QuestionAnswerAdvisor` 实现带检索的问答。

## 1. 为什么 & 是什么

你的差异化优势是**双栈**。Python 那套（langchain + LCEL）你已经熟了；今天证明**同一套 RAG 思想在 Java 里同样顺滑**，而且企业里大量后端就是 Spring。Spring AI（2024 年起孵化、2025 年 5 月 1.0 GA）把 RAG 抽象成了非常"Spring 味"的组件：`VectorStore`、`EmbeddingModel`、`Advisor`，依赖注入、自动配置一应俱全。

Python 概念 → Spring AI / LangChain4j 对照：

| RAG 概念 | Python（langchain） | Spring AI | LangChain4j |
|---|---|---|---|
| 嵌入模型 | `HuggingFaceEmbeddings` | `EmbeddingModel` Bean | `EmbeddingModel` |
| 向量库 | `PGVector` | `VectorStore`（`PgVectorStore`） | `EmbeddingStore`（`PgVectorEmbeddingStore`） |
| 切分 | `RecursiveCharacterTextSplitter` | `TextSplitter`（`TokenTextSplitter`） | `DocumentSplitter` |
| ETL 三步 | load→split→embed | `DocumentReader → TextSplitter → VectorStore.add` | `loadDocument → split → store.addAll` |
| RAG 链 | LCEL `retriever\|prompt\|llm` | `ChatClient` + `QuestionAnswerAdvisor` | `EmbeddingStoreContentRetriever` + `AiServices` |
| 引用/溯源 | 自己拼 source | `Document.metadata` + Advisor 模板 | `Metadata` + `ContentInjector` |

两个"Spring 味"亮点：

- **Advisor 模式 = 拦截器/AOP**：`QuestionAnswerAdvisor` 像给 `ChatClient` 装了个**前置拦截器**——请求过来时，它**自动**用你的问题去 `VectorStore` 检索、把召回内容塞进提示、再交给模型。你写的还是"问一句话"，检索增强**对调用方透明**。这正是 Spring 一贯的"横切关注点用 Advisor/拦截器抽走"哲学。Day 14 你已经体会过 Advisor，这次是它的 RAG 版。
- **自动配置省去胶水**：`spring-ai-starter-vector-store-pgvector` 一引入，`PgVectorStore` 自动装配、建表、对接 `EmbeddingModel`。对比 Python 里手动 `register_vector`、手写 SQL，Spring 把这些都"约定优于配置"掉了。

> **2026 选型**：Spring AI 1.0 已 GA，是 Java RAG 的官方主力。两条路线——**Spring AI**（生态正统、和 Spring Boot 无缝）vs **LangChain4j**（更贴近 Python langchain 的 API 手感、模块化）。本节主写 Spring AI，末尾给 LangChain4j 对照片段。

## 2. 跟着做（Hands-on）

**Step 1 — pom 依赖（复用 Day 17 的 pgvector 容器）**

```xml
<!-- 两个 starter：OpenAI 模型 + pgvector 向量库（版本由 Spring AI BOM 统一）-->
<dependency><groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-starter-model-openai</artifactId></dependency>
<dependency><groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-starter-vector-store-pgvector</artifactId></dependency>
```

**Step 2 — ETL：文档→切分→嵌入→入库**

```java
/** RAG 建库服务：把文档灌进 pgvector / ETL service: load docs into pgvector. */
@Service
public class RagEtlService {

    private static final int CHUNK_MAX_TOKENS = 500;  // 切块 token 上限 / token cap

    private final VectorStore vectorStore;            // 构造器注入 / constructor-injected

    public RagEtlService(VectorStore vectorStore) { this.vectorStore = vectorStore; }

    /**
     * 读入资源文件，切分、嵌入并写入向量库。
     *
     * @param resource 待入库的文本/Markdown 资源 / source text resource
     * @return 实际入库的文档块数量 / number of chunks stored
     */
    public int ingest(Resource resource) {
        if (resource == null || !resource.exists()) {       // 防御：空/不存在直接返回 0
            return 0;
        }
        // 1) 读取：Reader 抽取为 Document / extract to Documents
        List<Document> docs = new TextReader(resource).read();
        // 2) 切分：TokenTextSplitter 按 token 切块（带 overlap）/ split by tokens
        List<Document> chunks = new TokenTextSplitter(CHUNK_MAX_TOKENS, 350, 5, 10000, true)
                .apply(docs);
        // 3) 嵌入 + 入库：add 内部自动调用 EmbeddingModel / embeds then persists
        vectorStore.add(chunks);
        return chunks.size();
    }
}
```

**Step 3 — RAG 问答：QuestionAnswerAdvisor 一行接通检索**

```java
/** 文档问答服务：检索增强对调用方透明 / RAG QA, retrieval is transparent. */
@Service
public class RagQaService {

    // 召回 top-k 与相似度阈值（库外问题靠阈值拦截）/ top-k and threshold
    private static final int TOP_K = 4;
    private static final double SIMILARITY_THRESHOLD = 0.5D;

    private final ChatClient chatClient;

    public RagQaService(ChatClient.Builder builder, VectorStore vectorStore) {
        // QuestionAnswerAdvisor = RAG 版拦截器：自动检索→拼提示→生成
        // acts like an interceptor: auto retrieve, augment, then generate
        SearchRequest searchRequest = SearchRequest.builder()
                .topK(TOP_K)
                .similarityThreshold(SIMILARITY_THRESHOLD)
                .build();
        // 1.0 GA：带 SearchRequest 的构造器已移除，配置统一走 builder
        // 1.0 GA: the SearchRequest ctor is gone; configure via the builder
        this.chatClient = builder
                .defaultAdvisors(QuestionAnswerAdvisor.builder(vectorStore)
                        .searchRequest(searchRequest)
                        .build())
                .build();
    }

    /** 基于知识库回答问题（自动检索增强）/ ask, retrieval handled by the advisor. */
    public String ask(String question) {
        // 调用方只管问；检索增强被 Advisor 透明接管 / caller just asks
        return chatClient.prompt().user(question).call().content();
    }
}
```

**Step 4 — 串起来跑（在 `CommandLineRunner` 里注入两个 Service）**

```java
@Override
public void run(String... args) {
    int n = etl.ingest(new ClassPathResource("docs/rag-notes.md")); // 先灌库 / ingest
    System.out.println("入库块数 / chunks = " + n);
    System.out.println(qa.ask("pgvector 是用来做什么的？"));          // 再问答 / ask
}
```

预期：启动后打印入库块数，并基于 `rag-notes.md` 的内容回答 pgvector 的用途。整套 RAG 你**几乎没写检索代码**——`QuestionAnswerAdvisor` 把"检索→增强→生成"全包了，这就是 Spring AI 的抽象威力。

> **LangChain4j 对照（同一件事的另一种手感）**：

```java
// LangChain4j：更接近 Python langchain 的显式组装 / more explicit, langchain-like
EmbeddingStore<TextSegment> store = PgVectorEmbeddingStore.builder()
        .host("localhost").port(5432).database("postgres").user("postgres")
        .password("pass").table("lc4j_docs").dimension(1024).build();
EmbeddingStoreIngestor.ingest(Document.from(text), store);     // 切分+嵌入+入库 / ingest
// 检索器注入 AiServices，检索对调用方透明 / retriever feeds AiServices
ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
        .embeddingStore(store).maxResults(TOP_K).minScore(SIMILARITY_THRESHOLD).build();
// 1.x：ChatLanguageModel 更名为 ChatModel，builder 方法用 chatModel(...)
Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(model).contentRetriever(retriever).build();
String answer = assistant.chat("pgvector 是用来做什么的？");
```

## 3. 今日任务

1. 跑通 Spring AI 版：`ingest` 灌库 → `ask` 问答，确认答案基于 `rag-notes.md` 的内容。
2. **验证阈值拦截**：问一个库里没有的问题，确认 `similarityThreshold` 让 Advisor 召回为空、模型据此"答不出来"而非编造——对应 Python Day 20/22 的阈值拒答。
3. **对照差异记一笔**：写 3 行笔记，对比 `QuestionAnswerAdvisor`（隐式、自动）和 Python LCEL `retriever | prompt | llm`（显式、可控）各自的取舍，以及你更偏好哪种、为什么。

**验收标准**：Spring AI 的 ETL + 问答端到端跑通；阈值能拦住库外问题；你能讲清 Advisor 拦截器模式和 LCEL 显式编排的本质区别（横切自动化 vs 数据流显式可见）。

## 4. 自测清单

- [ ] 我能把 Python RAG 的每个概念映射到 Spring AI 的对应组件。
- [ ] 我理解 `QuestionAnswerAdvisor` 为什么像"RAG 版拦截器/AOP"。
- [ ] 我知道 `spring-ai-starter-vector-store-pgvector` 自动配置省了哪些胶水。
- [ ] 我能说清 Spring AI（正统、隐式）vs LangChain4j（显式、贴近 langchain）的取舍。
- [ ] 我跑通了 Java 版 RAG，并验证了阈值拒答。

## 5. 延伸 & 关联

- 引用/溯源在 Spring AI 里靠 `Document.getMetadata()`，可自定义 Advisor 的提示模板让答案带出处——对应 Python Day 22。
- 进阶检索：Spring AI 也支持 `FilterExpression` 做 metadata 过滤、`RerankModel` 接 reranker，对应 Python Day 23 的三件套。
- 本仓库已有的相关章节：
  - Java 对照日 1（Spring AI 工具映射）：[./Day-13-spring-ai-tools.md](./Day-13-spring-ai-tools.md)
  - Java 对照日 2（Advisor 模式初探）：[./Day-14-spring-ai-advisors.md](./Day-14-spring-ai-advisors.md)
