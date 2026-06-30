# Day 14 · ☕ Java 对照日 2：Advisor 模式 vs Python SDK

> **今日目标**：吃透 Spring AI 的 **Advisor** 设计——把"记忆/日志/RAG"等横切关注点做成可插拔拦截器，并对比 Python SDK 的取舍。
> **时长**：~2h ｜ **前置**：Day 11（记忆）、Day 13（Spring AI 工具）
> **今日产出**：一个给 `ChatClient` 挂上 `MessageChatMemoryAdvisor` + 自定义日志 Advisor 的工程，能多轮记忆并打印每次调用；外加一张"两生态优劣"对照表。

## 1. 为什么 & 是什么

Day 11 你在 Python 侧**手写**了记忆管理（自己维护 `messages`、自己截断）。Spring AI 给了另一种范式：**Advisor**——把这类"每次模型调用前后都要做的横切逻辑"抽成可插拔的拦截器链。

对 Java 工程师这是**送分题**——它就是你天天用的 **AOP / 拦截器 / Filter Chain**：

| Spring AI Advisor | 你早就会的 Java 概念 | 作用 |
|---|---|---|
| `Advisor` 接口 | Servlet `Filter` / Spring `HandlerInterceptor` | 在请求"经过"模型前后插逻辑 |
| Advisor 链 | `FilterChain` / AOP 拦截器栈 | 多个 Advisor 按 order 串成链 |
| `MessageChatMemoryAdvisor` | 一个负责"自动续历史"的拦截器 | 替你做 Day 11 的记忆维护 |
| `QuestionAnswerAdvisor` | 一个负责"检索并塞上下文"的拦截器 | RAG 的横切实现（Phase 2 会用） |
| 自定义 Advisor | 自定义 Filter（鉴权/日志/限流） | 日志、脱敏、token 统计…… |

**这正是两种生态的哲学分水岭**：**Python SDK** 偏显式命令式——记忆、日志你自己写在循环里，控得细但样板多；**Spring AI** 偏声明式面向切面——横切逻辑沉到 Advisor 链，主流程干净但"魔法"藏在框架里。没有谁对谁错。对你这种 Java 背景，**Advisor 让 Agent 的横切关注点和你现有的企业中间件（鉴权/审计/监控）用同一套心智**——这就是 Spring AI 对你最大的价值。

## 2. 跟着做（Hands-on）

接着 Day 13 的工程，给 `ChatClient` 装两个 Advisor：一个**内置记忆**、一个**自定义日志**。

**Step 1 — 自定义一个日志 Advisor（就是写个"Filter"）**

```java
package com.example.agent;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;

/**
 * 简易日志 Advisor：调用模型前后打印，等价于一个 HandlerInterceptor。
 * A logging advisor — like a Servlet Filter around the model call.
 */
public class LoggingAdvisor implements CallAdvisor {

    /** order 越小越先执行 / lower order runs earlier in the chain. */
    private static final int ADVISOR_ORDER = 0;

    @Override
    public String getName() {
        return "LoggingAdvisor";
    }

    @Override
    public int getOrder() {
        return ADVISOR_ORDER;
    }

    /** 环绕模型调用：前置打印 → nextCall 放行 → 后置打印 / wrap the call. */
    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request,
                                         CallAdvisorChain chain) {
        System.out.println("[Advisor] → 即将调用模型 / calling model...");  // 前置 pre
        ChatClientResponse response = chain.nextCall(request);  // 放行 / proceed
        System.out.println("[Advisor] ← 已拿到响应 / got response");  // 后置 post
        return response;
    }
}
```

**Step 2 — 把"记忆 Advisor + 日志 Advisor"挂上 ChatClient**

```java
package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

/** 演示：记忆 Advisor 让多轮对话自动记得，无需手维护 messages / auto-memory. */
@Component
public class AdvisorRunner implements CommandLineRunner {

    private static final String CONV_ID = "demo-user-001";  // 会话 id / conversation key
    private final ChatClient chatClient;

    public AdvisorRunner(ChatClient.Builder builder, QueryToolService tools) {
        // MessageWindowChatMemory 自带滑动窗口（对应 Day 11 截断）/ sliding-window memory
        ChatMemory memory = MessageWindowChatMemory.builder().maxMessages(20).build();
        this.chatClient = builder
                .defaultTools(tools)
                // Advisor 链：记忆 + 日志，按 order 串起来 / the advisor chain
                .defaultAdvisors(
                        MessageChatMemoryAdvisor.builder(memory).build(),
                        new LoggingAdvisor())
                .build();
    }

    @Override
    public void run(String... args) {
        ask("我叫张三，记住我。");              // 第一轮给事实 / turn 1: a fact
        ask("我叫什么名字？北京有多少人口？");  // 第二轮靠记忆 Advisor 续上 / relies on memory
    }

    /** 发一轮并指定会话 id，记忆 Advisor 据此隔离不同用户的历史。 */
    private void ask(String userText) {
        String answer = chatClient.prompt()
                .user(userText)
                // 用 conversationId 标识会话，记忆按它隔离 / scope memory per conversation
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, CONV_ID))
                .call()
                .content();
        System.out.println("A: " + answer);
    }
}
```

`mvn spring-boot:run`：第二轮你**没有手动重发"我叫张三"**，但模型答得出名字——因为 `MessageChatMemoryAdvisor` 在请求经过它时**自动把历史续了进去**（对照 Day 11 你手写的那一切，现在是一个拦截器在背后干）。同时 `LoggingAdvisor` 在每次调用前后打印——这就是 AOP 式横切。

> 对照 Day 11 的体感：Python 侧"记忆"是你**亲手 append + 截断**的明码逻辑；Spring 侧是**挂一个 Advisor** 就自动有了。前者透明可控、后者干净省事——这正是两种生态的取舍缩影。

## 3. 今日任务

1. 跑通 Advisor 版多轮对话，确认第二轮能答出第一轮给的名字（记忆 Advisor 生效）。
2. **加 order 实验**：再写一个 Advisor，调整两者 `getOrder()`，观察执行顺序——验证"Advisor 链按 order 串行"和你理解的 Filter 链一致。
3. **填对照表**：完成下面这张"两生态优劣"表（这是今天的核心产出，面试常被问）：

   | 维度 | Python SDK（命令式） | Spring AI（Advisor/声明式） |
   |---|---|---|
   | 横切逻辑（记忆/日志） | 自己写在循环里 | 挂 Advisor，自动生效 |
   | 可控/可见性 | 高（明码逻辑） | 较低（藏在框架） |
   | 样板代码 | 多 | 少 |
   | 与企业中间件融合 | 需自己搭 | 天然贴合 Spring 生态 |
   | 学习曲线 | 平（就是 Python） | 需懂 Spring + Advisor 心智 |
   | 适合谁 | AI 工程为主、要极致控制 | Java 团队、要落进现有系统 |

4. **写一句话结论**：基于你的背景，回答"什么场景你会选 Spring AI、什么场景选 Python SDK"。

**验收标准**：记忆 Advisor 让多轮对话生效；你能用 order 控制 Advisor 执行顺序；对照表填完整；并能讲清两生态的取舍与各自适用场景。

## 4. 自测清单

- [ ] 我能把 Advisor 类比成 Servlet Filter / Spring 拦截器链。
- [ ] 我理解 `MessageChatMemoryAdvisor` 替我做了 Day 11 手写的记忆维护。
- [ ] 我会写一个自定义 Advisor（前置/后置 + `chain.nextCall` 放行）。
- [ ] 我能说清 Python SDK（命令式/透明）与 Spring AI（声明式/省事）的取舍。
- [ ] 我能基于自身背景判断何时选哪种生态。

## 5. 延伸 & 关联

- **RAG 也是一个 Advisor**：Spring AI 的 `QuestionAnswerAdvisor` 把"检索→拼上下文"做成拦截器，Phase 2（Day 24 Java 对照日）会用到——届时你会再次感到"横切关注点 = Advisor"。
- **双栈预告**：Phase 5（Day 61+）正是你的差异化王牌——Python 编排层 + Java（Spring AI）服务层。今天的 Advisor 心智，那时会派上大用场。
- 关联章节：
  - Python 侧手写记忆（今天的对照）：[../agent-course/Day-11-memory-and-context.md](./Day-11-memory-and-context.md)
  - Spring AI 工具（Day 13 工程基础）：[../agent-course/Day-13-spring-ai-tools.md](./Day-13-spring-ai-tools.md)
  - RAG 基础（Advisor 化的检索增强）：[../07-llm-applications/03-rag/01-rag-basics.md](../07-llm-applications/03-rag/01-rag-basics.md)
