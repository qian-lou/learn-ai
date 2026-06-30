# Day 13 · ☕ Java 对照日 1：Spring AI 把工具映射成带 `@Tool` 的 Bean

> **今日目标**：用你最熟的 Spring Boot 重做 Day 8 的多工具 Agent——把工具写成带 `@Tool`/`@ToolParam` 注解的 Spring Bean，用 `ChatClient` 跑通工具调用。
> **时长**：~2h ｜ **前置**：Day 6~12（已懂工具调用原理）+ 基本 Spring Boot
> **今日产出**：一个 Spring Boot 工程，`ChatClient` 持有 2~3 个 `@Tool` 方法，运行后能自动调工具回答问题。

## 1. 为什么 & 是什么

你已经在 Python 侧把工具调用吃透了。今天换 **Spring AI**（1.0 GA，2025 年随 Spring Boot 3.4+ 落地）做同一件事——目的不是学新概念，而是看清：**同一套 tool calling 机制，在 Java 生态里长什么样**。对你这种 Java 后端，这才是能直接落进现有系统的形态。

最大的认知映射——**Python 的"装饰器生成 schema"，在 Spring 里是"注解 + 反射生成 schema"**：

| Python（Day 7~8） | Spring AI 等价物 | 说明 |
|---|---|---|
| `@function_tool` 装饰器 | `@Tool` 注解（标在方法上） | 都靠"声明式标注"把普通方法暴露成工具 |
| 函数 docstring | `@Tool(description=...)` | 给模型看的工具说明 |
| 参数类型注解 | 方法参数 + `@ToolParam(description=...)` | 反射读取参数类型/描述生成 schema |
| `Agent(tools=[...])` | `ChatClient...tools(myToolsBean)` | 把工具 Bean 挂到 ChatClient 上 |
| `Runner.run(...)` | `chatClient.prompt().user(...).call()` | 框架托管整个工具调用循环 |

一句话：**Spring AI 用注解 + Spring 容器，把 Day 8 那套手写循环收进了 `ChatClient`。** 工具就是 Spring Bean 里的普通方法，加个 `@Tool` 就被框架自动暴露给模型——非常"Spring 味"。

## 2. 跟着做（Hands-on）

**Step 1 — 依赖与配置**

```xml
<!-- pom.xml：引入 Spring AI BOM + OpenAI starter / Spring AI deps -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-starter-model-openai</artifactId>
</dependency>
```

```yaml
# application.yml：从环境变量读 key，绝不硬编码 / read key from env, never hardcode
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      chat:
        options:
          model: gpt-4o-mini
```

**Step 2 — 把工具写成带 `@Tool` 的 Bean**

```java
package com.example.agent;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 业务工具集 / a set of business tools exposed to the model.
 * 每个 @Tool 方法等价于 Day 8 的一个 Python 工具 + 其 schema。
 */
@Service
public class QueryToolService {

    /** 城市人口表（模拟）/ fake population table. */
    private static final Map<String, Integer> POPULATION = Map.of(
            "北京", 2184, "上海", 2487);

    /** 用户表（模拟只读数据源）/ fake read-only user store. */
    private static final List<Map<String, Object>> USERS = List.of(
            Map.of("id", 1, "name", "张三", "city", "北京"),
            Map.of("id", 2, "name", "李四", "city", "上海"));

    /** 查询城市人口。description 就是模型看到的工具说明 / the model reads it. */
    @Tool(description = "查询城市常住人口（单位：万人）")
    public int getPopulation(@ToolParam(description = "城市名，如 北京") String city) {
        return POPULATION.getOrDefault(city, 0);  // getOrDefault 避免 NPE / NPE-safe
    }

    /** 按城市查询用户列表（模拟数据库）/ query users by city. */
    @Tool(description = "按城市查询用户列表")
    public List<Map<String, Object>> queryUsersByCity(
            @ToolParam(description = "城市名，如 北京") String city) {
        // 时间 O(N) 空间 O(K)：线性过滤 / linear filter
        return USERS.stream()
                .filter(u -> Objects.equals(u.get("city"), city))
                .toList();
    }
}
```

注意对照 Day 8：你**没写一行 JSON Schema**。`@Tool` + `@ToolParam` 让 Spring AI 用反射读方法签名，自动生成那份 schema——和 Python `@function_tool` 从签名+docstring 生成是**同一个套路**，只是换了语言机制。

**Step 3 — 把工具挂到 ChatClient 并调用**

```java
package com.example.agent;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

/**
 * 演示运行：ChatClient 自动托管工具调用循环 / runs the tool-calling loop.
 */
@Component
public class AgentRunner implements CommandLineRunner {

    private final ChatClient chatClient;

    /** 构造注入 ChatClient.Builder 与工具 Bean / constructor injection. */
    public AgentRunner(ChatClient.Builder builder, QueryToolService tools) {
        // 把工具 Bean 挂到默认 ChatClient，等价 Python 的 Agent(tools=[...])
        this.chatClient = builder.defaultTools(tools).build();
    }

    @Override
    public void run(String... args) {
        // prompt().user().call() 内部跑完整三段式：决定→调工具→回填→答复
        // the call() drives the full Day-6 loop under the hood
        String answer = chatClient.prompt()
                .user("北京有哪些用户？另外北京有多少人口？")
                .call()
                .content();
        System.out.println(answer);
    }
}
```

`mvn spring-boot:run`：`ChatClient` 内部自动让模型决定调 `queryUsersByCity("北京")` 与 `getPopulation("北京")`、回填、综合作答——**整个 Day 8 的循环被 Spring AI 收进了 `.call()`**。你只声明工具，编排归框架。

## 3. 今日任务

1. 起一个最小 Spring Boot 工程，跑通上面三块，确认 `ChatClient` 能自动调到两个工具并作答。
2. **加第三个工具**：仿照写一个 `calculator`（Java 侧也**别用脚本引擎 eval 不可信输入**；用安全表达式库如 `exp4j`，或限定只接受数字运算）。
3. **观察自动 schema**：打开 DEBUG 日志（`logging.level.org.springframework.ai=DEBUG`），找到 Spring AI 发给模型的工具 schema，对照 Day 6 你手写的那份。
4. **横向对比**：列一张你自己的"Python `@function_tool` ↔ Spring `@Tool`"映射表，写下两者在"定义工具"上的体感差异。

**验收标准**：Spring Boot 工程能用 `@Tool` Bean 完成工具调用并作答；新增的第三个工具生效且未使用 eval；你能在日志里指认出框架自动生成的工具 schema。

## 4. 自测清单

- [ ] 我能把 `@Tool`/`@ToolParam` 对应到 Python 的 `@function_tool` + 类型注解/docstring。
- [ ] 我理解 Spring AI 用注解 + 反射自动生成工具 schema（无需手写 JSON Schema）。
- [ ] 我会用 `ChatClient.Builder.defaultTools(bean)` 把工具挂上去。
- [ ] 我知道 `chatClient.prompt()....call()` 内部跑的就是 Day 6 的三段式循环。
- [ ] 我在 Java 侧也坚持"绝不 eval 不可信输入"的安全红线。

## 5. 延伸 & 关联

- **LangChain4j 也是一条路**：用 `@Tool` 注解 + `AiServices` 构建，理念几乎一致。Spring AI 的优势是和 Spring 生态（DI、配置、Actuator）天然贴合——对你这种 Java 后端最顺手。
- 明天 Day 14 接着讲 Spring AI 的 **Advisor 模式**（横切关注点：记忆、日志、RAG），那是它相比 Python SDK 最"Spring"的设计。
- 关联章节：
  - Python 侧多工具 Agent（今天的对照原型）：[../agent-course/Day-08-multi-tool-agent.md](./Day-08-multi-tool-agent.md)
  - 用 SDK 定义第一个工具（`@function_tool`）：[../agent-course/Day-07-agents-sdk-first-tool.md](./Day-07-agents-sdk-first-tool.md)
  - API 服务化（把 Agent 包成 Spring 服务）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
