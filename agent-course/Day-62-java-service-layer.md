# Day 62 · Java 服务层：Spring Boot 提供业务 / 数据 / 鉴权 / 可观测

> **今日目标**：把 Day 61 的契约落成真代码——用 **Spring Boot 3.x** 实现 Java 服务层:分层架构(Controller→Service)、JWT 鉴权、幂等写操作、并暴露 Actuator + Micrometer 可观测端点,供 Python agent 调用。
> **时长**：~2h ｜ **前置**：Day 61(契约已定)、你的 Spring 底子
> **今日产出**：一个能跑的 Spring Boot 服务,`/api/v1/orders/{userId}` 带鉴权可查、`/api/v1/refunds` 幂等可退、`/actuator/prometheus` 暴露指标。

## 1. 为什么 & 是什么(概念 + Java 类比)

这一天**几乎全是你的主场**。这里没有新框架要学,就是把你最熟的 Spring Boot 用对——区别只在于:**这个服务的"调用方"是一个 AI agent,而 AI 调用方是不可信的**。所以三件事要格外较真:

| 关注点 | 普通内部服务 | **被 agent 调用的服务** |
|---|---|---|
| **鉴权** | 内网常常裸调 | 必须强制——agent 可能被注入诱导越权 |
| **幂等** | 锦上添花 | 必须——LLM/编排层会重试,不幂等就重复扣款 |
| **入参校验** | 信任上游 | 零信任——把 agent 传来的每个字段当外部输入校验 |
| **可观测** | 有就行 | 必须带 trace 透传——为 Day 65 两端打通铺路 |

一句话心智:**Java 层是整个双栈系统的"信任根(root of trust)"**。Python 编排层负责"想得灵活",Java 层负责"做得安全可靠"。退款金额、权限、库存这些**确定性约束,全部在 Java 重新校验**,绝不相信 agent 传过来的值。

## 2. 跟着做(Hands-on)

**Step 1 — 依赖(Spring Boot 3.x / Java 21)**

```xml
<!-- pom.xml 关键 starter；micrometer 两项为 Day 65 trace 透传埋点 -->
<dependencies>
  <dependency><groupId>org.springframework.boot</groupId><artifactId>spring-boot-starter-web</artifactId></dependency>
  <dependency><groupId>org.springframework.boot</groupId><artifactId>spring-boot-starter-security</artifactId></dependency>
  <dependency><groupId>org.springframework.boot</groupId><artifactId>spring-boot-starter-validation</artifactId></dependency>
  <dependency><groupId>org.springframework.boot</groupId><artifactId>spring-boot-starter-actuator</artifactId></dependency>
  <dependency><groupId>io.micrometer</groupId><artifactId>micrometer-registry-prometheus</artifactId></dependency>
  <dependency><groupId>io.micrometer</groupId><artifactId>micrometer-tracing-bridge-otel</artifactId></dependency>
</dependencies>
```

**Step 2 — DTO + Controller(严格分层,零信任入参)**

```java
// 退款请求 DTO：Bean Validation 强校验 agent 传来的每个字段；amount 上限/idempotencyKey 幂等键交 Service 把关
public record RefundRequestDTO(
        @NotBlank String orderId,
        @NotNull @DecimalMin("0.0") BigDecimal amount,
        @NotBlank String idempotencyKey) {}
public record RefundResultVO(String refundId, String status, BigDecimal amount) {}

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor  // 构造器注入 final 字段，便于单测
public class OrderController {
    private final OrderService orderService;
    // 查用户订单：鉴权由 SecurityConfig 强制，此处只取已认证主体并防越权
    @GetMapping("/orders/{userId}")
    public ResponseEntity<List<OrderVO>> listOrders(@PathVariable String userId, Authentication auth) {
        boolean admin = auth.getAuthorities().stream().anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
        if (!Objects.equals(auth.getName(), userId) && !admin) {   // 越权防御：只能查自己的
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
        return ResponseEntity.ok(orderService.listByUser(userId));
    }

    // 发起退款：写操作。@Valid 触发 DTO 校验，业务上限/幂等在 Service 内
    @PostMapping("/refunds")
    public ResponseEntity<RefundResultVO> refund(@Valid @RequestBody RefundRequestDTO request) {
        return ResponseEntity.ok(orderService.refund(request));
    }
}
```

**Step 3 — Service:幂等 + 业务校验(信任根的核心)**

```java
@Service
@RequiredArgsConstructor          // 注入 final 依赖：Controller→Service→Manager→DAO
public class OrderService {
    private static final BigDecimal MAX_REFUND = new BigDecimal("10000.00");  // 硬上限，非魔法值
    private final OrderManager orderManager;
    private final RefundDAO refundDAO;

    public List<OrderVO> listByUser(String userId) { return orderManager.findOrders(userId); }

    /** 退款：幂等 + 重新校验金额与订单归属（不信 agent 传来的 amount），越界抛 4xx。*/
    @Transactional
    public RefundResultVO refund(RefundRequestDTO request) {
        // 幂等：同 key 已处理则返回原结果，绝不重复扣款
        Optional<RefundResultVO> done = refundDAO.findByIdempotencyKey(request.idempotencyKey());
        if (done.isPresent()) return done.get();
        // 零信任：金额上限 + 库中实际可退额度，两道都按 DB 重算，agent 的值一律不信
        if (request.amount().compareTo(MAX_REFUND) > 0) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "超过退款上限");
        }
        BigDecimal refundable = orderManager.refundableAmount(request.orderId());
        if (request.amount().compareTo(refundable) > 0) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "退款金额超过可退额度");
        }
        // 事务内落库：执行退款 + 记录幂等键
        RefundResultVO result = orderManager.doRefund(request);
        refundDAO.saveWithKey(request.idempotencyKey(), result);
        return result;
    }
}
```

**Step 4 — 鉴权(JWT,机器对机器)**

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    // 无状态 JWT：agent 携带 bearer token 调用（M2M），纯 API 故关 CSRF
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http.csrf(csrf -> csrf.disable())
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/actuator/health", "/actuator/prometheus").permitAll()
                .anyRequest().authenticated())            // 其余一律需认证
            .oauth2ResourceServer(o -> o.jwt(Customizer.withDefaults()));
        return http.build();
    }
}
```

**Step 5 — 可观测(Actuator + Micrometer,为 Day 65 铺路)**

```yaml
# application.yml：暴露 Prometheus 指标 + 全采样 trace（学习期）
management:
  endpoints.web.exposure.include: health, prometheus, metrics
  tracing.sampling.probability: 1.0          # 全采样，生产应调低
  otlp.tracing.endpoint: http://localhost:4317   # 与 Python 同一后端，Day 65 两端汇合
```

跑起来自测:`mvn spring-boot:run` 后,无 token 请求 `/api/v1/orders/u1` 应 401,带 bearer token 可查,`curl /actuator/prometheus | grep http_server_requests` 能看到指标。

> 工程要点(Alibaba P3C):严格 `Controller→Service→Manager→DAO` 不跨层;DTO/VO/DO 后缀分明;`@Transactional` 保证退款落库与幂等记录的一致性;金额上限抽成 `static final` 常量而非魔法值。这些不是为了"好看",而是这个服务是**信任根**,必须可单测、可审计。

## 3. 今日任务

1. **实现 Day 61 契约的 Java 服务**:至少一个只读查询 + 一个带幂等的写操作,严格分层。
2. **跑通三道关**:①无 token → 401;②越权查他人数据 → 403;③同 `idempotencyKey` 重复写,第二次无副作用(不重复扣款/建单)。
3. **暴露可观测**:`/actuator/prometheus` 能抓到 `http_server_requests`;`application.yml` 配好 OTLP endpoint(Day 65 要用)。
4. **写一个 Service 单测**:"agent 传超上限金额 → 抛 400",证明信任根校验可独立验证。

**验收**:三道关可复现、幂等写经得起重复调用、指标可抓、上述单测通过。

## 4. 自测清单

- [ ] 我能讲清为什么"被 agent 调用的服务"在鉴权/幂等/校验上要格外较真。
- [ ] 我理解 Java 是"信任根":金额/权限/库存在此重新校验,写操作带幂等键扛重试。
- [ ] 我严格遵守了 Controller→Service→Manager→DAO 分层与 DTO/VO 命名。
- [ ] 我配好了 OTLP tracing endpoint,为 Day 65 两端 trace 汇合做了准备。

## 5. 延伸 & 关联

- 本仓库 API 服务开发(服务设计基础):[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
- 本仓库 评估与监控(可观测指标体系):[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 明天 Day 63 写 **Python Agent 层**,用 LangGraph 经 REST/MCP 调用今天这个 Java 服务。
- 本系列总计划:[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
