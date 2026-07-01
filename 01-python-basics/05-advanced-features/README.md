# 05-advanced-features — 高级特性

> **所属阶段**：阶段一 · Python 基础
> **学习目标**：掌握 Python 高级编程特性，对标 Java 的 AOP、Stream、泛型、异步——装饰器、生成器、上下文管理器、类型注解、asyncio，写出工程级、高性能代码
> **预估时长**：4-5 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [decorators](./01-decorators.md) | 装饰器 | `@decorator` = `func = decorator(func)` 语法糖；带参装饰器（三层嵌套）；`@functools.wraps` 保留元信息；`@lru_cache`；类装饰器（`__call__`）；对标 AOP/`@Cacheable` |
| 02 | [generators-and-iterators](./02-generators-and-iterators.md) | 生成器与迭代器 | `yield` 惰性产出、可暂停恢复的状态机；只遍历一次；`yield from` 委托；逐行读大文件不爆内存；对标 Java Stream；DataLoader 迭代模式 |
| 03 | [context-managers](./03-context-managers.md) | 上下文管理器 | `with` 自动资源管理；`__enter__`/`__exit__`（返回 True 抑制异常）；`@contextmanager` + yield；`torch.no_grad()`/`autocast`；对标 try-with-resources |
| 04 | [type-hints](./04-type-hints.md) | 类型注解 | `list[str]`/`str \| None`；`TypeVar`/`Generic` 泛型；`Protocol` 结构化类型；mypy 静态检查；Pydantic 运行时校验；对标 Java 泛型 |
| 05 | [async-programming](./05-async-programming.md) | 异步编程 | `async def`/`await`；单线程事件循环（非多线程）；`asyncio.gather` 并发、`Semaphore` 限流、`wait_for` 超时；并发调 LLM API；对标 CompletableFuture |

---

## 🔑 知识点详解

### 01 · 装饰器

- **核心概念**：装饰器是「接收函数、返回增强函数」的高阶函数；`@deco` 只是 `func = deco(func)` 的语法糖，用于把横切逻辑（计时、重试、缓存、鉴权）从业务里剥离。
- **关键 API / 语法**：`@functools.wraps(func)` **必须加**（否则丢失被包函数的 `__name__`/`__doc__`）；带参装饰器是**三层嵌套**（`deco(arg)` 返回真正的装饰器）；`@functools.lru_cache(maxsize=...)` 记忆化；实现 `__call__` 可让类作装饰器。
- **易错点**：① 忘记 `@functools.wraps` 导致栈追踪/文档错乱、依赖 `__name__` 的框架失灵；② 带参与不带参装饰器嵌套层数不同（多一层），写错括号；③ 装饰器在导入时（定义处）就执行外层，副作用要留意。
- **Java 视角**：≈ Spring AOP 切面 / 注解——`@Transactional`≈自定义 `@transactional`、`@Cacheable`≈`@lru_cache`、`@Timed`≈`@timer`；但 Python 装饰器是纯语言机制，无需容器/代理。
- **前置**：03-syntax（闭包、高阶函数、`*args/**kwargs`）。

### 02 · 生成器与迭代器

- **核心概念**：`yield` 把函数变成生成器——一台可暂停/恢复的状态机，每次 `next()` 执行到下一个 `yield` 就冻结整个帧并返回值，实现**惰性求值、内存恒定 O(1)**。
- **关键 API / 语法**：`yield` 产出、`yield from` 委托子生成器（扁平化）；生成器表达式 `(e for ...)`；迭代协议 `__iter__` + `__next__`；`sum(gen)`/`for x in gen` 拉动执行。
- **易错点**：① 生成器**只能遍历一次**，第二次 `list(g)` 得空——需复用就物化成 list；② 无 `len()`、不可索引；③ 惰性导致闭包读「最新值」的延迟绑定坑。
- **Java 视角**：≈ Java 8 Stream 的惰性——声明算子、终端操作（`collect`/`sum`）才触发，且都只消费一次；但生成器可用 `yield` 写任意产出逻辑，比固定算子更通用。大模型里 DataLoader 的分批加载就是迭代器模式。
- **前置**：03-data/04（推导式、生成器表达式）。

### 03 · 上下文管理器

- **核心概念**：`with` 语句保证「进入时获取资源、退出时释放」，即使中途异常也能清理；等价于 Java 的 try-with-resources 但更灵活（可用生成器 3 行创建）。
- **关键 API / 语法**：类实现 `__enter__`（返回资源）+ `__exit__(exc_type, exc_val, exc_tb)`（**返回 True 抑制异常**、False 放行）；或 `@contextmanager` 装饰生成器，`yield` 前是 enter、`yield` 后（放 `finally`）是 exit；支持 `with a() as x, b() as y:` 组合。
- **易错点**：① `@contextmanager` 里若 with 体可能抛异常，清理代码要放 `try/finally` 的 `finally`，否则异常时资源泄漏；② `__exit__` 误返回 True 会「吞掉」异常导致 bug 难查；③ 记录并恢复原状态（如随机种子、模型 train/eval 模式）比硬编码更稳健。
- **Java 视角**：`with ... as` ≈ `try (var r = ...)`；`__enter__` ≈ 构造/获取，`__exit__` ≈ `close()` + catch；PyTorch 的 `torch.no_grad()`/`autocast()`/`profiler` 都是上下文管理器。
- **前置**：02（`@contextmanager` 基于生成器）、04-oop（`__enter__`/`__exit__` 是魔术方法）。

### 04 · 类型注解

- **核心概念**：类型注解是**可选**的、运行时被忽略（不像 Java 强制），价值在 IDE 补全、代码即文档、mypy 静态检查、大型协作可维护性。
- **关键 API / 语法**：容器 `list[str]`/`dict[str, int]`（3.9+）；可空 `str | None`（3.10+）；泛型 `TypeVar("T")` + `Generic[T]`、有界 `TypeVar(bound=...)`；`Protocol` 做结构化类型（鸭子类型 + 检查，无需显式继承）；`Callable[[入参], 返回]`；`type X = ...` 别名（3.12+）；`@runtime_checkable`。
- **易错点**：① 以为注解会在运行时强制类型——不会，需跑 `mypy --strict` 才检查；② 循环导入下用 `if TYPE_CHECKING:` + 字符串注解避免运行时开销；③ Pydantic 的 `BaseModel` 才做**运行时**校验（与纯注解不同）。
- **Java 视角**：`list[str]` ≈ `List<String>`；`str | None` ≈ `Optional<String>`；`TypeVar(bound=...)` ≈ `<T extends Comparable>`；`Protocol` ≈ 结构化接口（近 Go interface）；mypy ≈ 编译期类型检查；Pydantic ≈ Bean Validation。
- **前置**：02-syntax（Type Hints 入门）、04-oop（泛型类）。

### 05 · 异步编程

- **核心概念**：`asyncio` 是**单线程事件循环**（协程协作式切换，非多线程、更接近 Node.js），专治 I/O 密集场景（并发调 LLM API、批量 Embedding、流式输出）。
- **关键 API / 语法**：`async def` 定义协程、`await` 等待、`asyncio.run(main())` 启动；`asyncio.gather(*tasks)` 并发等全部（≈ `allOf`）；`asyncio.Semaphore(n)` 限并发；`asyncio.wait_for(coro, timeout)` 超时；`asyncio.Queue`/`PriorityQueue` 生产者-消费者。
- **易错点**：① 在协程里调用**同步阻塞**函数（如普通 `requests`、`time.sleep`）会卡死整个事件循环——必须用异步库（aiohttp）和 `asyncio.sleep`；② CPU 密集任务放进 asyncio 无并行收益，应改 multiprocessing；③ 忘记 `await` 会得到未执行的协程对象。
- **Java 视角**：`async def` ≈ 返回 `CompletableFuture` 的方法 / 虚拟线程；`await` ≈ `.get()`/`join`（但不阻塞线程）；`gather` ≈ `CompletableFuture.allOf`；`Semaphore` ≈ `java.util.concurrent.Semaphore`；但底层是单线程协程而非线程池。
- **前置**：03-syntax（函数）、02（迭代/惰性思想）；I/O 密集选 asyncio、CPU 密集选 multiprocessing。

---

## 🎯 学习要点

- **把「装饰器/生成器/上下文/协程」对应到 Java 已知概念**：AOP→装饰器、Stream→生成器、try-with-resources→with、泛型→Type Hints、CompletableFuture→asyncio，只学差异，快速迁移。
- **写装饰器永远加 `@functools.wraps`**：这是保留元信息、不破坏栈追踪与框架反射的必备项，属于代码评审必查。
- **能惰性就惰性**：大数据/流式处理用生成器表达式喂给 `sum/any/join`，逐行读文件不爆内存；但记住生成器一次性、不可索引，需复用就物化。
- **资源管理一律走 with**：文件、连接、锁、GPU 状态、随机种子都用上下文管理器保证异常安全，`@contextmanager` 里清理放 `finally`。
- **类型注解 + mypy 当护栏**：为公共函数补全参数/返回注解，CI 里跑 `mypy --strict`；需要运行时校验（API 入参、配置）上 Pydantic。
- **异步只用于 I/O 密集**：并发调 LLM/Embedding 用 asyncio + Semaphore 限流 + wait_for 超时；协程里禁用同步阻塞调用，CPU 密集改多进程。

---

## 🔗 关联

- **上一模块**：[04-oop-in-python](../04-oop-in-python/) — 上下文管理器/迭代器/装饰器都建立在类与魔术方法之上（`@dataclass` 本身即装饰器）。
- **下一阶段**：[02-data-science-fundamentals](../../02-data-science-fundamentals/) — 生成器（数据流）、类型注解、dataclass 会在 NumPy/Pandas 数据管道中继续使用。
- **本阶段总览**：[阶段一 README](../README.md)
- **配套实战**：[agent-course/Day-40 performance](../../agent-course/Day-40-performance.md) 与 [Day-06 tool-calling](../../agent-course/Day-06-tool-calling-basics.md) — 用本模块的 asyncio 并发调用 LLM、装饰器封装重试/限流。
