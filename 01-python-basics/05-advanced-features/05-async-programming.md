# 异步编程（CompletableFuture 对比）
# Async Programming (vs CompletableFuture)

## 1. 背景（Background）

> **为什么要学这个？**
>
> Python 的 `asyncio` 类似 Java 的 `CompletableFuture` + 虚拟线程，用于 I/O 密集型任务。在大模型应用中，并发调用 LLM API、批量请求 Embedding 服务、流式输出都需要异步编程。
>
> 对于 Java 工程师来说，`async/await` 语法直观明了，但底层是**单线程事件循环**（不是多线程），更接近 Node.js 的模型。

## 2. 知识点（Key Concepts）

| Java | Python | 说明 |
|------|--------|------|
| `CompletableFuture` | `asyncio.Future` | 异步结果 |
| `async` (虚拟线程) | `async def` | 定义协程 |
| `await` | `await` | 等待结果 |
| `ExecutorService` | `asyncio.gather()` | 并发执行 |
| `CompletableFuture.allOf()` | `asyncio.gather()` | 等待全部 |
| `Semaphore` | `asyncio.Semaphore` | 并发控制 |

## 3. 内容（Content）

### 3.1 基础异步

```python
import asyncio

# ============================================================
# 定义异步函数 / Define async function
# ============================================================
async def fetch_data(url: str) -> str:
    """模拟异步 API 调用 / Simulate async API call."""
    print(f"开始请求: {url}")
    await asyncio.sleep(1)  # 模拟网络延迟（不阻塞事件循环）
    return f"data from {url}"

# ============================================================
# 并发执行 / Concurrent execution
# ============================================================
async def main():
    # 类似 Java CompletableFuture.allOf()
    results = await asyncio.gather(
        fetch_data("api/users"),
        fetch_data("api/models"),
        fetch_data("api/configs"),
    )
    # 1 秒完成 3 个请求（并发，不是顺序！）
    for r in results:
        print(r)

asyncio.run(main())
```

### 3.2 并发控制

```python
# ============================================================
# Semaphore 限制并发数 / Limit concurrency
# ============================================================
async def batch_process(urls: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_fetch(url: str) -> str:
        async with semaphore:  # 最多 5 个并发
            return await fetch_data(url)
    
    tasks = [limited_fetch(url) for url in urls]
    return await asyncio.gather(*tasks)

# ============================================================
# 超时控制 / Timeout control
# ============================================================
async def fetch_with_timeout(url: str, timeout: float = 5.0) -> str:
    try:
        result = await asyncio.wait_for(fetch_data(url), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        return f"Timeout: {url}"
```

### 3.3 LLM 应用中的异步

```python
# ============================================================
# 并发调用 LLM API / Concurrent LLM API calls
# ============================================================
import aiohttp

async def call_llm(session: aiohttp.ClientSession, prompt: str) -> str:
    """异步调用 LLM API."""
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
    ) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]

async def batch_inference(prompts: list[str]) -> list[str]:
    """批量并发推理 / Batch concurrent inference."""
    async with aiohttp.ClientSession() as session:
        tasks = [call_llm(session, p) for p in prompts]
        return await asyncio.gather(*tasks)

# 10 个请求并发执行，比串行快 10x
# results = asyncio.run(batch_inference(prompts))
```

### 3.4 流式消费（逐 token 打印）

LLM 的「打字机效果」靠的是 **SSE 流式响应**：服务端把答案切成一个个 `data:` 事件推来，客户端边收边显示。异步天然适合——`async for` 让协程在等下一个 chunk 时把事件循环让给别人。对标 Java：这就是 WebFlux 的 `Flux<String>` / SSE，`async for` ≈ `flux.subscribe(...)` 逐元素回调，但语法是线性的。

```python
# ============================================================
# 流式消费 LLM 响应 / Stream-consume LLM response
# 需要 API key：export OPENAI_API_KEY=sk-...
# pip install openai>=1.0
# ============================================================
import asyncio
from openai import AsyncOpenAI

async def stream_chat(prompt: str) -> str:
    """逐 token 消费流式响应，边到边打印 / Consume stream token-by-token."""
    client = AsyncOpenAI()  # 自动读取环境变量 OPENAI_API_KEY
    full = []
    # stream=True 返回一个异步迭代器 / returns an async iterator
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    # 关键：async for 在等下一个 chunk 时让出事件循环 / yields loop while awaiting next chunk
    async for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)  # 边到边打印，不等整条 / print as it arrives
        full.append(delta)
    print()
    return "".join(full)

# asyncio.run(stream_chat("用一句话解释事件循环"))
# 输出会像打字机一样逐字出现，而非等 3 秒后整段蹦出
```

要点：`resp` 本身是异步迭代器，`async for` 每取一个 chunk 就 `await` 一次网络读——**每个让出点都是切换其他协程的机会**。若同时开多路流，多个 `stream_chat` 可交错推进（下一节的并发同样适用于流）。

### 3.5 结构化并发：TaskGroup（Python 3.11+）

裸 `gather` 有个生产坑：一个子任务抛异常，**其余任务不会被取消**，继续在后台跑（浪费配额、日志错乱）。Python 3.11 引入 `asyncio.TaskGroup` 实现「结构化并发」——**任一子任务失败，自动取消同组其余任务**，且整组作为一个单元向上抛错。类比 Java 21 的 `StructuredTaskScope.ShutdownOnFailure`：一败俱取消。

```python
# ============================================================
# TaskGroup 重写批量推理 / Rewrite batch inference with TaskGroup
# ============================================================
import asyncio

async def batch_infer_tg(prompts: list[str]) -> list[str]:
    """结构化并发：一个失败自动取消其余 / one fails -> cancel the rest."""
    results: list[str] = [""] * len(prompts)
    # async with 退出时才等全组完成 / group awaited on block exit
    async with asyncio.TaskGroup() as tg:
        for i, p in enumerate(prompts):
            # create_task 立即调度；闭包捕获 i 写回对应槽位
            tg.create_task(_fill(results, i, p))
    # 走到这里说明全部成功；任一失败会在 async with 出口抛 ExceptionGroup
    return results

async def _fill(out: list[str], idx: int, prompt: str) -> None:
    out[idx] = await call_llm_single(prompt)  # 见下

async def call_llm_single(prompt: str) -> str:
    await asyncio.sleep(0.1)                   # 占位：真实为 API 调用 / stub
    if prompt == "boom":
        raise ValueError("模型拒答 / model refused")
    return f"answer: {prompt}"

# 对比语义：
# gather(...)                      → task2 失败，task1/task3 仍在后台跑完
# TaskGroup + 三个 create_task     → task2 失败，task1/task3 被 cancel，整组抛 ExceptionGroup

async def demo_taskgroup():
    try:
        await batch_infer_tg(["a", "boom", "c"])
    except* ValueError as eg:  # except* 专门解包 ExceptionGroup（3.11+）
        print(f"整组失败，已取消其余任务 / group cancelled: {eg.exceptions}")

# asyncio.run(demo_taskgroup())
```

超时同样有了现代写法。旧的 `asyncio.wait_for(coro, timeout)` 只能包一个协程；`asyncio.timeout()` 是**上下文管理器**，能给「一整块代码」设总预算，超时后自动取消块内所有 `await`：

```python
# ============================================================
# asyncio.timeout() 上下文写法（3.11+）/ timeout context
# ============================================================
async def infer_with_budget(prompts: list[str]) -> list[str]:
    async with asyncio.timeout(5.0):          # 整块 5 秒总预算 / whole-block budget
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(call_llm_single(p)) for p in prompts]
        return [t.result() for t in tasks]
    # 超时会抛 TimeoutError，并取消块内所有未完成任务 / cancels all on timeout
```

### 3.6 批量容错：让「一个失败」不拖垮整批

生产批处理的第二个坑：默认 `gather` 里一个请求抛异常，**整个 `gather` 立即抛错、已完成结果全丢**。两种容错模式各有适用场景。

**模式 A — `gather(return_exceptions=True)`：** 异常不再中断，而是作为结果元素返回，逐个甄别成功/失败。适合「要拿到完整对齐的结果列表、失败项占位」。

```python
# ============================================================
# 模式 A：限流 + 重试 + 收集异常 / throttle + retry + collect
# 并发度上限 = concurrency；总任务数 N，时间 O(N / concurrency * 单请求耗时)
# ============================================================
import asyncio, random

async def call_with_retry(prompt: str, sem: asyncio.Semaphore,
                          max_retry: int = 3) -> str:
    async with sem:                                  # 限流：同时最多 sem 个在飞
        for attempt in range(1, max_retry + 1):
            try:
                return await call_llm_single(prompt)
            except Exception as e:
                if attempt == max_retry:
                    raise                            # 用尽重试，抛给上层收集
                # 指数退避 + 抖动，避免重试风暴 / backoff + jitter
                await asyncio.sleep(2 ** attempt * 0.1 + random.random() * 0.1)
    raise RuntimeError("unreachable")

async def batch_tolerant(prompts: list[str], concurrency: int = 5) -> list:
    sem = asyncio.Semaphore(concurrency)
    tasks = [call_with_retry(p, sem) for p in prompts]
    # return_exceptions=True：失败项以异常对象出现在结果里，不中断整批
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    bad = [r for r in results if isinstance(r, Exception)]
    print(f"成功 {len(ok)} / 失败 {len(bad)}")
    return results  # 与输入一一对齐 / positionally aligned with prompts
```

**模式 B — `asyncio.as_completed`：** 谁先完成先拿谁，适合「边完成边落盘/边刷进度」的流式收尾，不必等最慢的那个。

```python
# ============================================================
# 模式 B：按完成顺序消费 / consume in completion order
# ============================================================
async def batch_stream_done(prompts: list[str], concurrency: int = 5):
    sem = asyncio.Semaphore(concurrency)
    tasks = [call_with_retry(p, sem) for p in prompts]
    # as_completed 返回一个「谁先好谁先出」的迭代器 / yields as each finishes
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            print(f"完成一条: {result}")   # 可在此处立即写库/刷新进度条
        except Exception as e:
            print(f"跳过失败项: {e}")      # 单点失败不影响其余继续
```

选型：**要对齐结果**用 A（`return_exceptions`）；**要尽早消费/低延迟落盘**用 B（`as_completed`）；**要一败全停、干净收场**用 3.5 的 `TaskGroup`。

## 4. 详细推理（Deep Dive）

### 4.1 单线程为何能扛住高并发 I/O？

核心洞察：**I/O 密集型任务的绝大部分时间在「等」，而不是在「算」**。一次 LLM 请求可能等 2 秒网络、只花几微秒解析。多线程为了这 2 秒的等待各占一个线程栈（Python 线程约 8MB 栈 + GIL 抢占），成本极高。

事件循环换了个思路——**用「等待」的空档去推进别的任务**：

```
一个协程调 await 时会发生什么（机制推导）:

  1. 协程执行到 `await session.get(url)`
  2. 底层把这个 socket 注册到 OS 的 I/O 多路复用器
     （epoll/kqueue），然后【把控制权交还事件循环】—— 这就是「让出点」
  3. 事件循环不空等：立刻挑另一个「就绪」的协程接着跑
  4. 网络数据回来后，epoll 通知事件循环「这个 socket 好了」
  5. 事件循环把对应协程标记为就绪，下一轮调度它从 await 处【恢复】

  → 单线程串行地推进 N 个协程，但因为「等」的时间被互相填满，
    墙上时钟 ≈ 最慢那个请求，而非 N 个之和。
```

对标 Java：这正是 Netty/WebFlux 的 Reactor 模型——少量 event-loop 线程 + 非阻塞 I/O。Java 21 虚拟线程则是另一条路（保留阻塞写法、由 JVM 在底层挂起载体线程），殊途同归，都是「别让线程干等」。

### 4.2 让出点到底在哪？——只有 `await` 会让出

这是新手最大的误区：**协程不是随时被抢占的，只有执行到 `await`（且该 await 真的要等）时才让出控制权**。这是「协作式调度」，区别于线程的「抢占式调度」。

```python
import asyncio

async def bad():
    total = 0
    for i in range(10**8):   # ← 纯 CPU 循环，中间【没有 await】
        total += i           #   事件循环被彻底霸占，其他协程全饿死
    return total
# 后果：这段跑几秒，同一循环里的所有其他协程一个都推进不了

async def cpu_bound(n: int) -> int:
    return sum(range(n))     # 同样是纯 CPU，不该直接在协程里跑

async def good():
    # to_thread 把阻塞/CPU 活儿丢到线程池，await 处让出事件循环
    return await asyncio.to_thread(cpu_bound, 10**8)
```

推论：`await asyncio.sleep(0)` 是一个「主动让出」的惯用法——它不真等，只是给事件循环一次调度其他协程的机会（相当于协作式的 `yield`）。

### 4.3 CPU 密集为什么必须 `to_thread` / 多进程？

因为 4.2：CPU 密集代码没有 `await`，会**卡死整个事件循环**。解法取决于活儿的性质：

```
选型决策（按「等 vs 算」区分）:

  纯等待的 I/O（网络/磁盘）      → asyncio 协程          最轻量，天生适配
  阻塞式库调用（老 SDK、无 async）→ asyncio.to_thread(fn)  丢线程池，await 让出
  CPU 密集且【释放 GIL】的库
    （numpy/torch 底层是 C）      → to_thread 也够          C 层不占 GIL，真并行
  CPU 密集且【纯 Python】
    （手写循环、正则、纯 Python 解析）→ ProcessPoolExecutor   多进程绕过 GIL
```

关键分水岭是 **GIL**：`to_thread` 起的是真线程，但纯 Python 计算受 GIL 串行化、并发无收益；只有当阻塞发生在 C 扩展里（numpy 矩阵、torch 前向、文件 syscall）——这些代码会**释放 GIL**——线程才真正并行。纯 Python 的 CPU 活儿只能靠多进程各自独立解释器来绕开 GIL。

> 一句话记忆：**`await` 是给「等」用的让出点；`to_thread` 是给「算」用的逃生舱；多进程是「纯 Python 硬算」的最后手段。**

（注：Python 3.13 起有实验性 free-threaded / no-GIL 构建，但 2026 生产默认仍是带 GIL 的解释器，上述选型依然成立。）

## 5. 例题（Worked Examples）

```python
# 异步生产者-消费者
async def producer(queue: asyncio.Queue):
    for i in range(10):
        await queue.put(f"task_{i}")
    await queue.put(None)  # 结束信号

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"处理: {item}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1：** 用 `asyncio.gather` 并发下载 5 个 URL。

*参考答案*：
```python
import asyncio

async def download(url: str) -> str:
    await asyncio.sleep(1)          # 模拟 I/O / simulate I/O
    return f"ok: {url}"

async def main() -> list[str]:
    urls = [f"http://site/{i}" for i in range(5)]
    # gather 并发等待全部 / concurrently await all, ~1s total not 5s
    return await asyncio.gather(*(download(u) for u in urls))

print(asyncio.run(main()))
```

**练习 2：** 用 Semaphore 限制最大并发数为 3。

*参考答案*：
```python
import asyncio

async def fetch(url: str, sem: asyncio.Semaphore) -> str:
    async with sem:                 # 同时最多 3 个进入 / at most 3 concurrently
        await asyncio.sleep(1)
        return f"ok: {url}"

async def main() -> list[str]:
    sem = asyncio.Semaphore(3)      # 并发上限 / concurrency cap
    urls = [f"http://site/{i}" for i in range(10)]
    return await asyncio.gather(*(fetch(u, sem) for u in urls))

asyncio.run(main())
```

### 进阶题
**练习 3：** 实现异步批量 LLM 推理，支持进度条和超时控制。

*参考答案*：
```python
import asyncio
from tqdm.asyncio import tqdm_asyncio   # 异步进度条 / async progress bar

async def call_llm(prompt: str, timeout: float = 10.0) -> str:
    try:
        # wait_for 实现单请求超时 / per-request timeout
        return await asyncio.wait_for(_infer(prompt), timeout=timeout)
    except asyncio.TimeoutError:
        return "TIMEOUT"

async def batch_infer(prompts: list[str], concurrency: int = 5) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    async def task(p: str) -> str:
        async with sem:
            return await call_llm(p)
    # tqdm_asyncio.gather 在完成时刷新进度 / updates bar as tasks finish
    return await tqdm_asyncio.gather(*(task(p) for p in prompts))
```

**练习 4：** 实现一个异步任务队列，支持优先级和重试。

*参考答案*：
```python
import asyncio

async def worker(q: asyncio.PriorityQueue, max_retry: int = 3) -> None:
    while True:
        priority, payload = await q.get()      # 数值越小优先级越高 / lower = higher
        for attempt in range(1, max_retry + 1):
            try:
                await handle(payload)           # 业务处理 / do work
                break
            except Exception:
                if attempt == max_retry:
                    print(f"丢弃 / drop: {payload}")
                else:
                    await asyncio.sleep(2 ** attempt)  # 指数退避重试 / backoff retry
        q.task_done()

# q = asyncio.PriorityQueue(); await q.put((0, "urgent")); await q.put((10, "low"))
```
