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

## 4. 详细推理（Deep Dive）

```
asyncio 事件循环 vs Java 线程池:

Python asyncio:
  - 单线程事件循环（不受 GIL 影响，因为是 IO 等待）
  - 适合 I/O 密集型（网络请求、文件读写）
  - 不适合 CPU 密集型（用 multiprocessing）

Java 线程池:
  - 多线程真并行
  - 适合 CPU 和 I/O 密集型
  - 线程开销大（虚拟线程改善了这点）

Python 并发选择:
  IO 密集 → asyncio（协程，最轻量）
  CPU 密集 → multiprocessing（多进程，绕过 GIL）
  混合场景 → concurrent.futures（线程/进程池）
```

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
**练习 2：** 用 Semaphore 限制最大并发数为 3。

### 进阶题
**练习 3：** 实现异步批量 LLM 推理，支持进度条和超时控制。
**练习 4：** 实现一个异步任务队列，支持优先级和重试。
