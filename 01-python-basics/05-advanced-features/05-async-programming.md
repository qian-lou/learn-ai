# 异步编程（CompletableFuture 对比）
# Async Programming (vs CompletableFuture)

## 1. 背景（Background）

> Python 的 `asyncio` 类似 Java 的 `CompletableFuture` + 虚拟线程，用于 I/O 密集型任务。在大模型应用中，异步 API 调用（如并发请求 LLM API）是常见场景。

## 2. 知识点（Key Concepts）

| Java | Python |
|------|--------|
| `CompletableFuture` | `asyncio.Future` |
| `async` (虚拟线程) | `async def` |
| `await` | `await` |
| `ExecutorService` | `asyncio.gather()` |

## 3. 内容（Content）

```python
import asyncio

# 定义异步函数 / Define async function
async def fetch_data(url: str) -> str:
    """模拟异步 API 调用 / Simulate async API call."""
    print(f"开始请求: {url}")
    await asyncio.sleep(1)  # 模拟网络延迟
    return f"data from {url}"

# 并发执行多个异步任务 / Concurrent async tasks
async def main():
    # 类似 Java CompletableFuture.allOf()
    results = await asyncio.gather(
        fetch_data("api/users"),
        fetch_data("api/models"),
        fetch_data("api/configs"),
    )
    # 1 秒完成 3 个请求（并发，不是顺序）
    for r in results:
        print(r)

# 运行 / Run
asyncio.run(main())

# 实际场景：并发调用 LLM API
# async def batch_inference(prompts: list[str]) -> list[str]:
#     async with aiohttp.ClientSession() as session:
#         tasks = [call_llm(session, p) for p in prompts]
#         return await asyncio.gather(*tasks)
```

## 4. 详细推理（Deep Dive）

- `asyncio` 是单线程事件循环（类似 Node.js），不受 GIL 影响
- 适合 I/O 密集型（网络请求、文件读写），不适合 CPU 密集型
- CPU 密集型用 `multiprocessing`（多进程）或 C 扩展

## 5-6. 例题/习题

**练习：** 用 `asyncio` 实现一个并发文件下载器，同时下载 5 个文件，限制最大并发数为 3。
