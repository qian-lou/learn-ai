# 装饰器（AOP 对比）
# Decorators (vs AOP)

## 1. 背景（Background）

> 装饰器是 Python 的核心特性之一，功能类似 Java 的 AOP（面向切面编程）或注解（`@Transactional`, `@Cacheable`）。本质上是"函数的函数"——接收一个函数，返回增强后的函数。在大模型开发中，`@torch.no_grad()` 是最常见的装饰器。

## 2. 知识点（Key Concepts）

| Java 注解/AOP | Python 装饰器 |
|--------------|--------------|
| `@Transactional` | 自定义 `@transactional` |
| `@Cacheable` | `@functools.lru_cache` |
| `@Timed` | 自定义 `@timer` |
| Spring AOP 切面 | 装饰器函数 |

## 3. 内容（Content）

```python
import functools
import time

# 基本装饰器（计时器）/ Basic decorator (timer)
def timer(func):
    """测量函数执行时间的装饰器 / Decorator to measure function execution time."""
    @functools.wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时: {elapsed:.4f}s")
        return result
    return wrapper

@timer  # 语法糖，等价于 train = timer(train)
def train(epochs: int) -> None:
    time.sleep(0.1 * epochs)

train(3)  # train 耗时: 0.3004s

# 带参数的装饰器 / Decorator with parameters
def retry(max_retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"重试 {attempt + 1}/{max_retries}: {e}")
        return wrapper
    return decorator

@retry(max_retries=5)
def unstable_api_call():
    ...

# 内置实用装饰器 / Built-in useful decorators
from functools import lru_cache

@lru_cache(maxsize=128)  # 类似 Java @Cacheable
def fibonacci(n: int) -> int:
    """Time: O(N) with cache, Space: O(N)"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

## 4. 详细推理（Deep Dive）

- 装饰器本质：`@decorator` 是 `func = decorator(func)` 的语法糖
- `@functools.wraps` 必须使用，否则装饰后的函数丢失 `__name__`、`__doc__`
- 类也可以作为装饰器（实现 `__call__` 方法）

## 5-6. 例题/习题

**练习：** 实现一个 `@validate_types` 装饰器，自动检查函数参数类型是否匹配 Type Hints。
