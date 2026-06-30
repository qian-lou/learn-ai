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

## 5. 例题（Worked Examples）

### 例题 1：实现一个通用的参数类型验证装饰器 / Implement a generic parameter type validation decorator

该例题展示如何编写一个装饰器，检查被装饰函数的输入参数类型是否与 Python 的类型提示（Type Hints）一致。

```python
import functools
import inspect
from typing import Any, Callable, Dict, get_type_hints

def validate_types(func: Callable[..., Any]) -> Callable[..., Any]:
    """类型检查装饰器 / Type validation decorator.
    
    Time: O(P) - P 为参数个数 / P is the number of parameters.
    Space: O(P) - 存储参数映射 / Store parameter mapping.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 获取函数签名与绑定的参数 / Get function signature and bound arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 获取类型提示 / Get type hints
        hints = get_type_hints(func)
        
        # 验证各参数类型 / Validate each parameter type
        for name, value in bound_args.arguments.items():
            if name in hints:
                expected_type = hints[name]
                # 简单类型验证 / Simple type validation
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"参数 '{name}' 期望类型 {expected_type}, 但收到 {type(value)}"
                    )
                    
        return func(*args, **kwargs)
    return wrapper

# 测试用例 / Test case
@validate_types
def add_user(user_id: int, username: str) -> str:
    return f"User {username} (ID: {user_id}) added."

# 正常调用 / Valid call
print(add_user(1001, "Alice"))

# 异常调用触发 TypeError / Invalid call triggers TypeError
try:
    add_user("1001", "Bob")  # user_id 传入了 str
except TypeError as e:
    print(f"捕获异常 / Caught expected error: {e}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：实现一个 `@timer` 装饰器，统计函数的执行时间并以秒为单位打印。
*参考答案*：
```python
import time
import functools
from typing import Any, Callable

def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Time: O(1), Space: O(1)"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[{func.__name__}] 耗时 / Elapsed: {end_time - start_time:.6f}s")
        return result
    return wrapper
```

### 进阶题
**练习 2**：实现一个带参数的装饰器 `@retry(max_retries=3)`，当被装饰的函数抛出异常时自动重试，直至达到最大重试次数。
*参考答案*：
```python
import functools
import time
from typing import Any, Callable

def retry(max_retries: int = 3) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"重试中 / Retrying {attempt + 1}/{max_retries}: {e}")
                    time.sleep(0.1)
        return wrapper
    return decorator
```\n