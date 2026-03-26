# 上下文管理器（try-with-resources 对比）
# Context Managers (vs try-with-resources)

## 1. 背景（Background）

> Python 的 `with` 语句等价于 Java 的 try-with-resources（`AutoCloseable`）。用于自动管理资源（文件、数据库连接、GPU 内存等）。

## 2. 知识点（Key Concepts）

| Java | Python |
|------|--------|
| `try (var r = new Resource())` | `with open(f) as r:` |
| `AutoCloseable` | `__enter__` + `__exit__` |
| - | `contextlib.contextmanager` |

## 3. 内容（Content）

```python
# 基础用法 / Basic usage
# Java: try (BufferedReader r = new BufferedReader(new FileReader("f")))
# Python:
with open("file.txt", "r") as f:
    content = f.read()
# 文件自动关闭，即使发生异常

# 自定义上下文管理器 / Custom context manager
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    """计时上下文管理器 / Timer context manager."""
    import time
    start = time.perf_counter()
    yield  # with 代码块在此执行
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")

with timer("训练"):
    # 模拟训练过程
    sum(range(1_000_000))

# 实际场景：PyTorch 禁用梯度计算
# import torch
# with torch.no_grad():  # 上下文管理器！
#     predictions = model(input_data)
```

## 4-6. 推理/例题/习题

**练习：** 实现一个 `DatabaseConnection` 上下文管理器，进入时获取连接，退出时自动关闭，异常时回滚。
