# 上下文管理器（try-with-resources 对比）
# Context Managers (vs try-with-resources)

## 1. 背景（Background）

> **为什么要学这个？**
>
> Python 的 `with` 语句等价于 Java 的 `try-with-resources`（`AutoCloseable`）。用于自动管理资源——文件、数据库连接、GPU 内存、分布式锁。在大模型开发中，`torch.no_grad()`、`torch.cuda.amp.autocast()` 都是上下文管理器。
>
> 对于 Java 工程师来说，理解 `with` 就是理解 `try-with-resources` 的 Python 版本，但 Python 版本更灵活——可以用装饰器和生成器快速创建。

## 2. 知识点（Key Concepts）

| Java | Python | 说明 |
|------|--------|------|
| `try (var r = new Resource())` | `with open(f) as r:` | 自动资源管理 |
| `AutoCloseable.close()` | `__exit__()` | 退出时调用 |
| - | `__enter__()` | 进入时调用 |
| - | `@contextmanager` | 生成器方式创建 |

## 3. 内容（Content）

### 3.1 基础用法

```python
# ============================================================
# Java 对比 / Java comparison
# ============================================================
# Java: try (BufferedReader r = new BufferedReader(new FileReader("f"))) {
#           content = r.readLine();
#       }  // 自动关闭

# Python:
with open("file.txt", "r") as f:
    content = f.read()
# 文件自动关闭，即使发生异常 / Auto-closed even on exception

# 嵌套 with / Nested with
with open("input.txt") as fin, open("output.txt", "w") as fout:
    fout.write(fin.read())
```

### 3.2 自定义上下文管理器

```python
# ============================================================
# 方式 1：类实现（类似 Java AutoCloseable）
# ============================================================
class DatabaseConnection:
    """数据库连接管理器 / Database connection manager."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = None
    
    def __enter__(self):
        """进入 with 块时调用 / Called when entering with block."""
        print(f"连接数据库: {self.db_url}")
        self.conn = {"url": self.db_url, "active": True}
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用 / Called when exiting with block."""
        if exc_type:
            print(f"异常发生，回滚: {exc_val}")
            # return True  # 返回 True 表示异常已处理
        print("关闭数据库连接")
        self.conn["active"] = False
        return False  # False = 不抑制异常

with DatabaseConnection("localhost:3306") as conn:
    print(f"使用连接: {conn}")


# ============================================================
# 方式 2：@contextmanager 装饰器（更简洁）
# ============================================================
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    """计时上下文管理器 / Timer context manager."""
    start = time.perf_counter()
    yield  # with 代码块在此执行 / with block executes here
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")

with timer("数据预处理"):
    data = [x**2 for x in range(1_000_000)]


@contextmanager
def gpu_memory_tracker():
    """GPU 显存追踪器 / GPU memory tracker."""
    # import torch
    # before = torch.cuda.memory_allocated()
    yield
    # after = torch.cuda.memory_allocated()
    # print(f"GPU 显存变化: {(after - before) / 1024**2:.1f} MB")
```

### 3.3 AI/ML 中常见的上下文管理器

```python
import torch

# ============================================================
# PyTorch 中的上下文管理器
# ============================================================

# 1. 禁用梯度计算（推理时必用）
with torch.no_grad():
    predictions = model(input_data)

# 2. 混合精度训练
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

# 3. 性能分析
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

## 4. 详细推理（Deep Dive）

### 4.1 `__exit__` 参数详解

```
__exit__(self, exc_type, exc_val, exc_tb):
  exc_type: 异常类型（如 ValueError），正常退出为 None
  exc_val:  异常值
  exc_tb:   异常追踪栈
  
  返回 True  → 抑制异常（不再向上传播）
  返回 False → 异常继续传播

类比 Java:
  __enter__ = 构造函数
  __exit__  = close() + catch
```

## 5. 例题（Worked Examples）

```python
# 组合使用
@contextmanager
def temp_seed(seed: int):
    """临时设置随机种子 / Temporarily set random seed."""
    import numpy as np
    state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(state)

with temp_seed(42):
    print(np.random.randn(3))  # 可复现
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现一个 `WorkingDirectory` 上下文管理器，进入时切换目录，退出时恢复。

*参考答案*：
```python
import os
from contextlib import contextmanager

@contextmanager
def working_directory(path: str):
    origin = os.getcwd()        # 记录原目录 / save original cwd
    os.chdir(path)
    try:
        yield                   # with 块在此执行 / with-body runs here
    finally:
        os.chdir(origin)        # 无论是否异常都恢复 / restore even on exception

with working_directory("/tmp"):
    print(os.getcwd())          # /tmp
```

**练习 2：** 用 `@contextmanager` 实现一个文件锁。

*参考答案*：
```python
import fcntl  # Unix/macOS 文件锁 / POSIX advisory lock
from contextlib import contextmanager

@contextmanager
def file_lock(path: str):
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)   # 获取排他锁（阻塞）/ exclusive lock
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # 释放锁 / release; 文件关闭也会自动释放

with file_lock("/tmp/app.lock"):
    print("临界区，独占执行 / critical section")
```

### 进阶题

**练习 3：** 实现 `DatabaseConnection`，支持自动提交/回滚事务。

*参考答案*：
```python
class Transaction:
    """类似 Java try-with-resources + 事务模板 / commit on success, rollback on error."""
    def __init__(self, conn) -> None:
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.conn.commit()      # 无异常则提交 / commit on success
        else:
            self.conn.rollback()    # 有异常则回滚 / rollback on error
        return False                # 不抑制异常，让其向上传播 / don't suppress

# with Transaction(conn) as c:
#     c.execute("INSERT ...")       # 出错自动回滚 / auto-rollback on failure
```

**练习 4：** 实现一个 `ModelEvalMode` 上下文管理器，进入时切换模型为 eval 模式，退出时恢复 train 模式。

*参考答案*：
```python
import torch
from contextlib import contextmanager

@contextmanager
def model_eval_mode(model: "torch.nn.Module"):
    was_training = model.training   # 记录原状态，更稳健 / save original state
    model.eval()                    # 切到推理模式 / switch to eval
    try:
        with torch.no_grad():       # 推理通常同时禁用梯度 / also disable grad
            yield model
    finally:
        model.train(was_training)   # 恢复原模式 / restore original mode
```
