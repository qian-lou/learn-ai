# 生成器与迭代器（Stream 对比）
# Generators and Iterators (vs Stream)

## 1. 背景（Background）

> 生成器是 Python 的惰性计算机制，类似 Java 8 的 Stream——数据按需生成，不会一次性加载到内存。大模型训练中，DataLoader 就是基于迭代器模式设计的。

## 2. 知识点（Key Concepts）

| Java | Python |
|------|--------|
| `Iterator<T>` | `__iter__` + `__next__` |
| `Stream.generate()` | `yield` 生成器 |
| `Iterable<T>` | 可迭代对象 |

## 3. 内容（Content）

```python
# 生成器函数（用 yield 代替 return）
# Generator function (use yield instead of return)
def count_up(n: int):
    """从 0 数到 n-1 / Count from 0 to n-1.
    
    惰性生成，不占用额外内存 / Lazy generation, no extra memory.
    """
    i = 0
    while i < n:
        yield i  # 暂停并返回值，下次调用继续执行
        i += 1

# 使用
for num in count_up(5):
    print(num)  # 0, 1, 2, 3, 4

# 实际案例：逐行读取大文件（处理 10GB 文件不爆内存）
def read_large_file(path: str):
    """逐行读取大文件 / Read large file line by line.
    
    Time: O(N)  Space: O(1) per line
    """
    with open(path) as f:
        for line in f:
            yield line.strip()

# 生成器表达式（类似列表推导式但惰性）
total = sum(x**2 for x in range(10_000_000))  # 不创建列表

# PyTorch DataLoader 的迭代器模式预览
# for batch in dataloader:  # DataLoader 是可迭代对象
#     loss = model(batch)
```

## 4. 详细推理（Deep Dive）

- `yield` 使函数变为生成器，每次 `next()` 调用执行到下一个 `yield`
- 生成器只能遍历一次（与 Java Stream 类似）
- `yield from` 用于委托子生成器（扁平化嵌套生成器）

## 5. 例题（Worked Examples）

### 例题 1：使用生成器实现一个高效的数据批处理加载器 / Implement an efficient data batch loader using generator

在大模型训练中，由于内存限制，我们需要分批次读取海量数据集。以下例题实现了一个类似 PyTorch DataLoader 的批处理生成器。

```python
from typing import Generator, Iterable, List, TypeVar

T = TypeVar('T')

def batched(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """将可迭代对象切分为固定大小的批次输出 / Yield batches of elements from an iterable.
    
    Time: O(N) - 遍历所有元素一次 / Iterate all elements once.
    Space: O(B) - B 为 batch_size，保存当前批次的缓冲区 / Buffer of batch size.
    """
    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []  # 清空缓冲区 / Clear buffer
    if batch:
        yield batch  # 产出最后一批 / Yield remaining elements

# 测试 / Test
dataset = range(10)
for i, batch in enumerate(batched(dataset, batch_size=3)):
    print(f"Batch {i}: {batch}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：实现一个无限生成斐波那契数列的生成器。
*参考答案*：
```python
from typing import Generator

def fibonacci() -> Generator[int, None, None]:
    """Time: O(1) per step, Space: O(1)"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

### 进阶题
**练习 2**：实现一个支持管道处理的生成器，第一阶段生成平方数，第二阶段过滤其中的奇数，最后求和。
*参考答案*：
```python
from typing import Iterable, Generator

def make_squares(nums: Iterable[int]) -> Generator[int, None, None]:
    for n in nums:
        yield n * n

def filter_odds(nums: Iterable[int]) -> Generator[int, None, None]:
    for n in nums:
        if n % 2 == 0:
            yield n

# 管道连接 / Pipeline chaining
nums = range(10)
squares = make_squares(nums)
evens = filter_odds(squares)
print(f"平方数中的偶数之和: {sum(evens)}")  # 120
```