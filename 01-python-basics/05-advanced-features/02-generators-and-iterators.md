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

## 5-6. 例题/习题

**练习 1：** 实现一个无限斐波那契数列生成器。

**练习 2：** 用生成器实现一个数据批处理函数 `batched(iterable, size)`。
