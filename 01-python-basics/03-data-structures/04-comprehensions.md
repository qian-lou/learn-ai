# 推导式（List/Dict/Set Comprehension）
# Comprehensions

## 1. 背景（Background）

> 推导式是 Python 最优美的语法糖之一，等价于 Java Stream API 但写法更简洁。掌握推导式是写出 "Pythonic" 代码的关键。

## 2. 知识点（Key Concepts）

| 推导式类型 | 语法 | Java 等价 |
|-----------|------|-----------|
| 列表推导式 | `[expr for x in iterable]` | `stream().map().collect(toList())` |
| 字典推导式 | `{k: v for ...}` | `stream().collect(toMap())` |
| 集合推导式 | `{expr for ...}` | `stream().collect(toSet())` |
| 生成器表达式 | `(expr for ...)` | `stream()`（惰性） |

## 3. 内容（Content）

```python
# 列表推导式 / List comprehension
# Time: O(N)  Space: O(N)
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# 嵌套 / Nested
flat = [x for row in [[1,2],[3,4],[5,6]] for x in row]
# [1, 2, 3, 4, 5, 6]

# 字典推导式 / Dict comprehension
word_len = {w: len(w) for w in ["hello", "world"]}

# 集合推导式 / Set comprehension
unique_len = {len(w) for w in ["hi", "hey", "hello"]}

# 生成器表达式（惰性，节省内存）/ Generator expression
total = sum(x**2 for x in range(1000000))  # 不创建列表
```

## 4. 详细推理（Deep Dive）

- 推导式比等价的 for 循环快 ~30%（CPython 字节码优化）
- 生成器表达式是惰性的，适合大数据处理
- 嵌套超过 2 层时，改用普通 for 循环可读性更好

## 5. 例题（Worked Examples）

```python
# 矩阵运算：筛选并变换
# Filter students with score >= 90
students = {"Alice": 95, "Bob": 72, "Charlie": 88, "David": 91}
honors = {name: score for name, score in students.items() if score >= 90}
```

## 6. 习题（Exercises）

**练习 1：** 用推导式生成勾股数 `(a,b,c)` 满足 a²+b²=c²，a,b,c < 30。

**练习 2：** 对比推导式和 for 循环的性能，使用 `%timeit` 测量。
