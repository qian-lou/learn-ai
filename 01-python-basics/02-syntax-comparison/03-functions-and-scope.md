# 函数与作用域
# Functions and Scope

## 1. 背景（Background）

> Python 的函数是"一等公民"（first-class citizen），可以像变量一样传递。Java 需要通过 `@FunctionalInterface` 和 Lambda 才能实现类似功能。Python 的函数支持默认参数、可变参数、关键字参数等灵活特性。

## 2. 知识点（Key Concepts）

| 特性 | Java | Python |
|------|------|--------|
| 函数定义 | `public int add(int a, int b)` | `def add(a, b):` |
| 返回值 | 单一返回类型 | 可返回多个值（元组） |
| 默认参数 | 方法重载 | `def f(x, y=10):` |
| 可变参数 | `int... args` | `*args, **kwargs` |
| Lambda | `(x) -> x * 2` | `lambda x: x * 2` |

## 3. 内容（Content）

### 3.1 函数定义与返回多值

```python
def add(a: int, b: int) -> int:
    """两数相加 / Add two numbers."""
    return a + b

# 返回多个值（Java 需要 DTO） / Return multiple values
def divide(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b

quotient, remainder = divide(17, 5)  # 解构赋值
```

### 3.2 参数类型

```python
# 默认参数 / Default parameters
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# ⚠️ 陷阱：可变对象作为默认参数
def add_item(item, items=None):  # 正确：用 None
    if items is None:
        items = []
    items.append(item)
    return items

# *args 可变位置参数 / Variable positional args
def sum_all(*args: int) -> int:
    return sum(args)

# **kwargs 可变关键字参数 / Variable keyword args
def create_user(**kwargs) -> dict:
    return kwargs

user = create_user(name="Alice", age=25)
```

### 3.3 Lambda 与高阶函数

```python
square = lambda x: x * x

# 排序 / Sorting
users = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 20}]
users.sort(key=lambda u: u["age"])

# 高阶函数 / Higher-order functions
from typing import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

result = apply_twice(lambda x: x * 2, 3)  # 12
```

### 3.4 作用域（LEGB 规则）

```python
# L-Local, E-Enclosing, G-Global, B-Built-in
x = "global"
def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)  # "local"
    inner()

# global/nonlocal 关键字
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment
```

### 3.5 拆包用法

```python
numbers = [1, 2, 3, 4, 5]
first, *rest = numbers        # first=1, rest=[2,3,4,5]
config = {**{"host": "localhost"}, "debug": True}  # 字典合并
```

## 4. 详细推理（Deep Dive）

### 参数传递机制

Python 是"对象引用传递"——不可变对象表现如值传递，可变对象表现如引用传递。

```python
def modify(x, items):
    x += 1          # 创建新 int，原变量不变
    items.append(4)  # 修改原列表，外部可见

n, lst = 10, [1, 2, 3]
modify(n, lst)
print(n, lst)  # 10 [1, 2, 3, 4]
```

## 5. 例题（Worked Examples）

```python
# 用闭包实现工厂模式
def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
print(double(5))  # 10
```

## 6. 习题（Exercises）

**练习 1：** 编写 `make_adder(n)` 返回加 n 的函数。

**练习 2：** 实现 `memoize` 装饰器函数，缓存函数调用结果。

**练习 3：** 解释为什么 `[lambda x: x*i for i in range(5)]` 中所有 lambda 返回相同结果，如何修复？
