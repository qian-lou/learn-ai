# 变量与类型系统（Java vs Python）
# Variables and Type System (Java vs Python)

## 1. 背景（Background）

> **为什么要学这个？**
>
> Java 是静态类型语言（编译期确定类型），Python 是动态类型语言（运行时确定类型）。这是两种语言最本质的差异之一。理解这个差异，能帮你快速适应 Python 的编码风格，避免以 Java 思维写 Python。
>
> **在整个体系中的位置：** 类型系统是编程语言的基础。理解 Python 的动态类型，才能理解后续的鸭子类型、装饰器、元编程等高级特性。

## 2. 知识点（Key Concepts）

| 特性 | Java | Python |
|------|------|--------|
| 类型系统 | 静态类型（编译期检查） | 动态类型（运行时确定） |
| 变量声明 | `int x = 10;` | `x = 10` |
| 类型推断 | `var x = 10;`（Java 10+） | 天生就是 |
| 常量 | `final int X = 10;` | `X = 10`（约定大写，无强制） |
| 类型注解 | 强制 | 可选（Type Hints） |
| 空值 | `null` | `None` |
| 基本类型 | `int, long, double, boolean` | `int, float, bool`（全是对象） |

## 3. 内容（Content）

### 3.1 变量声明与赋值

```python
# Python 不需要声明类型（对比 Java）
# Python doesn't require type declarations (vs Java)

# Java: int age = 25;
age = 25

# Java: String name = "Alice";
name = "Alice"

# Java: double pi = 3.14159;
pi = 3.14159

# Java: boolean isActive = true;
is_active = True  # 注意：Python 用 True/False，首字母大写

# Java: final int MAX_SIZE = 100;
MAX_SIZE = 100  # Python 没有 final，大写命名只是约定

# 动态类型：同一变量可以改变类型（Java 不允许）
# Dynamic typing: same variable can change type (Java doesn't allow this)
x = 10          # int
x = "hello"     # str（合法！Java 中这是编译错误）
x = [1, 2, 3]   # list（再次改变类型，合法！）
```

### 3.2 数据类型对比

```python
# ============================================
# 整数 int（Python 整数没有上限！）
# Integer (Python integers have no upper limit!)
# ============================================

# Java: int 最大 2^31-1 = 2,147,483,647
# Java: long 最大 2^63-1
# Python: 无限制！
big_number = 10 ** 100  # 10 的 100 次方，Java 需要 BigInteger
print(type(big_number))  # <class 'int'>

# ============================================
# 浮点数 float（对应 Java 的 double）
# Float (corresponds to Java's double)
# ============================================
price = 19.99
print(type(price))  # <class 'float'>

# 注意：Python 没有 Java 的 float/double 区分，只有一种浮点数
# Note: Python has no float/double distinction, only one float type

# ============================================
# 布尔 bool（注意大小写！）
# Boolean (note the capitalization!)
# ============================================
# Java: true, false (小写)
# Python: True, False (首字母大写)
is_valid = True
is_empty = False

# Python 的 bool 是 int 的子类！（这在 Java 中不可想象）
# Python's bool is a subclass of int! (unimaginable in Java)
print(True + True)   # 2
print(True * 10)     # 10
print(isinstance(True, int))  # True

# ============================================
# 字符串 str（不可变，类似 Java 的 String）
# String (immutable, like Java's String)
# ============================================
# 单引号和双引号等价（Java 只能用双引号）
s1 = 'hello'
s2 = "hello"
# 三引号用于多行字符串（类似 Java 15+ 的 Text Block）
s3 = """
这是一个
多行字符串
"""

# f-string 格式化（类似 Java 的 String.format，但更简洁）
# f-string formatting (like Java's String.format, but more concise)
name = "Alice"
age = 25
# Java: String.format("Name: %s, Age: %d", name, age)
# Python:
greeting = f"Name: {name}, Age: {age}"
# 还可以包含表达式：
info = f"Next year: {age + 1}, Name upper: {name.upper()}"

# ============================================
# None（对应 Java 的 null）
# None (corresponds to Java's null)
# ============================================
result = None

# 检查是否为 None（用 is，不用 ==）
# Check for None (use 'is', not '==')
# Java: if (result == null)
# Python:
if result is None:
    print("No result")
```

### 3.3 类型检查与转换

```python
# 类型检查
# Type checking

# Java: x instanceof String
# Python:
x = "hello"
print(type(x))              # <class 'str'>
print(isinstance(x, str))   # True

# 类型转换（显式）
# Type conversion (explicit)
# Java: (int) 3.14  或  Integer.parseInt("42")
# Python:
num_str = "42"
num = int(num_str)       # str → int
pi = float("3.14")      # str → float
text = str(42)           # int → str
flag = bool(1)           # int → bool (True)
items = list("hello")    # str → list ['h', 'e', 'l', 'l', 'o']
```

### 3.4 Python 的 "一切皆对象"

```python
# Python 中，一切都是对象（包括 int、function、class 本身）
# In Python, everything is an object (including int, function, class itself)

# Java 中 int 是基本类型，Integer 是包装类
# In Java, int is primitive, Integer is wrapper class
# Python 没有这种区分

x = 42
print(type(x))       # <class 'int'>
print(id(x))         # 内存地址
print(x.__class__)   # <class 'int'>
print(x.bit_length())  # 6（42 的二进制位数）

# 函数也是对象！（Java 需要 @FunctionalInterface）
# Functions are objects too! (Java needs @FunctionalInterface)
def greet(name):
    return f"Hello, {name}"

print(type(greet))   # <class 'function'>
my_func = greet      # 函数可以赋值给变量
print(my_func("Bob"))  # Hello, Bob
```

### 3.5 Type Hints（类型注解）

```python
# Python 3.5+ 支持类型注解（可选的，不影响运行）
# Python 3.5+ supports type hints (optional, doesn't affect runtime)

# 基本类型注解（类似 Java 的类型声明）
# Basic type hints (similar to Java type declarations)
name: str = "Alice"
age: int = 25
price: float = 19.99
is_active: bool = True

# 函数类型注解
# Function type hints
def add(a: int, b: int) -> int:
    """两数相加 / Add two numbers.
    
    Args:
        a: 第一个加数 / First addend.
        b: 第二个加数 / Second addend.
    
    Returns:
        两数之和 / Sum of a and b.
    """
    return a + b

# 复杂类型（Python 3.9+ 可以直接用内置类型）
# Complex types (Python 3.9+ can use built-in types directly)
from typing import Optional

# Java: List<String> names = new ArrayList<>();
names: list[str] = ["Alice", "Bob"]

# Java: Map<String, Integer> scores = new HashMap<>();
scores: dict[str, int] = {"Alice": 95, "Bob": 87}

# Java: Optional<String> maybe = Optional.of("value");
maybe: Optional[str] = None

# Java: Set<Integer> unique = new HashSet<>();
unique: set[int] = {1, 2, 3}

# 联合类型（Python 3.10+）
# Union types (Python 3.10+)
# Java: 没有直接对应，需要方法重载
def process(data: str | int) -> str:
    return str(data)
```

## 4. 详细推理（Deep Dive）

### 4.1 动态类型 vs 静态类型的本质

```
Java（静态类型）：
┌──────────┐     ┌──────────┐
│ 变量 x   │────→│ int 类型  │   变量绑定到类型
│ (编译期) │     │ 值: 42   │   编译器检查类型匹配
└──────────┘     └──────────┘

Python（动态类型）：
┌──────────┐     ┌──────────────┐
│ 名字 x   │────→│ int 对象     │   名字绑定到对象
│ (标签)   │     │ 值: 42       │   对象自己知道类型
└──────────┘     │ type: int    │
                 └──────────────┘

x = "hello" 后：
┌──────────┐     ┌──────────────┐
│ 名字 x   │────→│ str 对象     │   标签重新指向另一个对象
│ (标签)   │     │ 值: "hello"  │   原来的 int 对象被 GC
└──────────┘     │ type: str    │
                 └──────────────┘
```

**核心理解：** Python 的变量不是"盒子"（存储值），而是"标签"（指向对象）。

### 4.2 Python 的内存模型

```python
# 不可变对象（int, str, tuple）的行为
# Immutable object behavior

a = 42
b = 42
print(a is b)  # True！Python 会缓存小整数 [-5, 256]

a = 1000
b = 1000
print(a is b)  # False 或 True（取决于实现）

# == 比较值，is 比较身份（内存地址）
# == compares value, 'is' compares identity (memory address)
# Java: equals() vs ==
# Python: == vs is

# 可变对象（list, dict）的陷阱
# Mutable object pitfall
a = [1, 2, 3]
b = a        # b 和 a 指向同一个列表！
b.append(4)
print(a)     # [1, 2, 3, 4]  ← a 也变了！

# 正确的复制方式
# Correct way to copy
b = a.copy()        # 浅拷贝
b = list(a)          # 浅拷贝
import copy
b = copy.deepcopy(a)  # 深拷贝
```

### 4.3 鸭子类型（Duck Typing）

```python
# "如果它看起来像鸭子，走起来像鸭子，叫起来像鸭子，那它就是鸭子"
# "If it looks like a duck, walks like a duck, quacks like a duck, then it's a duck"

# Java 需要通过接口来实现多态
# Java requires interfaces for polymorphism

# Python 不关心类型，只关心行为
# Python doesn't care about type, only behavior

def get_length(obj):
    """获取任何有 len 的对象的长度 / Get length of any object with len."""
    return len(obj)

# 所有这些都能工作，不需要共同的接口！
# All of these work without a common interface!
print(get_length("hello"))      # 5 (str)
print(get_length([1, 2, 3]))    # 3 (list)
print(get_length({"a": 1}))     # 1 (dict)
print(get_length((1, 2, 3, 4))) # 4 (tuple)
```

## 5. 例题（Worked Examples）

### 例题 1：Java 到 Python 的代码翻译

**Java 代码：**
```java
public class TypeDemo {
    public static void main(String[] args) {
        int count = 0;
        String message = "Hello";
        double price = 19.99;
        boolean isValid = true;
        
        List<String> names = new ArrayList<>(Arrays.asList("Alice", "Bob"));
        Map<String, Integer> scores = new HashMap<>();
        scores.put("Alice", 95);
        scores.put("Bob", 87);
        
        Optional<String> result = Optional.ofNullable(null);
        String value = result.orElse("default");
        
        System.out.printf("Count: %d, Price: %.2f%n", count, price);
    }
}
```

**等价 Python 代码：**
```python
# 变量声明（不需要类型声明）
# Variable declarations (no type declaration needed)
count: int = 0
message: str = "Hello"
price: float = 19.99
is_valid: bool = True

# 列表和字典（字面量语法，不需要 new）
# List and dict (literal syntax, no 'new' needed)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95, "Bob": 87}

# Optional 处理（更简洁）
# Optional handling (more concise)
result: str | None = None
value = result if result is not None else "default"
# 或者更 Pythonic 的写法：
value = result or "default"  # 注意：空字符串也会走 default

# 格式化输出
# Formatted output
print(f"Count: {count}, Price: {price:.2f}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 将以下 Java 代码翻译为 Python：
```java
final String GREETING = "Welcome";
int total = 0;
for (int i = 1; i <= 100; i++) {
    total += i;
}
System.out.println(GREETING + "! Sum = " + total);
```

**练习 2：** 解释为什么以下 Python 代码不会报错：
```python
x = 10
x = "now I'm a string"
x = [1, 2, 3]
print(type(x))
```
如果在 Java 中写类似代码会怎样？

### 进阶题

**练习 3：** 预测以下代码的输出，并解释原因：
```python
a = [1, 2, 3]
b = a
c = a.copy()
a.append(4)
print(f"a={a}, b={b}, c={c}")
print(f"a is b: {a is b}")
print(f"a is c: {a is c}")
print(f"a == c: {a == c}")
```

> **参考答案：**
> ```
> a=[1, 2, 3, 4], b=[1, 2, 3, 4], c=[1, 2, 3]
> a is b: True    # b 和 a 指向同一对象
> a is c: False   # c 是独立的副本
> a == c: False   # 值也不同了（a 多了 4）
> ```

**练习 4：** 用 Type Hints 重写以下函数，添加完整的类型注解和 Google 风格 docstring：
```python
def process_scores(students, min_score):
    results = {}
    for name, score in students.items():
        if score >= min_score:
            results[name] = "PASS"
        else:
            results[name] = "FAIL"
    return results
```
