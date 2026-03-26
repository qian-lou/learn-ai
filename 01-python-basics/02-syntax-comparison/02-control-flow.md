# 控制流（if/for/while 对比）
# Control Flow (if/for/while Comparison)

## 1. 背景（Background）

> **为什么要学这个？**
>
> 控制流是任何编程语言的骨架。Python 的控制流与 Java 在语法上有重大差异：没有花括号 `{}`，用**缩进**来定义代码块。这是 Java 工程师最需要适应的语法变化之一。此外，Python 的 `for` 循环和 Java 完全不同——Python 只有 `for-each` 风格的循环。
>
> **在整个体系中的位置：** 控制流是编写任何算法和程序逻辑的基础。掌握 Python 控制流后，才能顺畅地编写数据处理和模型训练代码。

## 2. 知识点（Key Concepts）

| 特性 | Java | Python |
|------|------|--------|
| 代码块定义 | `{}` 花括号 | 缩进（4 个空格） |
| if 语法 | `if (condition) {}` | `if condition:` |
| 三元运算符 | `a ? b : c` | `b if a else c` |
| for 循环 | C 风格 `for(;;)` 和 for-each | 只有 for-each |
| switch/case | `switch-case` | `match-case`（3.10+） |
| 空语句 | `{}` 或 `;` | `pass` |

## 3. 内容（Content）

### 3.1 if 条件语句

```python
# Java vs Python 对比
# Java vs Python comparison

# ====== Java ======
# if (score >= 90) {
#     grade = "A";
# } else if (score >= 80) {
#     grade = "B";
# } else {
#     grade = "C";
# }

# ====== Python ======
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:      # 注意：elif，不是 else if
    grade = "B"
else:
    grade = "C"

# 三元运算符
# Ternary operator

# Java: String status = age >= 18 ? "adult" : "minor";
# Python:
age = 20
status = "adult" if age >= 18 else "minor"

# 链式比较（Python 独有，数学风格）
# Chained comparison (Python-unique, math-style)
x = 15
# Java: if (x > 10 && x < 20)
# Python:
if 10 < x < 20:  # 更接近数学写法！
    print("在范围内")

# 真值判断（Truthy / Falsy）
# Truthiness evaluation

# Python 中以下值为 Falsy（等价于 False）：
# The following values are Falsy in Python:
# None, False, 0, 0.0, "", [], {}, set(), ()
# 其他一切都是 Truthy

# 因此可以这样简写：
# So you can write:
items = [1, 2, 3]
if items:          # 代替 if len(items) > 0:
    print("列表非空")

name = ""
if not name:       # 代替 if name == "" or name is None:
    print("名字为空")
```

### 3.2 for 循环

```python
# ==================================================
# Python 的 for 只有 for-each 风格（没有 Java 的 C 风格 for）
# Python's for is only for-each style (no C-style for)
# ==================================================

# Java: for (String name : names) { ... }
# Python:
names = ["Alice", "Bob", "Charlie"]
for name in names:
    print(name)

# Java: for (int i = 0; i < 10; i++) { ... }
# Python: 使用 range()
for i in range(10):          # 0, 1, 2, ..., 9
    print(i)

for i in range(2, 10):       # 2, 3, ..., 9
    print(i)

for i in range(0, 10, 2):    # 0, 2, 4, 6, 8 (步长为 2)
    print(i)

# 带索引的遍历（类似 Java 的 for(int i=0; i<list.size(); i++)）
# Indexed iteration
# Java: for (int i = 0; i < names.size(); i++) {
#     System.out.println(i + ": " + names.get(i));
# }
# Python: 使用 enumerate()
for i, name in enumerate(names):
    print(f"{i}: {name}")

# 同时遍历两个列表（类似 Java Streams 的 zip）
# Iterate two lists simultaneously
names = ["Alice", "Bob"]
scores = [95, 87]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# 遍历字典
# Iterate over dict
# Java: for (Map.Entry<String, Integer> entry : map.entrySet()) {
#     entry.getKey(); entry.getValue();
# }
# Python:
scores = {"Alice": 95, "Bob": 87}
for name, score in scores.items():
    print(f"{name}: {score}")

# for-else（Python 独有！）
# for-else (Python-unique!)
# 如果循环正常结束（没有 break），执行 else
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            break
    else:
        # 注意：这个 else 属于 for，不是 if
        # Note: this else belongs to for, not if
        print(f"{n} 是质数")
```

### 3.3 while 循环

```python
# while 循环与 Java 类似
# while loop is similar to Java

# Java: while (count > 0) { count--; }
# Python:
count = 5
while count > 0:
    print(count)
    count -= 1  # 注意：Python 没有 count-- 语法

# while-else（Python 独有）
# while-else (Python-unique)
n = 10
while n > 0:
    if n == 5:
        break
    n -= 1
else:
    print("循环正常结束")  # 因为有 break，不会执行
```

### 3.4 match-case（Python 3.10+）

```python
# 类似 Java 的 switch（但更强大，支持模式匹配）
# Similar to Java's switch (but more powerful, supports pattern matching)

# Java 17+:
# switch (status) {
#     case "active" -> System.out.println("活跃");
#     case "inactive" -> System.out.println("不活跃");
#     default -> System.out.println("未知");
# }

# Python 3.10+:
status = "active"
match status:
    case "active":
        print("活跃")
    case "inactive":
        print("不活跃")
    case _:          # 类似 default
        print("未知")

# 模式匹配的强大之处（Java switch 做不到的）
# Power of pattern matching (what Java switch can't do)
point = (3, 4)
match point:
    case (0, 0):
        print("原点")
    case (x, 0):
        print(f"x 轴上，x={x}")
    case (0, y):
        print(f"y 轴上，y={y}")
    case (x, y):
        print(f"点 ({x}, {y})")

# 带条件的模式匹配（guard）
command = {"action": "move", "direction": "north", "speed": 5}
match command:
    case {"action": "move", "direction": d, "speed": s} if s > 3:
        print(f"快速移动到 {d}，速度 {s}")
    case {"action": "move", "direction": d}:
        print(f"慢速移动到 {d}")
```

### 3.5 推导式（Comprehension）

```python
# Python 最优美的语法之一，Java 中用 Stream 实现类似功能
# One of Python's most elegant syntaxes, Java uses Streams for similar functionality

# Java:
# List<Integer> squares = IntStream.range(1, 11)
#     .map(x -> x * x)
#     .boxed()
#     .collect(Collectors.toList());

# Python（一行搞定！）:
squares = [x ** 2 for x in range(1, 11)]
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 带条件的推导式
# Comprehension with condition
# Java: .filter(x -> x % 2 == 0).map(x -> x * x)
even_squares = [x ** 2 for x in range(1, 11) if x % 2 == 0]
# [4, 16, 36, 64, 100]

# 字典推导式
# Dict comprehension
# Java: stream.collect(Collectors.toMap(s -> s, s -> s.length()))
words = ["hello", "world", "python"]
word_lengths = {w: len(w) for w in words}
# {'hello': 5, 'world': 5, 'python': 6}

# 集合推导式
# Set comprehension
unique_lengths = {len(w) for w in words}
# {5, 6}

# 嵌套推导式
# Nested comprehension
# Java: 双重 for 循环
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## 4. 详细推理（Deep Dive）

### 4.1 Python 缩进的哲学

```
Python 之禅（import this）：
"There should be one-- and preferably only one --obvious way to do it."
"应该有且仅有一种明显的方式来做一件事。"

Java 允许不同的花括号风格（K&R, Allman, etc.），
Python 强制使用缩进，消除了代码风格争论。

缩进规则：
1. 推荐 4 个空格（不是 Tab）
2. 同一代码块的缩进必须一致
3. 混用 Tab 和空格会报 TabError

# 反例（会报错）
if True:
    print("4空格")
      print("6空格")  # IndentationError!
```

### 4.2 range() 的懒加载

```python
# range() 返回的是惰性序列，不会真正创建列表
# range() returns a lazy sequence, doesn't actually create a list

# 这个不会占用 10GB 内存！
# This doesn't use 10GB of memory!
r = range(10_000_000_000)  # 100 亿
print(9999999999 in r)     # True，O(1) 时间复杂度

# 类似 Java 的 IntStream.range()
# 如果需要实际的列表，可以 list(range(10))
```

## 5. 例题（Worked Examples）

### 例题 1：FizzBuzz（Java → Python 翻译）

```python
# Java 版本:
# for (int i = 1; i <= 100; i++) {
#     if (i % 15 == 0) System.out.println("FizzBuzz");
#     else if (i % 3 == 0) System.out.println("Fizz");
#     else if (i % 5 == 0) System.out.println("Buzz");
#     else System.out.println(i);
# }

# Python 版本:
# Time: O(N)  Space: O(1)
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# 更 Pythonic 的版本（使用推导式）:
result = [
    "FizzBuzz" if i % 15 == 0
    else "Fizz" if i % 3 == 0
    else "Buzz" if i % 5 == 0
    else str(i)
    for i in range(1, 101)
]
```

### 例题 2：统计单词频率

```python
# Java:
# Map<String, Integer> freq = new HashMap<>();
# for (String word : words) {
#     freq.merge(word, 1, Integer::sum);
# }

# Python（多种方式）:
# Time: O(N)  Space: O(N)
text = "the quick brown fox jumps over the lazy dog the fox"
words = text.split()

# 方式 1：字典
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1  # get 提供默认值，避免 KeyError

# 方式 2：collections.Counter（推荐）
from collections import Counter
freq = Counter(words)
print(freq.most_common(3))  # [('the', 3), ('fox', 2), ('quick', 1)]

# 方式 3：字典推导式
freq = {word: words.count(word) for word in set(words)}
# 注意：方式 3 的时间复杂度是 O(N^2)，不推荐！
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 将以下 Java 代码翻译为 Python：
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
List<Integer> evenSquares = numbers.stream()
    .filter(n -> n % 2 == 0)
    .map(n -> n * n)
    .collect(Collectors.toList());
System.out.println(evenSquares);
```

> **参考答案：** `even_squares = [n ** 2 for n in range(1, 11) if n % 2 == 0]`

**练习 2：** 使用 `for-else` 编写一个函数，判断一个数是否为质数。

### 进阶题

**练习 3：** 使用嵌套推导式，将一个 3×3 矩阵转置。

**练习 4：** 使用 `match-case` 实现一个简单的计算器，支持 +、-、*、/ 四则运算，并处理除零错误。
