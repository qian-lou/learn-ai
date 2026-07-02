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

### 3.1 四种推导式速览

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

### 3.2 过滤 vs 变换：`if` 放对位置（易错点）

```python
# 末尾的 if 是"过滤"（筛掉元素）/ trailing if FILTERS items
kept = [x for x in range(10) if x % 2 == 0]        # [0,2,4,6,8]

# for 之前的 if/else 是"条件表达式"（变换每个元素，不筛掉）/ ternary TRANSFORMS
labels = ["even" if x % 2 == 0 else "odd" for x in range(4)]  # 4 个都在
# ['even', 'odd', 'even', 'odd']

# 二者可叠加：先变换、后过滤 / transform then filter
out = [x * 2 for x in range(10) if x > 5]          # [12, 14, 16, 18]

# 多重 for 的顺序 = 嵌套循环从外到内 / multiple 'for' = outer→inner loop order
pairs = [(i, j) for i in range(2) for j in range(2)]
# [(0,0),(0,1),(1,0),(1,1)]  等价于双层 for / equivalent to nested loops
```

### 3.3 海象运算符 `:=`：在推导式里复用昂贵计算（3.8+）

```python
# 陷阱：下式对每个 x 调用 expensive() 两次（过滤一次、产出一次）
# Pitfall: expensive() runs TWICE per item — once to filter, once to emit
def expensive(x: int) -> int:
    return x * x  # 假设很贵 / pretend it's costly

bad = [expensive(x) for x in range(10) if expensive(x) > 20]   # 调用 2N 次

# 用 := 把结果绑定一次，过滤和产出共用 / bind once, reuse in filter & output
good = [y for x in range(10) if (y := expensive(x)) > 20]      # 调用 N 次
# Java 没有等价的"内联赋值进 Stream"，需提前 map 再 filter
```

### 3.4 生成器的惰性：威力与三大陷阱

```python
# (expr for ...) 是生成器：按需逐个产出，内存恒定 / lazy, O(1) memory
gen = (x * x for x in range(1_000_000))   # 此刻不计算任何东西 / nothing runs yet
first = next(gen)                          # 现在才算出第一个 / computed on demand

# 陷阱 1：生成器一次性耗尽，不能重复遍历 / single-use, exhausts after one pass
g = (x for x in range(3))
print(list(g))   # [0, 1, 2]
print(list(g))   # [] —— 已耗尽！需要复用就用 list / re-create or use a list

# 陷阱 2：惰性 = 延迟求值，闭包变量会"读最新值"而非"捕获当时值"
# Late binding: the generator reads the loop var at iteration time
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])               # [2, 2, 2] —— 都读到最终 i=2！
fixed = [lambda i=i: i for i in range(3)]  # 用默认参数当场快照 / snapshot via default
print([f() for f in fixed])               # [0, 1, 2] 正确

# 陷阱 3：聚合直接喂生成器，省去中间 list / feed generators straight to reducers
total = sum(x * x for x in range(1000))    # 无中间列表 / no temp list materialized
# 但若要多次用结果（len + 再遍历），就该物化成 list / materialize if reused
```

## 4. 详细推理（Deep Dive）

### 4.1 字节码层面：推导式为何比手写 for 快

把"手写 for + append"和"列表推导式"反汇编对比，差异一目了然：

```python
import dis

def by_loop():
    out = []
    for x in range(3):
        out.append(x)        # 每轮：LOAD out → LOAD_METHOD append → CALL
    return out

def by_comp():
    return [x for x in range(3)]   # 每轮：LIST_APPEND（单条字节码）

# dis.dis(by_loop)  # 循环体含 LOAD_FAST out / LOAD_METHOD append / CALL_METHOD
# dis.dis(by_comp)  # 循环体仅一条 LIST_APPEND，直接操作栈顶列表
```

两点关键差异：

1. **省去属性查找与函数调用**。`out.append(x)` 每轮都要：按名字 `out` 取列表 → 查 `append` 方法（`LOAD_METHOD`）→ 压参 → `CALL`。推导式则编译成专用字节码 **`LIST_APPEND`**（dict/set 推导对应 `MAP_ADD`/`SET_ADD`），**直接对栈上那个正在构建的列表追加，无名字解析、无方法查找、无 Python 级函数调用帧**。
2. **构建对象不暴露给用户代码**。正在生成的列表是解释器的内部栈对象，少了一层间接。

综合下来，推导式通常比等价的显式 `append` 循环快约 **30%**（与元素操作的轻重相关：操作越轻，省下的调用开销占比越大）。这与 §3.3 字符串里"循环 `+=` 慢"是一类根因——**Python 级别的重复函数调用/属性查找是大头开销**。

### 4.2 推导式有独立作用域：不再泄漏循环变量（Py3 行为）

```python
i = "outer"
_ = [i for i in range(3)]     # 推导式在自己的作用域里跑 / runs in its own scope
print(i)                       # 'outer' —— 未被污染！Py2 会变成 2
```

Python 3 把列表/字典/集合推导式和生成器表达式都实现为**一个隐式的内层函数**（CPython 用 `MAKE_FUNCTION` 生成一个匿名 code 对象并立即调用）。因此循环变量 `i` 活在那个内层函数栈帧里，**不泄漏到外层**——这修复了 Py2 推导式覆盖同名外部变量的历史坑。副作用是：推导式内部若要读外层变量，是按**闭包**规则捕获的，这也解释了 §3.4 陷阱 2 的"延迟绑定"现象。

### 4.3 生成器表达式：一台惰性状态机

生成器表达式编译出的是**生成器对象**，本质是一台可暂停/恢复的状态机：

- `next()` 触发时才执行到下一个 `yield`（这里是隐式产出），算出一个值就**冻结整个执行帧（局部变量、指令指针全部保留）并返回**；
- 再次 `next()` 从冻结点**恢复**继续，直到迭代源耗尽抛 `StopIteration`。

所以它**内存恒定 O(1)**（任意时刻只持有"当前一个值 + 帧状态"），适合流水线和大数据；代价是**单次性**（§3.4 陷阱 1）和**不可索引/无 `len`**。这正好对应 Java **Stream 的惰性**：`stream().map().filter()` 也是声明算子、`collect`/`reduce` 这类**终端操作**才真正拉动数据；Stream 同样**只能消费一次**。区别在于 Python 的生成器可由 `yield` 自定义任意产出逻辑，比 Stream 的固定算子更通用。

### 4.4 选型与可读性准则

- **要立即全量结果、且会多次使用** → 用 `list`/`dict`/`set` 推导式（一次构建、可重复遍历、可索引）。
- **数据流大、只过一遍、想省内存** → 用生成器表达式喂给 `sum`/`any`/`max`/`join` 等消费者。
- **逻辑含副作用、多重分支、嵌套 ≥ 3 层** → 退回普通 `for`。推导式的优势是"声明式变换"，一旦塞进复杂控制流，可读性反而崩塌——**能一眼读懂才是 Pythonic，不是越短越好**。

## 5. 例题（Worked Examples）

### 例题 1：筛选并变换 / Filter and transform

```python
# Filter students with score >= 90
students = {"Alice": 95, "Bob": 72, "Charlie": 88, "David": 91}
honors = {name: score for name, score in students.items() if score >= 90}
# {'Alice': 95, 'David': 91}
```

### 例题 2：流式预处理大语料（生成器表达式 = Python 版 Stream 流水线）

```python
from typing import Iterable, Iterator

# 场景：训练前要把巨大的文本文件逐行清洗——去空白、过滤过短行、截断超长行，
# 全程不把整份语料读进内存。每个环节都是惰性生成器，像搭流水线一样串起来。
# Scenario: stream-clean a huge corpus line by line, constant memory.
# Time: O(总字符数)  Space: O(单行长度) / linear time, O(1) lines in memory
def preprocess(lines: Iterable[str], min_len: int = 3, max_len: int = 128) -> Iterator[str]:
    """对文本行做流式清洗：去空白 → 过滤过短 → 截断过长。

    Stream-clean text lines: strip → drop too-short → truncate too-long.

    Args:
        lines: 任意行级可迭代对象（可为文件句柄，惰性读取）/ iterable of lines.
        min_len: 保留行的最小字符数 / minimum chars to keep a line.
        max_len: 单行最大字符数，超出则截断 / hard cap per line.

    Yields:
        清洗后的文本行 / cleaned lines, one at a time.
    """
    # 三段生成器表达式串成流水线，逐行流过、互不物化 / chained lazy pipeline
    stripped = (ln.strip() for ln in lines)                  # 1) 去首尾空白
    long_enough = (ln for ln in stripped if len(ln) >= min_len)  # 2) 过滤过短
    capped = (ln[:max_len] for ln in long_enough)            # 3) 截断超长
    yield from capped                                         # 委托产出 / delegate


raw = ["  hi  ", "  a valid sentence  ", "x", "  another good line  "]
# 惰性：直到这里被消费（list/循环/sum…）才真正逐行计算 / runs only when consumed
print(list(preprocess(raw, min_len=3)))
# ['a valid sentence', 'another good line']

# 真实用法：直接对接文件句柄，TB 级语料也只占一行的内存 / works on huge files
# with open("corpus.txt", encoding="utf-8") as f:
#     for clean_line in preprocess(f):
#         ...  # 喂给 tokenizer / feed to tokenizer, never load all in memory
```

要点：(1) 三个**生成器表达式首尾相接**形成流水线，数据**逐行穿过**、任何一步都不构建完整中间列表——这是 §4.3 惰性状态机的直接应用，内存恒定；(2) **整条管道在被消费前不执行任何计算**（`list(...)` 或 `for` 才拉动），与 Java Stream 的"终端操作触发"完全一致；(3) 直接 `for ln in open(...)` 即可处理远超内存的文件，把 `list` 推导式换成生成器表达式是这里**唯一**但**决定性**的改动。对应 Java：`Files.lines(path).map(String::strip).filter(...).map(...)` 一条惰性 Stream，思路同构。

## 6. 习题（Exercises）

**练习 1：** 用推导式生成勾股数 `(a,b,c)` 满足 a²+b²=c²，a,b,c < 30。

*参考答案*：
```python
# Time: O(N^3) Space: O(K)，N=30，K=结果数 / triple loop
# a<=b 去重，避免 (3,4,5)/(4,3,5) 重复 / a<=b removes mirror duplicates
triples = [(a, b, c)
           for a in range(1, 30)
           for b in range(a, 30)
           for c in range(b, 30)
           if a * a + b * b == c * c]
print(triples)  # [(3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25), (8, 15, 17), (9, 12, 15), ...]
```

**练习 2：** 对比推导式和 for 循环的性能，使用 `%timeit` 测量。

*参考答案*：
```python
# 在 Jupyter/IPython 中运行 / run in Jupyter
def by_loop():
    out = []
    for x in range(10000):
        out.append(x * x)   # 每次 append 有函数调用开销 / per-call overhead
    return out

def by_comp():
    return [x * x for x in range(10000)]  # 字节码层优化 / bytecode-optimized

%timeit by_loop()   # 较慢 / slower
%timeit by_comp()   # 通常快约 30% / typically ~30% faster
# 结论：推导式更快是因省去显式 append 的属性查找与调用 / avoids repeated append lookup
```
