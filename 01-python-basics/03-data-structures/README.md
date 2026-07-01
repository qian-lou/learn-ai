# 03-data-structures — 数据结构

> **所属阶段**：阶段一 · Python 基础
> **学习目标**：掌握 Python 内置数据结构的用法**与底层实现/复杂度**，与 Java 集合框架逐一对比，学会为数据管道选对容器
> **预估时长**：3-4 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [list-and-tuple](./01-list-and-tuple.md) | 列表与元组 | list 是动态数组（指针数组）非链表；append 均摊 O(1)、头部插入 O(N)；切片 `[start:stop:step]`；浅拷贝三大坑；tuple 不可变/可哈希/能做键 |
| 02 | [dict-and-set](./02-dict-and-set.md) | 字典与集合 | dict/set 开放寻址（vs Java 链地址）；compact dict 3.7+ 保序；视图对象是动态窗口；`get`/`setdefault`；Counter/defaultdict/OrderedDict |
| 03 | [string-processing](./03-string-processing.md) | 字符串处理 | f-string 格式化；`re` 正则（预编译热点）；不可变+join（避免 O(N²) 拼接）；str vs bytes 编码；字符串驻留 interning |
| 04 | [comprehensions](./04-comprehensions.md) | 推导式 | 四种推导式；末尾 if 过滤 vs 前置 if/else 变换；海象运算符 `:=` 复用计算；生成器表达式惰性；推导式字节码优势 |

---

## 🔑 知识点详解

### 01 · 列表与元组

- **核心概念**：`list` 底层是「指向指针数组的对象」（连续内存存 `PyObject*`），所以能混存类型、随机访问 O(1)，但缓存不友好（这正是 NumPy 快 100× 的对比点）；`tuple` 定长、不可变、可哈希。
- **关键 API / 复杂度**：`append`/`pop()` 尾部 **均摊 O(1)**；`insert(0,x)`/`pop(0)` 头部 **O(N)**（要整体 memmove）；`x in list` 是 **O(N)** 线性扫描；切片 `[::-1]` 反转、`[a:b]` 越界自动截断；频繁头部进出用 `collections.deque`（两端 O(1)）。
- **易错点**：① `b = a` 是别名不是拷贝；`[:]`/`.copy()`/`list()` 是**浅拷贝**，嵌套结构共享内层——嵌套必须 `copy.deepcopy`；② `[[0]*3]*2` 让两行是同一个对象，用推导式 `[[0]*3 for _ in range(2)]` 才对；③ tuple「不可变」指绑定不可变，内含的 list 仍可改，且此时整个 tuple 不可哈希。
- **Java 视角**：`list` ≈ `ArrayList`（都是引用数组，扩容因子 CPython≈1.125× vs Java 1.5×）；`deque` ≈ `ArrayDeque`；`tuple`/`namedtuple` ≈ `record`（不可变、值语义、适合做 Map 键）。
- **前置**：02-syntax（可变性、切片依赖类型/引用知识）。

### 02 · 字典与集合

- **核心概念**：`dict`/`set` 用**开放寻址**哈希表（单块连续数组，探测下一个空槽），比 Java 的链地址法更省内存、更利于 CPU 缓存；`set` 与 `dict` 同源，只是不存 value。
- **关键 API / 复杂度**：get/put/`in` **平均 O(1)**、最坏 O(N)；`d.get(k, default)` 一次哈希（优于先 `in` 再 `[]` 的两次）；`d.setdefault(k, v)` 读+缺则写；集合运算 `& | - ^`；`Counter`（计数）、`defaultdict`（自动建值）、`OrderedDict`（`move_to_end`）都是 dict 子类。
- **易错点**：① `.keys()/.values()/.items()` 返回**动态视图**而非快照，边遍历边改会抛 `RuntimeError`（需 `for k in list(d)` 先物化）；② 装载因子超 **2/3** 就扩容（Java 是 0.75）；③ `set` 去重**不保序**，要保序去重用 `dict.fromkeys()`；元素必须可哈希（list 不行，frozenset/tuple 可以）。
- **Java 视角**：`dict` ≈ `HashMap`，但 3.7+ 天然保序（靠 compact dict 的两层结构：稀疏 indices + 紧凑 entries），Java 需 `LinkedHashMap` 额外挂链表才保序；`set` ≈ `HashSet`；`defaultdict` ≈ `computeIfAbsent`；`get` ≈ `getOrDefault`。
- **前置**：01（哈希/可变性对比），02-syntax。

### 03 · 字符串处理

- **核心概念**：`str` 是**不可变**的 Unicode 码点序列（PEP 393 紧凑布局，纯 ASCII 每字符 1 字节）；`bytes` 是原始字节，I/O 边界必须显式 `encode`/`decode`。
- **关键 API / 心法**：f-string `f"{x:.2f}"`/`f"{n:,}"`/`f"{s:>10}"`；`"".join(iterable)` 拼接（O(N)，预算总长一次分配）；`re.compile()` 预编译热点模式复用；`removeprefix/removesuffix`（3.9+）；`sys.intern()` 手动驻留。
- **易错点**：① 循环里 `s += x` 是 **O(N²)** 灾难（每次新建整串），一律用 `join`；② `len(str)` 数码点、`len(bytes)` 数字节，中文/emoji 下两者不等；③ `str.split()` 无参按任意空白折叠并丢空串，`split(" ")` 保留空串——语义不同易错。
- **Java 视角**：`str` ≈ `String`（都不可变、可哈希、可做键）；`join` ≈ `String.join`/`StringBuilder`；`re` ≈ `java.util.regex`（`re.compile` ≈ `Pattern.compile`）；`sys.intern` ≈ `String.intern`；判等永远用 `==`，不用 `is`。
- **前置**：01（不可变/可哈希与 tuple 同理）。

### 04 · 推导式

- **核心概念**：推导式是「声明式变换」的语法糖，编译成专用字节码（`LIST_APPEND`/`MAP_ADD`/`SET_ADD`），省去显式 `append` 的属性查找与函数调用，通常比等价 for 快约 **30%**；生成器表达式则惰性、内存恒定 O(1)。
- **关键 API / 语法**：列表 `[e for x in it if c]`、字典 `{k:v for ...}`、集合 `{e for ...}`、生成器 `(e for ...)`；末尾 `if` 是**过滤**，`for` 之前的 `if/else` 是**变换**（三元）；海象 `:=`（3.8+）在推导式里复用昂贵计算一次。
- **易错点**：① 生成器**一次性耗尽**、不可重复遍历、无 `len`/不可索引；② 惰性 + 闭包导致延迟绑定（`[lambda: i for i in range(3)]` 全返回 2）；③ 逻辑含副作用或嵌套 ≥3 层时退回普通 for——可读性优先，不是越短越好。
- **Java 视角**：列表/字典/集合推导式 ≈ `stream().map/filter().collect(toList/toMap/toSet)`；生成器表达式 ≈ 惰性 `Stream`（都只消费一次、终端操作才触发），但生成器可用 `yield` 自定义任意产出逻辑，比固定算子更通用。
- **前置**：01、02（容器）；与 05-advanced 的生成器相通。

---

## 🎯 学习要点

- **先记复杂度，再写代码**：判重/去重/查表一律用 set/dict（平均 O(1)），别用 `list in`（O(N)）；频繁头部操作用 deque；字符串拼接用 join。选错容器在数据管道里被放大百倍。
- **警惕三大浅拷贝坑**：别名不是拷贝、浅拷贝共享内层、`[[0]*n]*m` 是同一行——嵌套结构一律 `copy.deepcopy` 或推导式逐行新建。
- **用标准库子类替手写逻辑**：`Counter` 计数、`defaultdict(list)` 分组（≈ `Collectors.groupingBy`）、`OrderedDict` 做 LRU，零学习成本、零性能损失。
- **理解「保序」的来龙去脉**：dict 3.7+ 保序是语言规范（compact dict 实现），set 去重不保序，需要保序去重记住 `dict.fromkeys()`。
- **编码边界显式处理**：str↔bytes 在文件/网络/tokenizer 边界必须显式 encode/decode，`len` 区分码点与字节，大模型 byte-level BPE 尤其要留意。
- **推导式要「一眼读懂」**：能声明式表达就用推导式（并优先生成器表达式省内存喂给 sum/any/join），逻辑一复杂就退回普通 for。

---

## 🔗 关联

- **上一模块**：[02-syntax-comparison](../02-syntax-comparison/) — 数据结构操作建立在变量、循环、推导式语法之上。
- **下一模块**：[04-oop-in-python](../04-oop-in-python/) — 用类把数据结构封装成领域对象。
- **本阶段总览**：[阶段一 README](../README.md)
- **配套实战**：[agent-course/Day-18 chunking](../../agent-course/Day-18-chunking.md) — RAG 文本切块直接用到本模块的列表切片、字符串处理与推导式。
