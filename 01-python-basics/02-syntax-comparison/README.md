# 02-syntax-comparison — 语法对比

> **所属阶段**：阶段一 · Python 基础
> **学习目标**：以 Java 为参照，逐点掌握 Python 核心语法差异——动态类型、缩进、一等公民函数、LEGB 作用域、运行时 import 机制，把 Java 直觉平移过来并识别陷阱
> **预估时长**：3-4 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [variables-and-types](./01-variables-and-types.md) | 变量与类型系统 | 动态 vs 静态类型的本质（名字是标签而非盒子）；一切皆对象、bool 是 int 子类、int 无上限；`is` vs `==`；小整数缓存；可选的 Type Hints |
| 02 | [control-flow](./02-control-flow.md) | 控制流 | 缩进代替花括号；只有 for-each（配 range/enumerate/zip）；for-else/while-else；三元 `b if a else c`；链式比较；Truthy/Falsy；match-case 模式匹配 |
| 03 | [functions-and-scope](./03-functions-and-scope.md) | 函数与作用域 | 返回多值（元组）；默认参数、`*args`/`**kwargs`；lambda 与高阶函数；LEGB 规则；闭包与 `nonlocal`；可变默认参数与延迟绑定陷阱 |
| 04 | [modules-and-packages](./04-modules-and-packages.md) | 模块与包管理 | import 是运行时语句；`__init__.py` 三种用途；绝对 vs 相对导入；`sys.path`/`sys.modules`；循环导入成因与解法；uv + pyproject.toml |

---

## 🔑 知识点详解

### 01 · 变量与类型系统

- **核心概念**：Python 变量不是「存值的盒子」而是「指向对象的标签」；对象自己知道类型，变量可随时改指别的类型。
- **关键 API / 心法**：判空用 `x is None`（不用 `==`）；类型检查 `isinstance(x, T)`；`==` 比值、`is` 比身份（内存地址）。
- **易错点**：① `b = a` 只是加一个标签指向同一对象，改可变对象（list/dict）会互相影响，需 `.copy()` 或 `copy.deepcopy()`；② 小整数 `[-5, 256]` 被缓存，`a is b` 对小整数为 True、大整数不定，**判等永远用 `==`**。
- **Java 视角**：动态类型 vs 静态类型；`None` ≈ `null`；`x | int` 联合类型无 Java 直接对应；Type Hints ≈ 可选的类型声明，但运行时不检查（要 mypy）。
- **前置**：无。

### 02 · 控制流

- **核心概念**：用缩进（4 空格）定义代码块，语法强制统一；`for` 只有 for-each 风格，C 风格计数循环用 `range()` 模拟。
- **关键 API / 语法**：`enumerate()` 带索引遍历、`zip()` 并行遍历、`dict.items()` 遍历键值；三元 `x if cond else y`；链式比较 `10 < x < 20`；`match-case`（3.10+）支持结构模式匹配与 guard。
- **易错点**：① Falsy 值（`None/False/0/""/[]/{}/()/set()`）可直接 `if items:` 判空，但 `x or default` 会把空串、0 也当默认值走掉；② `for-else` 的 `else` 属于 for（循环无 break 正常结束才执行），极易误读为 if 的 else。
- **Java 视角**：`elif` ≈ `else if`；`match-case` ≈ 增强版 `switch`（还能解构元组/字典）；`pass` ≈ 空语句 `{}`；无 `i++`，用 `i += 1`。
- **前置**：01（真值判断依赖类型知识）。

### 03 · 函数与作用域

- **核心概念**：函数是一等公民（可赋值、传参、返回），无需 Java 的 `@FunctionalInterface`；作用域按 LEGB（Local → Enclosing → Global → Built-in）逐层查找。
- **关键 API / 语法**：`def f(x, y=10)` 默认参数、`*args` 收位置参数、`**kwargs` 收关键字参数；`lambda x: x*2`；闭包内改外层变量用 `nonlocal`，改全局用 `global`。
- **易错点**：① 可变对象作默认参数（`def f(items=[])`）会在多次调用间共享同一个 list——用 `None` 哨兵再在体内新建；② 循环里创建 lambda 闭包按引用捕获变量，循环结束后都读到最终值（延迟绑定），用默认参数 `lambda x, i=i: ...` 当场快照修复。
- **Java 视角**：`lambda` ≈ Java Lambda；返回多值（元组解构）≈ 返回 DTO 但更轻；Python 是「对象引用传递」——不可变对象表现如值传递、可变对象表现如引用传递。
- **前置**：01（引用/可变性）、02（控制流）。

### 04 · 模块与包管理

- **核心概念**：Java 的 import 是编译期符号解析；Python 的 `import` 是**运行时执行的一条语句**，走「查缓存 → finder 查找 → loader 执行模块顶层代码 → 绑定名字」的协议，因此有循环导入这类时序问题。
- **关键 API / 语法**：`import x` / `from x import y` / `import x as z`；`if __name__ == "__main__":` 入口；`sys.path`（≈ classpath）、`sys.modules`（进程级模块缓存，导入只执行一次）；`importlib.import_module(name)` 动态导入。
- **易错点**：① 建了与标准库同名的 `random.py`/`json.py`，因脚本目录排 `sys.path` 最前会遮蔽标准库；② 循环导入报 `partially initialized module`——首选重构解耦，或延迟导入 / 用 `if TYPE_CHECKING:` + 字符串注解；③ 直接 `python foo.py` 跑带相对导入的文件会报错，应 `python -m pkg.foo`。
- **Java 视角**：pip/uv ≈ Maven/Gradle；`pyproject.toml` ≈ `pom.xml`；`__init__.py` ≈ `package-info.java`（文档/聚合）+ `module-info.java`（`exports` 可见性）；`importlib.import_module` ≈ `Class.forName`；`sys.modules` 缓存 ≈ JVM 类加载缓存。
- **前置**：01-03（模块内会用到类型、函数等）。

---

## 🎯 学习要点

- **把「名字是标签」当第一性原理**：动态类型、可变默认参数、浅拷贝、闭包延迟绑定这四个高频坑，根因都是同一句话——变量指向对象、可变对象被共享。
- **用「Java 片段 → Python 翻译」建立肌肉记忆**：把 `for(int i=0;...)` 翻成 `range/enumerate`、`Stream.map().filter()` 翻成推导式、三元和 switch 逐一对照，写完用 Ruff 校验风格。
- **牢记 `is` 与 `==` 的分工**：判空、判单例用 `is None`；判内容相等一律 `==`，不要因小整数/字符串驻留的巧合而误用 `is`。
- **默认参数只用不可变值**：需要可变默认值时用 `None` 哨兵在函数体内新建，这是 Python 代码评审的必查项。
- **理解 import 是运行时行为**：这解释了「配置为何只初始化一次」（`sys.modules` 单例）和循环导入报错；工程上坚持单向依赖（Controller→Service→DAO）从源头规避环。
- **2026 用 uv + pyproject.toml 起项目**：`uv init/add/sync/run` 一条龙，自带锁文件与虚拟环境，是 pip 的工程化升级版。

---

## 🔗 关联

- **上一模块**：[01-environment-and-tools](../01-environment-and-tools/) — 先搭好环境才能跑这些语法示例。
- **下一模块**：[03-data-structures](../03-data-structures/) — 语法就绪后深入 list/dict/set 的实现与性能。
- **本阶段总览**：[阶段一 README](../README.md)
- **配套实战**：[agent-course/Day-01](../../agent-course/Day-01-first-call.md) — 用本模块的函数、import、入口写法完成第一次 API 调用脚本。
