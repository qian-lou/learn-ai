# 阶段一：Python 基础（Java 工程师快速过渡）

> **预估周期**：2-3 周
> **核心目标**：以 Java 为参照，快速补齐 Python 基础语法、数据结构、OOP 与高级特性，为后续数据科学、深度学习、大模型开发打底。

---

## 🧭 阶段学习地图

```
环境搭建 ──→ 语法过渡 ──→ 数据结构 ──→ 面向对象 ──→ 高级特性
（工具链）   （类型/控制流） （list/dict/set） （类/继承/魔术方法） （装饰器/生成器/异步）
   │            │              │                │                    │
 能跑起来    能读会写      能选对容器      能建模领域对象      能写出 Pythonic 工程代码
```

学习主线：**先能跑（环境）→ 再能写（语法）→ 会选数据结构（性能）→ 会建模（OOP）→ 写出地道高级代码（advanced）**。每个模块都以「Java 已有的概念」为锚点，只学差异，不重复学共性。

---

## 📋 模块大纲

### [01-environment-and-tools](./01-environment-and-tools/) — 环境与工具

搭建 Python 开发环境：用 pyenv 管理多版本、用 venv/conda/uv 隔离依赖、用 VS Code + Jupyter + Ruff 构建高效工具链。这是一切的起点，等价于 Java 里装 JDK + 配 Maven + 开 IDEA。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [python-installation-and-version](./01-environment-and-tools/01-python-installation-and-version.md) | Python 安装与版本管理（pyenv、CPython、GIL） |
| 02 | [virtual-environment](./01-environment-and-tools/02-virtual-environment.md) | 虚拟环境（venv/conda/uv、requirements.txt） |
| 03 | [ide-and-toolchain](./01-environment-and-tools/03-ide-and-toolchain.md) | IDE 与工具链（VS Code、Jupyter、Ruff、pdb） |

---

### [02-syntax-comparison](./02-syntax-comparison/) — 语法对比

以 Java 为参照逐点对比核心语法差异：动态类型 vs 静态类型、缩进 vs 花括号、一等公民函数、LEGB 作用域、运行时 import 机制与包管理。目标是把 Java 直觉平移到 Python，同时识别「不能这么想」的陷阱。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [variables-and-types](./02-syntax-comparison/01-variables-and-types.md) | 变量与类型系统（动态类型、一切皆对象、Type Hints） |
| 02 | [control-flow](./02-syntax-comparison/02-control-flow.md) | 控制流（缩进、for-each、match-case、真值判断） |
| 03 | [functions-and-scope](./02-syntax-comparison/03-functions-and-scope.md) | 函数与作用域（*args/**kwargs、闭包、LEGB） |
| 04 | [modules-and-packages](./02-syntax-comparison/04-modules-and-packages.md) | 模块与包管理（import 机制、循环导入、uv/pyproject.toml） |

---

### [03-data-structures](./03-data-structures/) — 数据结构

Python 内置数据结构详解，与 Java 集合框架逐一对比，重点讲**底层实现与复杂度心智模型**：list 动态数组的均摊 O(1)、dict/set 的开放寻址、compact dict 保序、字符串不可变与 join、推导式的字节码优势。选对容器是写高性能数据管道的前提。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [list-and-tuple](./03-data-structures/01-list-and-tuple.md) | 列表与元组（动态数组、切片、浅拷贝陷阱） |
| 02 | [dict-and-set](./03-data-structures/02-dict-and-set.md) | 字典与集合（开放寻址、compact dict、Counter/defaultdict） |
| 03 | [string-processing](./03-data-structures/03-string-processing.md) | 字符串处理（f-string、正则、编码、驻留） |
| 04 | [comprehensions](./03-data-structures/04-comprehensions.md) | 推导式（过滤 vs 变换、海象运算符、生成器表达式） |

---

### [04-oop-in-python](./04-oop-in-python/) — Python 面向对象

Python OOP 体系，与 Java 全面对比：显式 self、约定私有、@property、鸭子类型、多继承与 C3 MRO、魔术方法运算符重载、dataclass/Enum。PyTorch 的 `nn.Module`、Dataset 都建立在这套 OOP 之上，是后续所有建模代码的基础。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [class-and-object](./04-oop-in-python/01-class-and-object.md) | 类与对象（self、@property、类/静态方法） |
| 02 | [inheritance-and-polymorphism](./04-oop-in-python/02-inheritance-and-polymorphism.md) | 继承与多态（多继承、MRO、ABC、Mixin） |
| 03 | [magic-methods](./04-oop-in-python/03-magic-methods.md) | 魔术方法（\_\_repr\_\_/\_\_eq\_\_/\_\_getitem\_\_ 等） |
| 04 | [dataclass-and-enum](./04-oop-in-python/04-dataclass-and-enum.md) | dataclass 与枚举（@dataclass、frozen、Enum） |

---

### [05-advanced-features](./05-advanced-features/) — 高级特性

Python 高级编程特性，对标 Java 的 AOP、Stream、泛型、异步：装饰器（函数增强）、生成器（惰性求值）、上下文管理器（资源管理）、类型注解（静态检查）、asyncio（协程并发）。写出工程级、可维护、高性能代码的关键一跃。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [decorators](./05-advanced-features/01-decorators.md) | 装饰器（语法糖本质、带参装饰器、functools） |
| 02 | [generators-and-iterators](./05-advanced-features/02-generators-and-iterators.md) | 生成器与迭代器（yield、惰性求值、DataLoader） |
| 03 | [context-managers](./05-advanced-features/03-context-managers.md) | 上下文管理器（with、\_\_enter\_\_/\_\_exit\_\_、@contextmanager） |
| 04 | [type-hints](./05-advanced-features/04-type-hints.md) | 类型注解（泛型、Protocol、mypy、Pydantic） |
| 05 | [async-programming](./05-advanced-features/05-async-programming.md) | 异步编程（async/await、事件循环、并发控制） |

---

## 🎯 阶段学习要点

- **只学差异，不学共性**：Java 已有的 if/for/类/异常等概念直接平移，把精力集中在动态类型、缩进、一等公民函数、多继承、装饰器/生成器/协程这些「Java 没有或不一样」的点上。
- **建立性能心智模型**：Python 抽象层高，但容器复杂度不能靠感觉——list 头部插入 O(N)、dict/set 查表平均 O(1)、字符串循环拼接 O(N²)，选错容器在数据管道里会被放大百倍。
- **理解「一切皆对象 + 名字是标签」**：变量是指向对象的标签而非盒子，这条主线贯穿动态类型、可变默认参数陷阱、浅拷贝、闭包延迟绑定等一系列坑。
- **用工具替代自律**：Ruff（格式+lint）、mypy（静态类型）、Jupyter（交互探索）是标配，把 Java 里 IDEA 帮你做的事在 Python 里显式配起来。
- **动手翻译 Java 代码**：每个模块都用「Java 片段 → Python 等价写法」的方式练习，最快建立肌肉记忆；写完用 Ruff/mypy 校验。
- **面向大模型埋点**：本阶段的生成器（DataLoader）、dataclass（配置类）、asyncio（并发调 LLM）、Protocol（鸭子类型建模）都会在后续阶段反复出现，学时留意这些 AI 场景锚点。

---

## 🔗 关联

- **下一阶段**：[02-data-science-fundamentals](../02-data-science-fundamentals/) — 用 NumPy/Pandas/Matplotlib 做数据处理，直接建立在本阶段的列表/推导式/OOP 之上。
- **配套实战**：[agent-course/Day-01](../agent-course/Day-01-first-call.md)（第一次调用大模型 API，需要本阶段的环境与语法基础）；[Day-40 performance](../agent-course/Day-40-performance.md) 与 [Day-06 tool-calling](../agent-course/Day-06-tool-calling-basics.md) 会用到本阶段的 asyncio 并发。
