# 04-oop-in-python — Python 面向对象

> **所属阶段**：阶段一 · Python 基础
> **学习目标**：掌握 Python OOP 体系并与 Java 全面对比——显式 self、约定私有、@property、鸭子类型、多继承与 MRO、魔术方法、dataclass/Enum，为 PyTorch 建模打底
> **预估时长**：3-4 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [class-and-object](./01-class-and-object.md) | 类与对象 | `class`/`__init__`（初始化器非构造器）；self 必须显式；无 new；`_`/`__` 约定私有与 name mangling；`@property` 替代 getter/setter；`@staticmethod`/`@classmethod` |
| 02 | [inheritance-and-polymorphism](./02-inheritance-and-polymorphism.md) | 继承与多态 | 多继承（Java 不支持）；`ABC`/`@abstractmethod` 替代接口；`super()` 遵循 MRO；C3 线性化解菱形继承；Mixin 模式；`nn.Module` 继承预览 |
| 03 | [magic-methods](./03-magic-methods.md) | 魔术方法 | `__str__` vs `__repr__`；`__eq__`+`__hash__` 成对；`__add__`/`__mul__` 运算符重载；`__len__`/`__getitem__` 让对象类容器化（PyTorch Dataset 必备） |
| 04 | [dataclass-and-enum](./04-dataclass-and-enum.md) | dataclass 与枚举 | `@dataclass` 自动生成 init/repr/eq；`field(default_factory=...)`；`frozen=True` 不可变；`__post_init__` 校验；`Enum`/`auto()`/`@unique` |

---

## 🔑 知识点详解

### 01 · 类与对象

- **核心概念**：Python OOP 灵活不强制——没有访问修饰符、用约定代替强制；`self` 必须显式声明为第一个参数；实例化无需 `new`。
- **关键 API / 语法**：`def __init__(self, ...)` 初始化；`@property` + `@x.setter` 把方法伪装成属性（替代 getter/setter）；`@classmethod`（收 `cls`，常作工厂方法）、`@staticmethod`（不收 self/cls）。
- **易错点**：① 忘写 `self`；② 以为 `__init__` 是构造器——真正的构造器是 `__new__`，`__init__` 只是初始化已创建的对象；③ `_name` 只是约定私有（外部仍可访问），`__name` 才触发 name mangling（改名为 `_Class__name`）。
- **Java 视角**：`self` ≈ 显式的 `this`；`@property` ≈ getter/setter 但调用方像访问字段；`@classmethod` ≈ 工厂静态方法；类变量 ≈ `static` 字段。
- **前置**：02-syntax（函数、装饰器语法）。

### 02 · 继承与多态

- **核心概念**：Python 支持**多继承**，方法查找按 **MRO**（C3 线性化算法）确定顺序，保证菱形继承里公共父类只被调用一次；鸭子类型让「行为一致即可替换」，无需共同接口。
- **关键 API / 语法**：`class Child(Parent):`；`super().__init__()`（遵循 MRO 链，PyTorch 子类必须调用）；`from abc import ABC, abstractmethod` 定义抽象基类（替代 Java 接口）；`Cls.__mro__` 查看解析顺序。
- **易错点**：① 多继承忘调 `super().__init__()` 导致父类状态未初始化（`nn.Module` 尤其致命）；② 菱形继承里方法解析顺序反直觉，需靠 `__mro__` 确认而非猜测；③ 未实现全部 `@abstractmethod` 的子类无法实例化。
- **Java 视角**：`extends` ≈ `class Child(Parent)`；`implements` ≈ 继承 `ABC`；`super.method()` ≈ `super().method()`；Java 无多继承（用接口 + 默认方法），Python 的 Mixin ≈ 组合能力的「横切类」（如 `LoggingMixin`）。
- **前置**：01（类基础）。

### 03 · 魔术方法

- **核心概念**：双下划线方法（dunder）让自定义类能像内置类型一样参与运算符、`len()`、`[]` 索引、迭代等协议；数量比 Java 的 `toString/equals/hashCode/compareTo` 多得多。
- **关键 API / 语法**：`__repr__`（开发者友好、应能 eval 还原）vs `__str__`（用户展示）；`__eq__` 与 `__hash__` **必须成对**实现（否则对象无法进 set/dict）；`__add__`/`__mul__` 重载 `+`/`*`；`__len__`/`__getitem__` 让对象支持 `len()` 和 `obj[i]`。
- **易错点**：① 只实现 `__eq__` 不实现 `__hash__`，对象放进 set/dict 行为异常；② `__eq__` 里类型不匹配应返回 `NotImplemented`（大写，交给对方处理）而非 `False`；③ 混淆 `__str__`/`__repr__` 的用途。
- **Java 视角**：`__str__`/`__repr__` ≈ `toString()`；`__eq__` ≈ `equals()`；`__hash__` ≈ `hashCode()`；`__lt__` 等 ≈ `compareTo()`；`__getitem__`/`__len__` 无直接对应（Python 用协议而非接口实现容器化）。
- **前置**：01（类）、03-data（可哈希契约与 dict/set）。

### 04 · dataclass 与枚举

- **核心概念**：`@dataclass`（3.7+）按字段声明自动生成 `__init__`/`__repr__`/`__eq__`，消灭样板代码；`Enum` 提供类型安全的具名常量。
- **关键 API / 语法**：`@dataclass` + 类型注解字段；可变默认值必须用 `field(default_factory=list)`；`@dataclass(frozen=True)` 变不可变（≈ record）；`__post_init__` 做校验/派生字段；`Enum` 成员有 `.name`/`.value`，`auto()` 自动赋值，`@unique` 禁重复值，`asdict()` 序列化。
- **易错点**：① 用可变对象直接当默认值（`tags: list = []`）会在实例间共享——必须 `field(default_factory=list)`；② 校验逻辑要写在 `__post_init__` 里（`__init__` 是自动生成的）；③ 有默认值的字段必须排在无默认值字段之后。
- **Java 视角**：`@dataclass` ≈ Lombok `@Data`；`@dataclass(frozen=True)` ≈ Java 14+ `record`；`Enum` ≈ Java `enum`；`asdict`/`from_dict` ≈ 序列化/反序列化（配合 FastAPI/Pydantic 做 API DTO）。
- **前置**：01（类）、03（魔术方法）；类型注解见 05-advanced/04。

---

## 🎯 学习要点

- **接受「约定优于强制」**：Python 无 private 关键字，靠 `_`/`__` 命名约定和文档自律；别试图用 Java 的访问修饰符思维硬套。
- **优先鸭子类型 + Protocol，谨慎用继承**：需要「行为契约」时用 `ABC`（强约束）或 `Protocol`（结构化、无需显式继承），而非层层继承；多继承只在 Mixin 这类横切能力时使用。
- **多继承必调 super().__init__()**：尤其继承 `nn.Module` 时不调用会导致模型参数注册失败；理解 `super()` 走的是 MRO 链而非直接父类。
- **`__eq__` 与 `__hash__` 成对出现**：这是把对象放进 set/做 dict 键的前提，与 03-data 的可哈希契约一脉相承。
- **配置类/DTO 一律用 dataclass**：训练超参、请求参数用 `@dataclass`（+ `__post_init__` 校验）或 Pydantic，比手写 `__init__` 清晰且不易错；可变默认值记得 `field(default_factory=...)`。
- **为 PyTorch 建模埋点**：`nn.Module` 的 `__init__`/`forward`、Dataset 的 `__len__`/`__getitem__`、配置 dataclass，都是本模块知识在大模型代码里的直接落地，学时对照记忆。

---

## 🔗 关联

- **上一模块**：[03-data-structures](../03-data-structures/) — 魔术方法的可哈希契约、dataclass 字段都建立在数据结构之上。
- **下一模块**：[05-advanced-features](../05-advanced-features/) — 装饰器（@dataclass 本身就是装饰器）、生成器、上下文管理器是 OOP 的进阶延伸。
- **本阶段总览**：[阶段一 README](../README.md)
- **配套实战**：[agent-course/Day-04 structured-output](../../agent-course/Day-04-structured-output.md) — 用 dataclass/Pydantic 建模结构化输出；[Day-07 agents-sdk](../../agent-course/Day-07-agents-sdk-first-tool.md) 的工具定义也依赖 OOP 建模。
