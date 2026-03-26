# 类与对象（Java 对比）
# Class and Object (vs Java)

## 1. 背景（Background）

> Python 是多范式语言，支持 OOP 但不强制。与 Java "一切皆类" 不同，Python 的 OOP 更灵活——没有访问修饰符、支持多继承、用约定代替强制规则。大模型中的 `nn.Module`（PyTorch 模型基类）就是经典的 OOP 应用。

## 2. 知识点（Key Concepts）

| 特性 | Java | Python |
|------|------|--------|
| 类定义 | `public class User {}` | `class User:` |
| 构造器 | `public User(String n)` | `def __init__(self, n):` |
| this/self | `this`（隐式） | `self`（显式！） |
| 访问控制 | `private/protected/public` | `_private` `__mangled`（约定） |
| 接口 | `interface` | `ABC`（抽象基类） |
| toString | `toString()` | `__str__()` / `__repr__()` |

## 3. 内容（Content）

### 3.1 类定义对比

```python
# Java:
# public class User {
#     private String name;
#     private int age;
#     public User(String name, int age) {
#         this.name = name; this.age = age;
#     }
#     public String getName() { return name; }
# }

# Python:
class User:
    """用户类 / User class."""
    
    def __init__(self, name: str, age: int) -> None:
        self.name = name   # 注意：必须显式写 self
        self.age = age
        self._email = None  # _前缀 = "约定私有"
    
    def greet(self) -> str:
        return f"Hi, I'm {self.name}, {self.age} years old"
    
    def __str__(self) -> str:  # 类似 toString()
        return f"User(name={self.name}, age={self.age})"
    
    def __repr__(self) -> str:  # 开发者友好的表示
        return f"User({self.name!r}, {self.age})"

user = User("Alice", 25)  # 不需要 new！
print(user.name)  # 直接访问（无需 getter）
print(user)        # User(name=Alice, age=25)
```

### 3.2 属性（Property）

```python
class Temperature:
    """用 @property 替代 Java 的 getter/setter."""
    
    def __init__(self, celsius: float) -> None:
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32

t = Temperature(100)
print(t.celsius)     # 100（像属性一样访问，实则调用 getter）
t.celsius = 200      # 调用 setter
print(t.fahrenheit)  # 392.0（计算属性）
```

### 3.3 类方法与静态方法

```python
class MathUtils:
    PI = 3.14159  # 类变量（类似 Java static final）
    
    @staticmethod  # Java: public static
    def add(a: int, b: int) -> int:
        return a + b
    
    @classmethod   # Java 无直接对应，类似工厂方法
    def from_string(cls, s: str) -> "MathUtils":
        return cls()

MathUtils.add(1, 2)  # 静态调用
```

## 4. 详细推理（Deep Dive）

- Python 没有 `private` 关键字，`_name` 是约定私有，`__name` 触发名称修饰（name mangling）
- `__init__` 不是构造器（constructor），而是初始化器（initializer），`__new__` 才是真正的构造器
- Python 的 `self` 必须显式写，Java 的 `this` 是隐式的

## 5. 例题（Worked Examples）

```python
# 实现一个 Java 风格的 Builder 模式
class QueryBuilder:
    def __init__(self):
        self._table = ""
        self._conditions = []
    
    def from_table(self, table: str) -> "QueryBuilder":
        self._table = table
        return self
    
    def where(self, condition: str) -> "QueryBuilder":
        self._conditions.append(condition)
        return self
    
    def build(self) -> str:
        sql = f"SELECT * FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        return sql

query = QueryBuilder().from_table("users").where("age > 18").where("active = 1").build()
```

## 6. 习题（Exercises）

**练习 1：** 将一个 Java POJO 类翻译为 Python 类，包含属性验证。

**练习 2：** 实现一个 `BankAccount` 类，支持存款、取款，余额不能为负。用 `@property` 实现 balance 的只读访问。
