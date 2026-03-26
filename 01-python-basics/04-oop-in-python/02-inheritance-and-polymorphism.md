# 继承与多态
# Inheritance and Polymorphism

## 1. 背景（Background）

> Java 只支持单继承 + 接口，Python 支持**多继承**。PyTorch 中 `nn.Module` 的继承是大模型开发的核心模式。

## 2. 知识点（Key Concepts）

| 特性 | Java | Python |
|------|------|--------|
| 单继承 | `extends` | `class Child(Parent):` |
| 多继承 | 不支持 | 支持！ |
| 接口 | `implements` | `ABC`（抽象基类） |
| super 调用 | `super.method()` | `super().method()` |
| 方法解析顺序 | 无（单继承） | MRO（C3 线性化） |

## 3. 内容（Content）

### 3.1 继承

```python
class Animal:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def speak(self) -> str:
        raise NotImplementedError

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name}: Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name}: Meow!"

# 多态 / Polymorphism
animals: list[Animal] = [Dog("Rex"), Cat("Kitty")]
for a in animals:
    print(a.speak())
```

### 3.2 抽象基类（替代 Java 接口）

```python
from abc import ABC, abstractmethod

class Repository(ABC):  # 类似 Java interface
    @abstractmethod
    def find_by_id(self, id: int) -> dict:
        ...
    
    @abstractmethod
    def save(self, entity: dict) -> None:
        ...

class UserRepository(Repository):  # 必须实现所有抽象方法
    def find_by_id(self, id: int) -> dict:
        return {"id": id, "name": "Alice"}
    
    def save(self, entity: dict) -> None:
        print(f"Saving {entity}")
```

### 3.3 多继承与 MRO

```python
class A:
    def method(self): return "A"

class B(A):
    def method(self): return "B"

class C(A):
    def method(self): return "C"

class D(B, C):  # 多继承！Java 不允许
    pass

d = D()
print(d.method())    # "B"（MRO 决定）
print(D.__mro__)     # D → B → C → A → object
```

### 3.4 PyTorch nn.Module 继承预览

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()  # 必须调用！
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
```

## 4. 详细推理（Deep Dive）

- MRO（方法解析顺序）使用 C3 线性化算法，保证一致性
- Mixin 模式：多继承的实用场景（如 `LoggingMixin`, `JsonMixin`）
- `super()` 在多继承中遵循 MRO 链

## 5. 例题 & 6. 习题

**练习 1：** 用 ABC 定义一个 `Shape` 接口，`Circle` 和 `Rectangle` 实现 `area()` 和 `perimeter()`。

**练习 2：** 实现 `LoggingMixin`，通过多继承为任何类添加日志功能。
