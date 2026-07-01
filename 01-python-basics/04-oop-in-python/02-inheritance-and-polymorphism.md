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

## 5. 例题（Worked Examples）

### 例题 1：使用多继承与 Mixin 模式为神经网络模型添加日志功能 / Mixin Pattern in PyTorch Modules

本例模拟如何在 PyTorch 的模型定义中，使用 Mixin 模式动态为模型前向传播添加追踪日志。

```python
import torch
import torch.nn as nn
from typing import Any

class LoggingMixin:
    """日志混入类 / Logging Mixin class.
    
    Provides log method.
    """
    def log(self, message: str) -> None:
        print(f"[MODEL LOG] {message}")

class CustomLinear(nn.Module, LoggingMixin):
    """自定义线性层，多继承 nn.Module 和 LoggingMixin / Custom linear layer using multiple inheritance.
    
    Time: O(In * Out) per forward pass / Computational cost.
    Space: O(Out) / Output memory footprint.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()  # 调用父类初始化 / Call parents init.
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.log(f"输入形状 / Input shape: {list(x.shape)}")
        out = self.linear(x)
        self.log(f"输出形状 / Output shape: {list(out.shape)}")
        return out

# 测试多继承模型 / Test model
model = CustomLinear(10, 5)
x = torch.randn(2, 10)  # Shape: [2, 10]
y = model(x)            # Shape: [2, 5]
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：使用 ABC（抽象基类）定义一个接口 `Repository`，声明 `save()` 方法。实现它的具体子类 `DbRepository`。
*参考答案*：
```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def save(self, data: dict) -> None:
        pass

class DbRepository(Repository):
    def save(self, data: dict) -> None:
        print(f"Data saved to database: {data}")
```

### 进阶题
**练习 2**：推导 Python 的 MRO（方法解析顺序）关系：如果有类 `A`, `class B(A)`, `class C(A)`, `class D(B, C)`，请画出其继承结构并写出 `D.__mro__` 的输出结果。
*参考答案*：
这是一个典型的菱形继承。
- 继承结构：D 继承 B 和 C，B 和 C 分别继承 A。
- `D.__mro__` 的顺序为：`(__main__.D, __main__.B, __main__.C, __main__.A, object)`。这确保了父类 A 的方法只会被调用一次。