# 魔术方法（__init__, __str__ 等）
# Magic Methods (Dunder Methods)

## 1. 背景（Background）

> Python 的"魔术方法"（双下划线方法）类似 Java 的 `toString()`, `equals()`, `hashCode()`, `compareTo()` 等，但数量更多、功能更强大。它们让自定义类可以像内置类型一样使用运算符。

## 2. 知识点（Key Concepts）

| Java 方法 | Python 魔术方法 | 用途 |
|-----------|----------------|------|
| `toString()` | `__str__` / `__repr__` | 字符串表示 |
| `equals()` | `__eq__` | 相等判断 |
| `hashCode()` | `__hash__` | 哈希值 |
| `compareTo()` | `__lt__`, `__gt__` 等 | 比较运算 |
| `iterator()` | `__iter__` / `__next__` | 迭代 |
| - | `__len__` | `len()` 支持 |
| - | `__getitem__` | `obj[key]` 支持 |
| - | `__add__` | `+` 运算符 |

## 3. 内容（Content）

```python
class Vector:
    """二维向量，演示魔术方法 / 2D vector demonstrating magic methods."""
    
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    
    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __abs__(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __len__(self) -> int:
        return 2

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)       # Vector(4, 6) —— 自动调用 __add__
print(v1 * 3)         # Vector(3, 6) —— 自动调用 __mul__
print(abs(v2))        # 5.0
print(v1 == Vector(1, 2))  # True
```

## 4. 详细推理（Deep Dive）

- `__repr__` 用于开发调试（应能 `eval()` 还原对象），`__str__` 用于用户展示
- 实现 `__eq__` 时通常也需要实现 `__hash__`（否则对象不能放入 set/dict）
- `__getitem__` 使对象支持 `[]` 索引，这在 PyTorch Dataset 中广泛使用

## 5. 例题（Worked Examples）

### 例题 1：实现一个自定义矩阵类，支持运算符加法与元素索引 / Matrix class with add and getitem operator

本例演示如何使用 `__add__` 和 `__getitem__` 让自定义类支持类似 `matrix_1 + matrix_2` 以及 `matrix[row][col]` 运算。

```python
from typing import List

class SimpleMatrix:
    """支持常用魔术方法的二维矩阵类 / Simple 2D Matrix class supporting magic methods.
    
    Time: O(R * C) for operations, Space: O(R * C)
    """
    def __init__(self, data: List[List[float]]) -> None:
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        
    def __repr__(self) -> str:
        return f"SimpleMatrix({self.data})"
        
    def __getitem__(self, idx: int) -> List[float]:
        return self.data[idx]
        
    def __add__(self, other: "SimpleMatrix") -> "SimpleMatrix":
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩阵维度不一致，无法相加 / Dimension mismatch!")
        # 矩阵相加
        new_data = [
            [self.data[r][c] + other.data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        return SimpleMatrix(new_data)

# 测试 / Test
m1 = SimpleMatrix([[1.0, 2.0], [3.0, 4.0]])
m2 = SimpleMatrix([[5.0, 6.0], [7.0, 8.0]])
print(m1 + m2)     # SimpleMatrix([[6.0, 8.0], [10.0, 12.0]])
print(m1[0][1])    # 2.0 (自动调用 __getitem__)
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：实现一个具有 `__len__` 魔术方法的类，使其支持调用 `len(obj)` 来返回其内部列表的元素个数。
*参考答案*：
```python
from typing import List, Any

class CustomList:
    def __init__(self, items: List[Any]) -> None:
        self.items = items
    def __len__(self) -> int:
        return len(self.items)
```

### 进阶题
**练习 2**：在 PyTorch 的自定义数据集加载器中，必须继承并实现哪两个魔术方法才能配合 DataLoader 进行按 Batch 迭代加载？编写一段伪代码说明。
*参考答案*：
必须实现 `__len__`（返回数据集大小）和 `__getitem__`（支持按索引加载单条样本数据）。
```python
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, idx: int) -> torch.Tensor:
        # 加载第 idx 个样本，转换为张量返回
        return torch.tensor([ord(c) for c in self.texts[idx]])
```\n