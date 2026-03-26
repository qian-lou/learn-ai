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

## 5. 例题 & 6. 习题

**练习：** 实现一个 `Matrix` 类，支持 `+`、`*`、`len()`、下标访问 `m[i][j]`。
