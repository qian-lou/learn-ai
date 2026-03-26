# 类型注解（泛型对比）
# Type Hints (vs Generics)

## 1. 背景（Background）

> Python 3.5+ 支持类型注解，类似 Java 的泛型但更灵活。类型注解不影响运行时，通过 mypy 等工具进行静态检查。大模型项目中使用类型注解能大幅提升代码可维护性。

## 2. 知识点（Key Concepts）

| Java | Python |
|------|--------|
| `List<String>` | `list[str]` |
| `Map<String, Integer>` | `dict[str, int]` |
| `Optional<String>` | `str \| None` 或 `Optional[str]` |
| `<T extends Comparable>` | `TypeVar("T", bound=...)` |
| `? extends Number` | 协变/逆变 |

## 3. 内容（Content）

```python
from typing import TypeVar, Generic, Protocol

# 基本类型注解 / Basic type hints
def process(items: list[str], limit: int = 10) -> dict[str, int]:
    return {item: len(item) for item in items[:limit]}

# 泛型（类似 Java Generic） / Generics
T = TypeVar("T")

class Stack(Generic[T]):
    """泛型栈 / Generic stack."""
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

stack: Stack[int] = Stack()
stack.push(42)

# Protocol（结构类型，类似 Go 的 interface）
class Trainable(Protocol):
    def train(self, data: list) -> float: ...
    def predict(self, x: list) -> list: ...

# 任何实现了 train 和 predict 的类都满足 Trainable
# 不需要显式继承！（鸭子类型 + 类型检查）
```

## 4-6. 推理/例题/习题

**练习：** 为一个数据处理函数添加完整的类型注解，包含泛型和回调函数类型。
