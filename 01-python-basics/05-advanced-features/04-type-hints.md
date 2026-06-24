# 类型注解（泛型对比）
# Type Hints (vs Generics)

## 1. 背景（Background）

> **为什么要学这个？**
>
> Python 3.5+ 支持类型注解，类似 Java 的泛型但更灵活。类型注解不影响运行时，通过 mypy 等工具进行静态检查。大模型项目中使用类型注解能大幅提升代码可维护性。
>
> 对于 Java 工程师来说，Python 的类型系统是**可选的**（不像 Java 强制），但在大型项目中强烈建议使用。

## 2. 知识点（Key Concepts）

| Java | Python | 说明 |
|------|--------|------|
| `List<String>` | `list[str]` | 泛型容器 |
| `Map<String, Integer>` | `dict[str, int]` | 字典类型 |
| `Optional<String>` | `str \| None` | 可空类型 |
| `<T extends Comparable>` | `TypeVar("T", bound=...)` | 有界泛型 |
| `interface` | `Protocol` | 结构化类型 |

## 3. 内容（Content）

### 3.1 基础类型注解

```python
# ============================================================
# 基本类型 / Basic types
# ============================================================
name: str = "Alice"
age: int = 30
score: float = 95.5
active: bool = True

# 函数签名 / Function signatures
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}! " * times

# 容器类型 (Python 3.9+) / Container types
items: list[str] = ["a", "b"]
scores: dict[str, float] = {"math": 95.0}
pair: tuple[int, str] = (1, "one")
unique: set[int] = {1, 2, 3}

# 可选类型 / Optional type (Python 3.10+)
def find_user(id: int) -> str | None:
    return "Alice" if id == 1 else None
```

### 3.2 泛型与高级类型

```python
from typing import TypeVar, Generic, Protocol, Callable
from collections.abc import Iterator

# ============================================================
# 泛型类（类似 Java Generic）
# ============================================================
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


# ============================================================
# Protocol（结构类型，类似 Go 的 interface）
# ============================================================
class Trainable(Protocol):
    """任何实现了 train/predict 的类自动满足此协议."""
    def train(self, data: list) -> float: ...
    def predict(self, x: list) -> list: ...

# 不需要显式继承！鸭子类型 + 类型检查
class LinearModel:
    def train(self, data: list) -> float:
        return 0.5
    def predict(self, x: list) -> list:
        return x

def evaluate(model: Trainable) -> None:
    """接受任何满足 Trainable 协议的对象."""
    model.train([])


# ============================================================
# 回调函数类型 / Callback types
# ============================================================
Callback = Callable[[str, int], bool]

def register(name: str, callback: Callback) -> None:
    callback(name, 0)


# ============================================================
# TypeAlias（类型别名）
# ============================================================
type Tensor = list[list[float]]  # Python 3.12+
type ModelConfig = dict[str, str | int | float]
```

### 3.3 mypy 静态检查

```bash
# 安装和使用
pip install mypy
mypy your_code.py --strict

# 常见配置 (pyproject.toml)
# [tool.mypy]
# python_version = "3.11"
# strict = true
# warn_return_any = true
```

## 4. 详细推理（Deep Dive）

```
Python vs Java 类型系统对比:

Java: 编译时强制检查，运行时擦除泛型
Python: 运行时完全忽略类型注解，需要 mypy 等工具检查

Python 类型注解的价值:
  1. IDE 自动补全和错误提示
  2. 文档作用（代码即文档）
  3. CI/CD 中集成 mypy 静态检查
  4. 大型项目协作的可维护性

大模型项目中的实践:
  - Tensor 形状注解: # Shape: [B, S, D]
  - 配置对象: Pydantic BaseModel（运行时校验）
  - API 参数: FastAPI 自动类型校验
```

## 5. 例题（Worked Examples）

```python
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    """自带运行时校验的配置类."""
    model_name: str
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 3

config = TrainingConfig(model_name="bert", learning_rate=2e-5)
```

## 6. 习题（Exercises）

### 基础题
**练习 1：** 为一个数据处理函数添加完整类型注解。

*参考答案*：
```python
from collections.abc import Iterable

def normalize(values: Iterable[float], lo: float = 0.0, hi: float = 1.0) -> list[float]:
    """Min-max 归一化 / min-max normalize.

    Args:
        values: 输入数值序列 / input numbers.
        lo: 目标区间下界 / target lower bound.
        hi: 目标区间上界 / target upper bound.
    Returns:
        归一化后的列表 / normalized list.
    """
    data = list(values)
    mn, mx = min(data), max(data)
    span = mx - mn or 1.0          # 防止除零 / avoid div-by-zero
    return [lo + (x - mn) / span * (hi - lo) for x in data]
```

**练习 2：** 用 Protocol 定义一个 `Serializable` 协议。

*参考答案*：
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Serializable(Protocol):
    """任何实现 to_json 的类自动满足 / structural typing, no inheritance needed."""
    def to_json(self) -> str: ...

class User:                         # 未显式继承 Serializable / no explicit inherit
    def to_json(self) -> str:
        return "{}"

def dump(obj: Serializable) -> str:  # 接受任何满足协议的对象 / duck-typed
    return obj.to_json()
```

### 进阶题
**练习 3：** 用 Pydantic 实现一个带校验的模型配置类。

*参考答案*：
```python
from pydantic import BaseModel, Field, field_validator

class ModelConfig(BaseModel):
    """运行时校验，类似 Java Bean Validation / runtime validation."""
    model_name: str
    learning_rate: float = Field(default=1e-4, gt=0)   # 必须 > 0 / must be positive
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=3, ge=1)

    @field_validator("model_name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name 不能为空 / must not be blank")
        return v

cfg = ModelConfig(model_name="bert", learning_rate=2e-5)  # 非法值会抛 ValidationError
```

**练习 4：** 配置 mypy strict 模式，修复项目中所有类型错误。

*参考答案*：
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true                 # 开启全部严格检查 / enable all strict flags
warn_return_any = true
```
```bash
mypy src/ --strict            # 逐条修复报错 / fix errors one by one
```
修复要点 / Key fixes：为所有函数补全参数与返回注解；用 `X | None` 替代裸 `None` 返回；消除 `Any`（改用 `TypeVar`/`Protocol`）；对外部无类型库加 `# type: ignore[import-untyped]` 或安装其 stubs。
