# dataclass 与枚举
# dataclass and Enum

## 1. 背景（Background）

> `dataclass`（Python 3.7+）类似 Java 14+ 的 `Record` 和 Lombok 的 `@Data`——自动生成 `__init__`, `__repr__`, `__eq__` 等。在大模型开发中常用于配置类和数据传输对象。

## 2. 知识点（Key Concepts）

| Java | Python |
|------|--------|
| `record User(String name, int age) {}` | `@dataclass class User:` |
| Lombok `@Data` | `@dataclass` |
| `enum Status { ACTIVE, INACTIVE }` | `class Status(Enum):` |

## 3. 内容（Content）

### 3.1 dataclass

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """训练配置 / Training configuration (like Java DTO)."""
    model_name: str
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    tags: list[str] = field(default_factory=list)

# 自动生成: __init__, __repr__, __eq__
config = TrainingConfig("bert-base", learning_rate=2e-5)
print(config)
# TrainingConfig(model_name='bert-base', learning_rate=2e-05, ...)

# frozen=True 使其不可变（类似 Java Record）
@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

### 3.2 枚举

```python
from enum import Enum, auto

class ModelType(Enum):
    BERT = auto()
    GPT = auto()
    T5 = auto()

class HttpStatus(Enum):
    OK = 200
    NOT_FOUND = 404
    ERROR = 500

# 使用
model = ModelType.BERT
print(model.name)   # "BERT"
print(model.value)  # 1

# match-case 配合
match model:
    case ModelType.BERT:
        print("使用 BERT 模型")
```

## 4-6. 推理/例题/习题

**练习：** 用 `@dataclass` 定义一个 `ModelConfig`，包含模型名、维度、层数、头数，加上参数验证（`__post_init__`）。
