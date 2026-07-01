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

## 4. 详细推理（Deep Dive）

- `dataclass` 通过自动添加底层 Python 类的内置魔术方法，消除了样板代码。
- `__post_init__` 用法：在自动生成的 `__init__` 函数执行完毕后，用来执行更高级的属性校验或派生字段计算。
- `Enum` 重复项防范：使用 `@unique` 装饰器强制所有枚举成员必须具有互不相同的唯一数值。

## 5. 例题（Worked Examples）

### 例题 1：使用 dataclass 声明包含数据约束校验的大模型配置类 / Configuration DTO with validations

本例题展示如何使用 `@dataclass` 声明配置类，并在 `__post_init__` 方法中对配置的参数范围合法性进行严格验证。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """大模型超参数配置类 / LLM Hyperparameters Config DTO.
    
    Time: O(1) creation / Instant setup.
    Space: O(1) memory / Single instance data.
    """
    model_name: str
    vocab_size: int = 32000
    d_model: int = 4096
    num_layers: int = 32
    dropout: Optional[float] = 0.1
    
    def __post_init__(self) -> None:
        # 在参数构建完后进行条件校验 / Parameter validation checks
        if self.vocab_size <= 0:
            raise ValueError(f"词表大小词频需大于0, 但收到 {self.vocab_size}")
        if self.d_model % 8 != 0:
            raise ValueError(f"模型维度必须是 8 的倍数, 但收到 {self.d_model}")

# 测试 / Test
try:
    config = ModelConfig(model_name="llama-8b", d_model=4095)  # 触发验证失败 / Validation fails.
except ValueError as e:
    print(f"参数验证捕获异常 / Expected validation catch: {e}")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：声明一个非可变的（frozen）`dataclass`，用于存储三维点空间坐标 `(x, y, z)`，尝试对其属性赋值观察异常。
*参考答案*：
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

# 测试
pt = Point3D(1.0, 2.0, 3.0)
try:
    pt.x = 10.0  # 会抛出 dataclasses.FrozenInstanceError 异常
except Exception as e:
    print(f"不可变类校验异常: {type(e)}")
```

### 进阶题
**练习 2**：在模型接口服务中，利用 `@dataclass` 接收用户传入的 JSON 请求参数。请实现一个将 `ModelConfig` 对象自动序列化为 Python 字典，以及从字典快速反序列化回配置对象的两个辅助函数。
*参考答案*：
```python
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ServingRequest:
    prompt: str
    max_tokens: int = 100

def to_dict(req: ServingRequest) -> Dict[str, Any]:
    """Time: O(N) properties, Space: O(N)"""
    return asdict(req)

def from_dict(data: Dict[str, Any]) -> ServingRequest:
    """Time: O(N) properties, Space: O(N)"""
    return ServingRequest(**data)
```