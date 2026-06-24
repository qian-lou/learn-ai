# 模块与包管理（Maven vs pip）
# Modules and Packages (Maven vs pip)

## 1. 背景（Background）

> Java 使用 Maven/Gradle + `import` 管理依赖，Python 使用 pip + `import`。核心差异：Java 的 `import` 是编译期行为，Python 的 `import` 是运行时执行。

## 2. 知识点（Key Concepts）

| 概念 | Java | Python |
|------|------|--------|
| 包管理器 | Maven/Gradle | pip/conda/uv |
| 导入语法 | `import java.util.List` | `import os` 或 `from os import path` |
| 包结构标识 | 目录自动识别 | `__init__.py`（3.3+ 可省略） |
| 入口点 | `public static void main` | `if __name__ == "__main__":` |

## 3. 内容（Content）

### 3.1 import 语法对比

```python
# Java: import java.util.List;
# Python:
import os                        # 导入整个模块
from os import path              # 导入模块中的特定成员
from os.path import join, exists  # 导入多个成员
import numpy as np               # 别名导入（Java 无此功能）
from typing import *             # 通配符导入（不推荐！）

# 导入顺序（Google Style）：标准库 → 第三方 → 本地
# Import order (Google Style): stdlib → third-party → local
import os
import sys

import numpy as np
import torch

from mypackage import utils
```

### 3.2 包结构

```
my_project/                    # Java: src/main/java/com/example/
├── __init__.py                # Java: package-info.java
├── models/
│   ├── __init__.py
│   └── user.py               # Java: User.java
├── services/
│   ├── __init__.py
│   └── user_service.py        # Java: UserService.java
└── utils/
    ├── __init__.py
    └── helpers.py
```

**`__init__.py` 的三种用途 / Three roles of `__init__.py`：**

```python
# my_project/models/__init__.py
# 用途 1：标识这是一个"常规包"（regular package），并在 import 时执行
# Role 1: Mark a "regular package"; its code runs on first import
# 类比 Java 的 package-info.java（但 package-info 不会执行，仅承载注解/文档）
# Analogy: Java's package-info.java (but that file never executes; it only holds annotations/docs)

# 用途 2：聚合并重导出子模块，对外提供扁平 API（门面 / facade）
# Role 2: Re-export submodules to expose a flat public API
from .user import User           # 现在可写 from my_project.models import User
from .order import Order

# 用途 3：用 __all__ 显式声明 `from package import *` 时导出的名字
# Role 3: __all__ controls what `from package import *` exports
__all__ = ["User", "Order"]      # 类比 Java module-info.java 的 exports 语句
                                  # Analogy: Java module-info.java `exports` clause
```

> Java 类比：`models/` 目录 ≈ Java 包 `com.example.models`；`__init__.py` 同时扮演 `package-info.java`（文档/聚合）和 `module-info.java`（`exports` 可见性控制）两个角色。区别在于 Python 的 `__init__.py` 是**运行时执行的真实代码**，可以放初始化逻辑、注册器、版本号等。

**绝对导入 vs 相对导入 / Absolute vs Relative imports：**

```python
# my_project/services/user_service.py

# 绝对导入（推荐）：从顶层包根算起，路径清晰、可读、可被工具静态分析
# Absolute import (PREFERRED): resolved from the top-level package root
from my_project.models.user import User
from my_project.utils.helpers import format_name

# 相对导入：从当前模块所在包算起。'.' = 当前包，'..' = 上一级包
# Relative import: '.' = current package, '..' = parent package
from .models.user import User        # 同级 services 看不到 models？不——根都是 my_project
from ..utils.helpers import format_name   # 上跳一级再进 utils

# Google Python Style Guide 明确推荐"绝对导入"，仅在包内部强耦合时用相对导入
# 类比 Java：Java 只有全限定名（绝对），没有相对导入这个概念
# Analogy: Java has only fully-qualified (absolute) names; no relative-import concept
```

> 关键规则：相对导入**只能在被当作包的一部分导入时使用**（即模块的 `__package__` 非空）。用 `python services/user_service.py` 直接运行带相对导入的文件会报 `ImportError: attempted relative import with no known parent package`——因为此时它的 `__name__` 是 `"__main__"`、`__package__` 为空。正确做法是 `python -m my_project.services.user_service`（以模块方式运行，保留包上下文）。

### 3.3 `__name__ == "__main__"` 模式

```python
# 类似 Java 的 public static void main
# 当文件被直接运行时执行，被 import 时不执行
def main():
    print("程序入口 / Program entry")

if __name__ == "__main__":
    main()
```

原理：每个模块都有一个 `__name__` 属性。**直接运行**该文件时（`python foo.py`），解释器把 `__name__` 设为字符串 `"__main__"`；该模块**被别处 import** 时，`__name__` 等于它的模块全名（如 `"mypkg.foo"`）。因此 `if __name__ == "__main__":` 块里的代码"仅在作为脚本启动时执行，作为库导入时静默"——这正对应 Java `public static void main(String[] args)` 的入口语义。差异：Java 的 `main` 是约定方法签名、由 JVM 指定主类调用；Python 是运行时变量判断，任何模块都可以"既是库又是脚本"。

### 3.4 `sys.path` 与 `PYTHONPATH`：导入到哪里找？

```python
import sys
# sys.path 是一个"搜索路径列表"，等价于 Java 的 classpath
# sys.path is the search-path list; the Python analog of the Java classpath
print(sys.path)
# 典型顺序 / Typical order:
#   ['', '/usr/lib/python3.12', '.../site-packages', ...]
#   ① '' 或脚本所在目录（最高优先级，易引发"同名遮蔽"陷阱）
#   ② 标准库目录
#   ③ 第三方安装目录 site-packages（pip/uv 装包落地处）

# 运行时动态追加（不推荐进生产，仅调试用）
# Append at runtime (debug only; prefer packaging over path hacks)
sys.path.append("/opt/extra_libs")
```

`sys.path` 的构造来源（优先级从高到低）：当前脚本目录 → 环境变量 `PYTHONPATH`（`export PYTHONPATH=/my/libs`，等价 `-cp` 指定额外 classpath）→ 解释器内置标准库路径 → `site-packages`。

> Java 对比：`sys.path` ≈ `classpath`；`PYTHONPATH` ≈ `CLASSPATH` 环境变量 / `java -cp`。陷阱：因为脚本所在目录排在最前，若你建了 `random.py`/`json.py` 这类与标准库同名的文件，`import random` 会**优先命中你的文件**，导致诡异错误——Java 因有包前缀（`java.util.Random`）天然规避了这个问题。

### 3.5 模块缓存 `sys.modules`：导入只执行一次

```python
import sys

# sys.modules 是"已加载模块"的全局缓存字典 {模块名: 模块对象}
# sys.modules is the global cache of already-imported modules {name: module}
import json
print("json" in sys.modules)        # True
print(sys.modules["json"])          # <module 'json' ...>

# 重复 import 同一模块：第二次起直接命中缓存，模块顶层代码不会再次执行
# Re-importing: cache hit; the module's top-level code does NOT run again
import json                          # 不重新执行 json.py 的顶层语句

# 强制重新加载（改了源码想热更新时用 importlib.reload）
import importlib
importlib.reload(json)               # 重新执行模块顶层代码 / re-runs top-level code
```

这解释了一个常见困惑："为什么我在多个文件里 `import config`，配置只初始化一次？"——因为模块对象被缓存为**进程级单例**。这与 Java 的"类只被 ClassLoader 加载一次、`static` 块只跑一次"语义高度一致；`sys.modules` 缓存 ≈ JVM 的类加载缓存。

### 3.6 `importlib`：动态导入（运行时按名加载）

```python
import importlib

# 按字符串名导入模块——等价 Java 的 Class.forName("com.example.Foo")
# Import a module by string name — analog of Java's Class.forName(...)
module_name = "mypkg.plugins.csv_loader"
mod = importlib.import_module(module_name)
loader_cls = getattr(mod, "Loader")     # 取出类 / fetch the class
instance = loader_cls()                 # 实例化 / instantiate

# 典型用途：插件系统、根据配置选择实现（策略模式的运行时装配）
# Use case: plugin systems, config-driven implementation selection
```

> Java 对比：`importlib.import_module(name)` ≈ `Class.forName(name)`；二者都是"反射式按名加载"，是插件框架（Spring 的 `@ComponentScan`、Python 的 entry-points）背后的基础设施。

### 3.7 命名空间包（Namespace Packages, PEP 420）

```
# 同一个逻辑包 acme，物理上分散在多个目录/多个 pip 包里——且都没有 __init__.py
# One logical package `acme` split across directories, NONE having __init__.py
path_a/acme/foo.py        # 由 acme-core 提供
path_b/acme/bar.py        # 由 acme-plugins 提供（独立发布的 pip 包）
```

PEP 420 起，**没有 `__init__.py` 的目录也能作为"命名空间包"被导入**，且其内容可由分散在 `sys.path` 多个位置的目录合并而成。这让大型组织把一个顶层命名空间（如 `acme.*`）拆成多个可独立发布的子包成为可能。

> Java 对比：等价于 Java 中"同一个 package `com.acme` 的 class 分散在多个 JAR 里"——JVM 会把它们合并视图。常规包（带 `__init__.py`）= 一个目录一个包；命名空间包 = 多目录合并。**业务代码建议老老实实写 `__init__.py`（常规包）**，命名空间包仅用于跨团队拆分大型命名空间的场景。

### 3.8 依赖管理：pip → 2026 推荐 uv + `pyproject.toml`

```bash
# 传统 pip（仍可用，但 2026 已非首选）/ Legacy pip (works, but no longer first choice)
pip install requests              # mvn install <dep>
pip install "requests==2.32.*"    # 指定版本 / pin version
pip install -r requirements.txt   # 按清单安装 ≈ mvn install (from pom)
pip list                          # ≈ mvn dependency:tree
```

**2026 推荐栈：`uv`（Rust 实现的高速包/项目管理器）+ `pyproject.toml`（PEP 621 标准元数据）。**

```toml
# pyproject.toml —— Python 工程的"单一事实来源"，地位等同 Java 的 pom.xml / build.gradle
# The single source of truth for a Python project; equivalent to pom.xml / build.gradle
[project]
name = "my-llm-app"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [          # ≈ <dependencies> in pom.xml
    "pandas>=2.2",
    "pyarrow>=16.0",
]

[project.optional-dependencies]      # ≈ Maven optional / profile-scoped deps
dev = ["pytest>=8.0", "ruff>=0.5"]

[build-system]
requires = ["hatchling"]             # 构建后端 ≈ Maven 的 build plugin
build-backend = "hatchling.build"
```

```bash
# uv 工作流 / uv workflow（比 pip 快 10–100×，并自带锁文件与虚拟环境管理）
uv init my-llm-app           # 初始化工程，生成 pyproject.toml ≈ mvn archetype:generate
uv add pandas pyarrow        # 加依赖并写入 pyproject.toml ≈ 编辑 pom + mvn install
uv add --dev pytest ruff     # 加开发依赖 ≈ Maven <scope>test</scope>
uv sync                      # 按 uv.lock 精确还原环境 ≈ mvn 依赖锁定（dependency:lock）
uv run python main.py        # 在受管虚拟环境中运行 ≈ mvn exec:java
```

> Java 对比一览：`pyproject.toml` ↔ `pom.xml`（声明式工程元数据 + 依赖）；`uv add` ↔ 编辑 `<dependencies>` 后 `mvn install`；`uv.lock`（锁定可复现版本）↔ Maven 的依赖解析锁定 / Gradle 的 `gradle.lockfile`；`uv` 自带的虚拟环境 ↔ 每个项目独立 classpath，避免全局污染。核心收益：**可复现（lock 文件）+ 隔离（venv）+ 极快（Rust）**，正好补齐了 pip 在工程化上的短板。

## 4. 详细推理（Deep Dive）

### 4.1 `import` 的底层机制：finder / loader 两段式

Java 的 `import` 是**编译期**符号解析，运行时由 ClassLoader 按 classpath 懒加载 `.class`。Python 的 `import` 是**运行时执行的一条语句**，背后是一套"导入协议"。当你写下 `import foo` 时，CPython 大致执行：

```
import foo 的内部流程 / The internal import algorithm:
  1. 查缓存：foo 在 sys.modules 里吗？
     ├─ 在 → 直接返回缓存的模块对象（这就是"只执行一次"的根因）
     └─ 不在 → 进入查找阶段
  2. 查找（Finding）：遍历 sys.meta_path 上的"查找器"（finder/MetaPathFinder）
        每个 finder 回答："我能定位 foo 吗？" 能则返回一个 ModuleSpec（含 loader）
        ├─ BuiltinImporter   → 内置模块（如 sys、_io，编译进解释器）
        ├─ FrozenImporter    → 冻结字节码
        └─ PathFinder        → 沿 sys.path 找 foo.py / foo/ 目录 / .so 扩展
  3. 加载（Loading）：用 spec 里的 loader.exec_module()
        ① 先在 sys.modules 里登记一个"空壳"模块对象（关键！见 4.2 循环导入）
        ② 把模块源码编译为字节码（首次会缓存到 __pycache__/foo.cpython-312.pyc）
        ③ 在该模块的命名空间里**自顶向下执行**全部顶层语句
  4. 绑定：把模块对象绑定到当前作用域的名字 foo
```

要点：**"查找"和"加载"职责分离**（finder 负责定位、loader 负责执行），这套可插拔协议正是 `importlib` 能实现"从 zip、从网络、从加密源"导入的扩展点——类比 Java 自定义 `ClassLoader` 改写类的加载来源。`__pycache__` 里的 `.pyc` 字节码缓存，则对应 JVM 把 `.java` 编译成 `.class`（区别：`.pyc` 是运行时按需生成的缓存，源码改动后会自动失效重编）。

### 4.2 循环导入（Circular Import）：成因与解法

理解了"加载阶段先登记空壳、再自顶向下执行"，循环导入就一目了然了。设 `a.py` 顶层 `from b import B`，而 `b.py` 顶层又 `from a import A`：

```python
# a.py
from b import B          # ① 执行 a 到这一行时，转去加载 b
class A: ...

# b.py
from a import A          # ② 加载 b 又转回 a，但 a 还停在第①行、A 尚未定义！
class B: ...             #    → ImportError: cannot import name 'A' from partially initialized module 'a'
```

成因：步骤 ②时，`a` 已在 `sys.modules` 中（空壳），但它的执行卡在第①行，`class A` 还没轮到执行，于是 `from a import A` 取不到名字 `A`，抛 `ImportError`（"partially initialized module"）。

**解法（按推荐度排序）/ Fixes (most to least preferred）：**

```python
# 解法 1（首选）：重构消除环——把 A、B 共同依赖的部分抽到第三个模块 common.py
# Fix 1 (BEST): break the cycle by extracting shared parts into a third module
# 类比 Java：两个类互相 import 通常是"职责未分清"的信号，提取接口/公共类解耦

# 解法 2：延迟导入（把 import 移进函数体，用到时才执行，绕开顶层时序）
# Fix 2: lazy import inside the function body, dodging top-level ordering
def make_b():
    from b import B      # 调用时 b 早已加载完毕，安全
    return B()

# 解法 3：import 整个模块而非具体名字，靠"运行时晚绑定"延后取属性
# Fix 3: import the module, not the name; resolve the attribute later (late binding)
import b                 # 顶层只拿到 b 模块对象（空壳也行）
def make_b():
    return b.B()         # 真正用到时 b.B 已存在
```

> Java 对比：Java 也允许两个类互相引用（编译器分两阶段解析符号，运行时按需加载），所以"互相 import"在 Java 里通常不报错；Python 因 `import` 是**顺序执行的运行时语句**而对时序敏感。但无论哪种语言，**双向依赖都是架构坏味道**——P3C/分层架构强调单向依赖（`Controller → Service → DAO`），从源头规避循环。

### 4.3 运行时导入的两个实用技巧

```python
# 条件导入：按环境/可用性选择实现（Java 难以等价表达，需反射 + try-catch）
try:
    import orjson as json     # 高性能 JSON（C 实现）/ faster JSON
except ImportError:
    import json               # 优雅降级到标准库 / graceful fallback

# 延迟导入：把重量级依赖（如 torch）推迟到真正调用时再加载，缩短启动时间
def run_inference():
    import torch              # 进程启动不付出 torch 的导入开销
    ...
```

## 5. 例题（Worked Examples）

### 例题 1：搭建一个可发布的包，用 `__init__.py` 暴露门面 API

需求：创建包 `calc_kit`，内含 `calculator`（业务）与 `utils`（工具）两个模块，对外只暴露 `add`、`safe_divide` 两个公共 API，并用 `pyproject.toml` 声明工程元数据。

```
calc_kit/
├── pyproject.toml          # 工程元数据 ≈ pom.xml
└── calc_kit/
    ├── __init__.py         # 门面：聚合并重导出公共 API / facade
    ├── calculator.py       # 业务逻辑 / business logic
    └── utils.py            # 工具函数 / utilities
```

```python
# calc_kit/utils.py
def is_zero(x: float) -> bool:
    """判断浮点数是否为零（含容差）/ Check near-zero with tolerance.

    Args:
        x: 待判断的浮点数 / the float to test.
    Returns:
        是否近似为零 / True if |x| < 1e-12.
    """
    # Time: O(1) Space: O(1)
    return abs(x) < 1e-12


# calc_kit/calculator.py —— 绝对导入，路径清晰可静态分析
from calc_kit.utils import is_zero    # 绝对导入（Google Style 推荐）

def add(a: float, b: float) -> float:
    """两数相加 / Add two numbers. Time: O(1) Space: O(1)."""
    return a + b

def safe_divide(a: float, b: float) -> float:
    """安全除法，除零时抛 ValueError / Safe division, raises on zero divisor.

    Args:
        a: 被除数 / dividend.
        b: 除数 / divisor.
    Returns:
        商 a / b / the quotient.
    Raises:
        ValueError: 当除数为零 / when divisor is zero.
    """
    # Time: O(1) Space: O(1)
    if is_zero(b):
        raise ValueError("除数不能为零 / divisor must not be zero")
    return a / b


# calc_kit/__init__.py —— 只暴露公共 API，隐藏内部模块结构
from calc_kit.calculator import add, safe_divide   # 重导出 / re-export
__all__ = ["add", "safe_divide"]                   # 控制 `import *` 的可见面
__version__ = "0.1.0"
```

```python
# 使用方只需面对扁平、稳定的门面，无需关心内部分了几个模块
# Consumers see a flat, stable facade; internal module split stays hidden
from calc_kit import add, safe_divide   # 而非 from calc_kit.calculator import ...
print(add(2, 3))                        # 5
print(safe_divide(10, 2))               # 5.0
```

> 设计要点：`__init__.py` 充当**门面（Facade）**——内部 `calculator.py` 将来拆成 `basic.py`/`advanced.py` 也不破坏 `from calc_kit import add` 的调用方。这等价于 Java 用 `module-info.java` 的 `exports` 精确控制对外可见的包，把实现细节封装在模块内部。

### 例题 2：定位并修复一个循环导入

现象：运行 `python -m shop.main` 报 `ImportError: cannot import name 'Order' from partially initialized module 'shop.order'`。

```python
# shop/order.py（出问题的版本 / buggy version）
from shop.user import User       # 顶层导入 user
class Order:
    def __init__(self, buyer: User) -> None:
        self.buyer = buyer

# shop/user.py（出问题的版本 / buggy version）
from shop.order import Order      # 顶层又导回 order → 形成环
class User:
    def history(self) -> list["Order"]:
        ...
```

诊断：`order` 加载到 `from shop.user import User` 时转去加载 `user`，`user` 又 `from shop.order import Order`，但此刻 `order` 仍是"半初始化"空壳、`Order` 尚未定义 → 报错。

```python
# 修复（解法 3：模块级导入 + 类型注解用字符串延后求值）
# Fix: import the module, defer the name; use string annotations for type hints
# shop/user.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:                # 仅类型检查时导入，运行时不执行 → 不形成运行时环
    from shop.order import Order # Imported only for type-checkers, never at runtime

class User:
    def history(self) -> "list[Order]":   # 字符串注解，运行时不立即解析 Order
        ...
# shop/order.py 保持 `from shop.user import User` 即可——环已被打破（单向）
```

> 关键技巧：`if TYPE_CHECKING:` 块的导入**只对类型检查器（mypy/pyright）可见，运行时被跳过**，配合字符串形式的类型注解（`"list[Order]"`），既保留了完整类型提示，又消除了运行时的循环依赖。根因仍是双向依赖——更彻底的做法是把 `User`/`Order` 的共享部分上提，使依赖单向化（呼应分层架构）。

## 6. 习题（Exercises）

**练习 1：** 解释 `import os` 和 `from os import *` 的区别和利弊。

*参考答案*：

- `import os`：只绑定 `os` 命名空间，使用时写 `os.getcwd()`。优点是不污染命名空间、来源清晰（类似 Java 的全限定名）。
- `from os import *`：把 `os` 的所有公开名字直接导入当前作用域，可直接写 `getcwd()`。缺点是污染命名空间、易与本地名冲突、可读性差、IDE 难追踪来源。
- 结论 / Verdict：生产代码避免 `import *`；如需精简可显式 `from os import path, getcwd`。

**练习 2：** 创建一个包含 3 个模块的 Python 包，模拟 Java 的 Service/DAO 分层结构。

*参考答案*：
```
user_app/
├── __init__.py            # 暴露公共 API / expose public API
├── model.py               # User 数据对象 / data object (类似 DO)
├── dao.py                 # UserDAO 数据访问 / data access
└── service.py             # UserService 业务逻辑 / business logic
```
```python
# dao.py
from .model import User
class UserDAO:                       # 数据层 / DAO layer
    def find_by_id(self, uid: int) -> User: ...

# service.py —— Service 调用 DAO，不跨层 / Service -> DAO, no layer skipping
from .dao import UserDAO
class UserService:
    def __init__(self) -> None:
        self._dao = UserDAO()
    def get_user(self, uid: int) -> User:
        return self._dao.find_by_id(uid)

# __init__.py
from .service import UserService     # 仅暴露入口 / expose entry only
```
