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

### 3.3 `__name__ == "__main__"` 模式

```python
# 类似 Java 的 public static void main
# 当文件被直接运行时执行，被 import 时不执行
def main():
    print("程序入口 / Program entry")

if __name__ == "__main__":
    main()
```

### 3.4 pip 常用命令

```bash
pip install package_name          # mvn install
pip install package==1.2.3        # 指定版本
pip install -r requirements.txt   # mvn install (from pom.xml)
pip list                          # mvn dependency:tree
pip show package_name             # 查看包信息
pip uninstall package_name        # 卸载
pip install --upgrade package     # 升级
```

## 4. 详细推理（Deep Dive）

Python 的 `import` 在运行时执行，可以条件导入、延迟导入：
```python
# 条件导入（根据环境选择实现）
try:
    import ujson as json  # 更快的 JSON 库
except ImportError:
    import json           # 回退到标准库

# 延迟导入（避免循环导入）
def process():
    from heavy_module import heavy_function
    return heavy_function()
```

## 5. 例题（Worked Examples）

创建一个 Python 包，包含 `calculator` 模块和 `utils` 模块，通过 `__init__.py` 暴露公共 API。

## 6. 习题（Exercises）

**练习 1：** 解释 `import os` 和 `from os import *` 的区别和利弊。

**练习 2：** 创建一个包含 3 个模块的 Python 包，模拟 Java 的 Service/DAO 分层结构。
