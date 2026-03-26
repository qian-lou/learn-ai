# IDE 与开发工具链
# IDE and Development Toolchain

## 1. 背景（Background）

> **为什么要学这个？**
>
> 作为 Java 工程师，你可能习惯了 IntelliJ IDEA 的强大功能——智能补全、重构、调试。Python 开发也有同样完善的工具链。好的工具能让你的学习效率翻倍，特别是在大模型开发中，Jupyter Notebook 作为交互式编程环境几乎是标配。
>
> **在整个体系中的位置：** 工具链决定了你的开发体验和效率。选择合适的 IDE 和工具是高效学习的前提。

## 2. 知识点（Key Concepts）

| 工具类型 | Java 常用 | Python 推荐 | 说明 |
|----------|----------|-------------|------|
| IDE | IntelliJ IDEA | PyCharm / VS Code | 全功能开发环境 |
| 交互式环境 | JShell | Jupyter Notebook / IPython | 交互式编程，大模型必备 |
| 代码格式化 | Google Java Format | Black / Ruff | 代码格式统一 |
| 静态检查 | SpotBugs / PMD | Ruff / mypy | 代码质量检查 |
| 调试 | IDEA Debugger | pdb / IDE Debugger | 断点调试 |
| 文档生成 | Javadoc | Sphinx / mkdocs | API 文档 |

## 3. 内容（Content）

### 3.1 IDE 选择

#### VS Code（推荐 ✅）

优势：轻量、插件丰富、支持 Jupyter、远程开发完美

```
推荐安装的 VS Code 扩展：
┌────────────────────────────┬──────────────────────────────┐
│ 扩展                        │ 功能                          │
├────────────────────────────┼──────────────────────────────┤
│ Python (ms-python)          │ Python 语言支持（必装）        │
│ Pylance                     │ 智能补全与类型检查             │
│ Jupyter                     │ Jupyter Notebook 支持         │
│ Python Debugger              │ 调试工具                      │
│ Ruff                        │ 超快的 Linter + Formatter     │
│ Remote - SSH                │ 远程服务器开发（GPU 服务器）    │
│ GitHub Copilot              │ AI 代码补全                    │
└────────────────────────────┴──────────────────────────────┘
```

**VS Code 配置示例（settings.json）：**

```json
{
    // Python 相关配置
    // Python-related settings
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        }
    },
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    
    // Jupyter 配置
    // Jupyter settings
    "jupyter.askForKernelRestart": false,
    "notebook.formatOnSave.enabled": true
}
```

#### PyCharm

适合习惯 IntelliJ 全家桶的 Java 工程师：

```
PyCharm 版本对比：
┌──────────────┬────────────┬────────────────┐
│ 功能          │ Community  │ Professional   │
├──────────────┼────────────┼────────────────┤
│ Python 开发   │ ✅         │ ✅             │
│ Jupyter 支持  │ ✅         │ ✅             │
│ 远程开发      │ ❌         │ ✅             │
│ 数据库工具    │ ❌         │ ✅             │
│ Web 框架支持  │ ❌         │ ✅             │
│ 价格          │ 免费       │ 付费            │
└──────────────┴────────────┴────────────────┘

大模型学习阶段用 Community 版即可。
如果你有 JetBrains 全家桶许可证，直接用 Professional。
```

### 3.2 Jupyter Notebook

> **这是 Python 数据科学和大模型开发的核心工具，Java 世界没有对应物。**

```bash
# 安装 Jupyter
# Install Jupyter
pip install jupyter jupyterlab

# 启动 JupyterLab（推荐，比经典 Notebook 功能更强）
# Launch JupyterLab (recommended over classic Notebook)
jupyter lab

# 也可以在 VS Code 中直接创建 .ipynb 文件
# You can also create .ipynb files directly in VS Code
```

**Jupyter 的核心概念：**

```python
# Cell 1: Markdown 单元（文档说明）
# 这是一段说明文字，支持公式渲染
# $E = mc^2$

# Cell 2: Code 单元（代码执行）
import numpy as np
x = np.random.randn(100)
print(f"均值: {x.mean():.4f}")
# 输出直接显示在单元下方

# Cell 3: 可视化单元
import matplotlib.pyplot as plt
plt.hist(x, bins=20)
plt.title("随机数分布")
plt.show()
# 图表直接嵌入在笔记本中
```

**为什么大模型开发离不开 Jupyter？**
1. **交互式探索**：逐步执行代码，观察中间结果（训练过程中的 loss 值）
2. **富文本输出**：直接显示图表、表格、模型结构
3. **快速实验**：修改参数后立即重新执行某个 Cell，不用重跑整个脚本
4. **文档即代码**：Markdown + Code 混排，记录实验思路

### 3.3 代码质量工具

```bash
# Ruff —— 超快的 Python Linter + Formatter（Rust 编写）
# Ruff — Ultra-fast Python Linter + Formatter (written in Rust)
pip install ruff

# 检查代码
# Lint code
ruff check .

# 格式化代码（替代 Black）
# Format code (replaces Black)
ruff format .

# pyproject.toml 中配置 Ruff（类似 Maven 的 checkstyle 配置）
# Configure Ruff in pyproject.toml
```

```toml
# pyproject.toml 示例
# pyproject.toml example
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
# E: pycodestyle errors
# F: pyflakes
# I: isort (import sorting)
# N: pep8-naming
# W: pycodestyle warnings
# UP: pyupgrade

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### 3.4 调试技巧

```python
# Python 调试方式对比 Java
# Python debugging compared to Java

# 方式 1：print 调试（最简单，类似 System.out.println）
# Method 1: print debugging
print(f"变量值: {variable}")

# 方式 2：breakpoint()（Python 3.7+，类似 Java 条件断点）
# Method 2: breakpoint() (Python 3.7+)
def complex_function(data):
    result = process(data)
    breakpoint()  # 执行到这里会进入 pdb 交互调试
    return result

# 方式 3：IDE 图形化调试（与 IntelliJ 类似）
# Method 3: IDE graphical debugging (similar to IntelliJ)
# VS Code: 在行号左侧点击设置断点，按 F5 启动调试

# 方式 4：icecream 库（更好的 print 调试）
# Method 4: icecream library (better print debugging)
# pip install icecream
from icecream import ic
x = 42
ic(x)  # 输出: ic| x: 42（自动包含变量名！）
```

### 3.5 推荐项目结构

```
my-llm-project/                    # Java 对比
├── pyproject.toml                  # ← pom.xml
├── requirements.txt                # ← pom.xml (dependencies)
├── README.md                       # ← README.md
├── .gitignore                      # ← .gitignore
├── .python-version                 # ← (无对应)
├── src/                            # ← src/main/java/
│   └── my_project/
│       ├── __init__.py             # ← 包标识（类似 package-info.java）
│       ├── main.py                 # ← Application.java
│       ├── models/                 # ← models/
│       │   └── transformer.py
│       └── utils/                  # ← utils/
│           └── helpers.py
├── tests/                          # ← src/test/java/
│   ├── __init__.py
│   └── test_main.py
├── notebooks/                      # ← (无对应，Jupyter 笔记本)
│   └── experiment.ipynb
├── data/                           # ← (无对应，数据集)
│   └── train.csv
└── configs/                        # ← src/main/resources/
    └── config.yaml
```

## 4. 详细推理（Deep Dive）

### 4.1 IPython vs Python REPL

```
标准 Python REPL（类似 JShell）：
>>> 1 + 1
2
>>> # 功能非常有限

IPython（增强版 REPL，Jupyter 的内核）：
In [1]: 1 + 1
Out[1]: 2

In [2]: # 支持的特性：
        # 1. Tab 补全
        # 2. 魔术命令：%timeit, %debug, %matplotlib
        # 3. Shell 命令：!ls, !pwd
        # 4. 历史搜索
        # 5. 富文本输出
```

**常用 IPython 魔术命令：**

```python
# 性能测量（类似 Java 的 JMH 基准测试）
# Performance measurement (like Java's JMH benchmarks)
%timeit sum(range(1000000))
# 输出: 11.3 ms ± 123 µs per loop

# 内存分析
# Memory profiling
%load_ext memory_profiler
%memit np.zeros((1000, 1000))

# 运行外部脚本
# Run external script
%run script.py

# 查看变量类型和帮助
# View variable type and help
x = [1, 2, 3]
x?      # 查看 x 的类型和信息
x??     # 查看 x 的源码（如果可用）
```

## 5. 例题（Worked Examples）

### 例题 1：配置完整的开发环境

**问题：** 从零开始为一个大模型项目配置 VS Code 开发环境。

**解答：**

```bash
# 1. 创建项目
mkdir my-llm-project && cd my-llm-project

# 2. 设置 Python 版本
pyenv local 3.11.7

# 3. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 4. 安装开发依赖
pip install ruff pytest jupyter ipython icecream

# 5. 创建项目配置
cat > pyproject.toml << 'EOF'
[project]
name = "my-llm-project"
version = "0.1.0"
requires-python = ">=3.10"

[tool.ruff]
target-version = "py311"
line-length = 88
EOF

# 6. 创建 VS Code 配置
mkdir .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    }
}
EOF

# 7. 创建 .gitignore
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
EOF
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 在 VS Code 中创建一个 `.ipynb` Jupyter Notebook 文件，执行以下操作：
1. 创建一个 Markdown Cell，写上你的学习目标
2. 创建一个 Code Cell，导入 `sys` 并打印 Python 版本
3. 使用 `%timeit` 魔术命令测量 `sum(range(1000000))` 的执行时间

**练习 2：** 安装 Ruff，并配置 `pyproject.toml` 使其：
- 行宽限制为 120
- 启用 import 排序
- 目标 Python 版本为 3.11

### 进阶题

**练习 3：** 对比 Java 和 Python 的项目结构，画一张表格说明每个目录/文件的对应关系。思考：Python 为什么不需要 `src/main` 和 `src/test` 的分层？

**练习 4：** 使用 Jupyter Notebook 编写一个交互式的数据探索脚本：
1. 使用 `numpy` 生成 1000 个随机数
2. 计算均值和标准差
3. 画一个直方图
4. 使用 `%timeit` 比较 Python 原生 `sum()` 和 `numpy.sum()` 的性能差异
