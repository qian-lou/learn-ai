# Python 安装与版本管理
# Python Installation and Version Management

## 1. 背景（Background）

> **为什么要学这个？**
> 
> 作为 Java 工程师，你习惯了 JDK 的版本管理（Java 8/11/17/21）。Python 的版本管理同样重要，但机制完全不同。Java 通过 `JAVA_HOME` 切换版本，Python 的生态更加碎片化——系统自带 Python、Homebrew 安装的 Python、Anaconda 的 Python 可能同时并存，如果不理解版本管理，后续安装 PyTorch、TensorFlow 等库时会踩很多坑。
>
> **在整个体系中的位置：** 这是一切的起点。正如写 Java 需要先安装 JDK，学 Python 大模型首先需要正确安装和管理 Python 环境。

## 2. 知识点（Key Concepts）

| 概念 | Java 对应 | Python 对应 | 说明 |
|------|-----------|-------------|------|
| 运行时 | JRE/JDK | CPython | Python 的默认实现 |
| 版本管理 | jenv / sdkman | pyenv | 多版本共存管理工具 |
| 包管理 | Maven / Gradle | pip / conda | 依赖管理工具 |
| 版本号 | Java 8, 11, 17, 21 | Python 3.8, 3.9, 3.10, 3.11, 3.12 | 主流版本 |

**核心要点：**
- Python 2 已于 2020 年停止维护，**只学 Python 3**
- 大模型开发推荐 Python **3.10 或 3.11**（兼容性最佳）
- macOS/Linux 自带的 Python **不要动**，用 pyenv 管理独立版本

## 3. 内容（Content）

### 3.1 Python 版本选择

```
Python 版本选择决策树：
                    ┌─ 需要最新特性？ ─→ Python 3.12
                    │
你要学大模型 ──────┼─ 需要最佳兼容性？ ─→ Python 3.10 / 3.11 ✅ 推荐
                    │
                    └─ 维护老项目？ ─→ Python 3.8 / 3.9
```

**为什么推荐 3.10/3.11？**
- PyTorch、TensorFlow 对这两个版本支持最稳定
- Hugging Face Transformers 官方测试基于 3.10+
- 3.10 引入了 `match-case` 语法（类似 Java 的 `switch` 增强版）

### 3.2 安装方式对比

| 安装方式 | 优点 | 缺点 | 推荐场景 |
|----------|------|------|----------|
| 官网下载 | 简单直接 | 不方便多版本管理 | 初学入门 |
| pyenv | 多版本管理灵活 | 需要命令行操作 | 专业开发 ✅ |
| Anaconda | 自带数据科学包 | 体积大（3GB+），可能与 pip 冲突 | 数据科学探索 |
| Miniconda | 轻量 conda | 需要手动安装包 | 生产环境 ✅ |

### 3.3 pyenv 安装与使用（macOS）

```bash
# 安装 pyenv（使用 Homebrew）
# Install pyenv via Homebrew
brew install pyenv

# 配置 shell（zsh 用户）
# Configure shell for zsh users
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 查看可安装的 Python 版本
# List available Python versions
pyenv install --list | grep "3\.\(10\|11\)"

# 安装 Python 3.11.7
# Install Python 3.11.7
pyenv install 3.11.7

# 设置全局默认版本
# Set global default version
pyenv global 3.11.7

# 验证
# Verify installation
python --version  # 应该输出 Python 3.11.7
which python      # 应该指向 ~/.pyenv/shims/python
```

### 3.4 pyenv 常用命令速查

```bash
# 类比 Java 的 sdkman 命令
# Analogous to Java's sdkman commands

pyenv versions          # 查看已安装版本（类似 sdkman list java）
pyenv install 3.10.13  # 安装指定版本
pyenv uninstall 3.10.13 # 卸载指定版本
pyenv global 3.11.7    # 设置全局版本
pyenv local 3.10.13    # 设置当前目录版本（生成 .python-version 文件）
pyenv shell 3.10.13    # 设置当前 shell 会话版本
```

### 3.5 验证安装

```python
# verify_install.py
# 验证 Python 安装是否正确
# Verify Python installation
import sys
import platform

print(f"Python 版本 / Python Version: {sys.version}")
print(f"Python 路径 / Python Path: {sys.executable}")
print(f"操作系统 / OS: {platform.system()} {platform.release()}")
print(f"架构 / Architecture: {platform.machine()}")

# 验证 pip 可用
# Verify pip is available
import subprocess
result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                       capture_output=True, text=True)
print(f"pip 版本 / pip Version: {result.stdout.strip()}")
```

## 4. 详细推理（Deep Dive）

### 4.1 Python 的版本管理为什么比 Java 复杂？

**Java 的简单性：**
- JDK 向后兼容性极好（Java 8 代码基本能在 Java 21 上跑）
- `JAVA_HOME` 一个环境变量搞定全部
- Maven/Gradle 管理依赖，与 JDK 版本解耦

**Python 的复杂性：**
- Python 3.x 各小版本间也有不兼容变化（如 3.10 引入 `match-case`）
- C 扩展库（NumPy、PyTorch）必须与 Python 版本**精确匹配**
- 系统级 Python 可能被其他系统工具依赖，改动风险高
- `pip install` 会安装到全局，不同项目依赖可能冲突

### 4.2 CPython vs 其他实现

```
Python 实现对比：
┌──────────┬──────────────────────────┬──────────────────┐
│ 实现      │ 说明                      │ 类比 Java        │
├──────────┼──────────────────────────┼──────────────────┤
│ CPython  │ 官方 C 语言实现，最常用     │ HotSpot JVM      │
│ PyPy     │ JIT 编译，纯 Python 代码快  │ GraalVM          │
│ Jython   │ 运行在 JVM 上的 Python     │ -                │
│ Cython   │ Python 到 C 的编译器        │ JNI              │
└──────────┴──────────────────────────┴──────────────────┘

大模型开发固定使用 CPython，因为 PyTorch/TensorFlow 的 C++ 后端
只兼容 CPython。
```

### 4.3 GIL（全局解释器锁）简述

这是 Python 与 Java 最大的并发差异之一：

```python
# Python 的 GIL 限制了真正的多线程并行
# Python's GIL limits true multi-threaded parallelism

# Java 中，多线程可以真正并行执行 CPU 密集型任务
# In Java, multiple threads can truly run in parallel on CPU-intensive tasks

# Python 中，同一时刻只有一个线程执行 Python 字节码
# In Python, only one thread executes Python bytecode at a time

# 解决方案：
# Solutions:
# 1. 多进程（multiprocessing）—— 每个进程有独立 GIL
# 2. C 扩展（NumPy/PyTorch）—— 在 C 层面释放 GIL
# 3. asyncio —— I/O 密集场景用协程
# 4. Python 3.13+ 引入 free-threaded 模式（实验性）
```

> **对 Java 工程师的启示：** 不用太担心 GIL。在大模型开发中，计算密集部分由 PyTorch（C++ 后端）处理，GIL 不是瓶颈。

## 5. 例题（Worked Examples）

### 例题 1：多版本 Python 环境配置

**问题：** 你有两个项目：
- 项目 A 使用 Python 3.10（兼容老版本 TensorFlow）
- 项目 B 使用 Python 3.11（使用最新 PyTorch）

请配置 pyenv 使两个项目使用不同版本。

**解答：**

```bash
# 步骤 1：安装两个版本
# Step 1: Install both versions
pyenv install 3.10.13
pyenv install 3.11.7

# 步骤 2：在项目 A 目录设置 local 版本
# Step 2: Set local version for project A
cd ~/projects/projectA
pyenv local 3.10.13
python --version  # Python 3.10.13
cat .python-version  # 会看到 3.10.13

# 步骤 3：在项目 B 目录设置 local 版本
# Step 3: Set local version for project B
cd ~/projects/projectB
pyenv local 3.11.7
python --version  # Python 3.11.7

# 步骤 4：验证切换
# Step 4: Verify switching
cd ~/projects/projectA && python --version  # 3.10.13
cd ~/projects/projectB && python --version  # 3.11.7
```

### 例题 2：诊断 Python 环境问题

**问题：** 运行 `python` 显示 Python 2.7，但你确认已安装 Python 3.11，如何排查？

**解答：**

```bash
# 排查步骤
# Troubleshooting steps

# 1. 查看 python 指向哪里
# 1. Check where python points to
which python
# 如果输出 /usr/bin/python，说明用的是系统自带版本

# 2. 查看 PATH 中 python 的搜索顺序
# 2. Check python search order in PATH
echo $PATH | tr ':' '\n' | head -10

# 3. 查看 pyenv 是否正确初始化
# 3. Check if pyenv is properly initialized
pyenv versions
pyenv which python

# 4. 修复：确保 pyenv 的 shims 在 PATH 最前面
# 4. Fix: Ensure pyenv shims are at the front of PATH
# 在 ~/.zshrc 中确保 eval "$(pyenv init -)" 在最后
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 使用 pyenv 安装 Python 3.11，并验证以下信息：
- Python 版本号
- pip 版本号
- Python 可执行文件的路径

**练习 2：** 解释以下命令的区别：
```bash
pyenv global 3.11.7
pyenv local 3.11.7
pyenv shell 3.11.7
```

### 进阶题

**练习 3：** 你在一个目录下同时设置了 `pyenv local 3.10.13`（生成 `.python-version` 文件）和 `pyenv shell 3.11.7`，此时运行 `python --version` 会显示什么版本？为什么？

> **提示：** pyenv 的版本优先级为 `shell > local > global`

**练习 4：** 编写一个 Python 脚本 `check_env.py`，它能检测：
1. 当前 Python 版本是否 >= 3.10
2. 是否安装了 pip
3. 是否在虚拟环境中运行
4. GPU 是否可用（尝试 `import torch`，如果失败则提示未安装 PyTorch）

> **参考答案：**
> ```python
> import sys
> import os
> 
> # 检查 Python 版本 / Check Python version
> major, minor = sys.version_info[:2]
> print(f"Python {major}.{minor}", "✅" if minor >= 10 else "❌ 建议升级到 3.10+")
> 
> # 检查虚拟环境 / Check virtual environment
> in_venv = hasattr(sys, 'real_prefix') or (
>     hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
> )
> print(f"虚拟环境: {'是 ✅' if in_venv else '否 ⚠️'}")
> 
> # 检查 GPU / Check GPU
> try:
>     import torch
>     print(f"PyTorch: {torch.__version__}, GPU: {'✅' if torch.cuda.is_available() else '❌'}")
> except ImportError:
>     print("PyTorch: 未安装 ⚠️")
> ```
