# 01-environment-and-tools — 环境与工具

> **所属阶段**：阶段一 · Python 基础
> **学习目标**：搭建可复现、可隔离、可调试的 Python 开发环境——用 pyenv 管理多版本、用 venv/conda/uv 隔离依赖、用 VS Code + Jupyter + Ruff 构成工具链
> **预估时长**：2-3 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [python-installation-and-version](./01-python-installation-and-version.md) | Python 安装与版本管理 | 为什么只学 Python 3、2026 推荐 3.11/3.12；pyenv 多版本共存（global/local/shell 优先级）；CPython vs PyPy/Jython；GIL 对并发的影响 |
| 02 | [virtual-environment](./02-virtual-environment.md) | 虚拟环境（venv/conda/uv） | pip 默认全局安装导致的版本冲突；venv 隔离原理；conda 管 CUDA；uv 作为 2026 事实标准；requirements.txt vs pom.xml |
| 03 | [ide-and-toolchain](./03-ide-and-toolchain.md) | IDE 与开发工具链 | VS Code + Pylance 配置；Jupyter/IPython 交互式开发与魔术命令；Ruff 一体化 lint+format；pdb/breakpoint() 调试 |

---

## 🔑 知识点详解

### 01 · Python 安装与版本管理

- **核心概念**：Python 生态碎片化（系统自带 / Homebrew / conda 多版本并存），必须用版本管理工具隔离，别动系统 Python。
- **关键 API / 命令**：`pyenv install <ver>` 装版本、`pyenv global/local/shell <ver>` 三级切换，优先级 **shell > local > global**（local 生成 `.python-version` 文件）。
- **易错点**：① `which python` 显示的不是你以为的版本，多因 pyenv 的 shims 未排在 PATH 最前；② C 扩展库（NumPy/PyTorch）必须与 Python 小版本精确匹配，随意升级 Python 会连带崩掉一堆库。
- **Java 视角**：pyenv ≈ `sdkman`/`jenv`；CPython ≈ HotSpot JVM，PyPy（JIT）≈ GraalVM，Jython ≈ 跑在 JVM 上的 Python。但 Python 小版本兼容性远不如 JDK 的向后兼容。
- **前置**：无，本阶段起点。

### 02 · 虚拟环境（venv/conda/uv）

- **核心概念**：虚拟环境是 Python 的「项目级依赖隔离」，等价于给每个项目一个独立的 Maven Local Repository；不用它，项目 A 与 B 的依赖版本会在全局互相污染。
- **关键 API / 命令**：`python -m venv .venv` 创建、`source .venv/bin/activate` 激活、`pip freeze > requirements.txt` 锁版本；现代栈用 `uv init/add/sync/run`。
- **易错点**：① 忘记激活环境导致 `pip install` 装到全局、`import` 又报 ModuleNotFoundError（先 `which python` / `which pip` 排查）；② PyTorch 官方已弃用 conda 渠道，即使在 conda 环境里也要用 `pip install ... --index-url`；③ `.venv/` 属于 target 类产物，绝不提交 Git。
- **Java 视角**：`requirements.txt`/`pyproject.toml` ≈ `pom.xml`；`pip freeze`/`uv.lock` ≈ 依赖锁定；PyPI ≈ Maven Central；`uv add` ≈ 编辑 `<dependencies>` 后 `mvn install`。
- **前置**：01（需要先能确定并切换 Python 版本）。

### 03 · IDE 与开发工具链

- **核心概念**：好工具让学习效率翻倍；Jupyter Notebook 是 Python 数据科学/大模型开发的核心交互环境，**Java 世界没有对应物**（比 JShell 强得多）。
- **关键 API / 命令**：`ruff check .`（lint）+ `ruff format .`（format）一体化，配置写进 `pyproject.toml` 的 `[tool.ruff]`；调试用内置 `breakpoint()` 进 pdb；IPython 魔术命令 `%timeit`（基准）、`%run`、`!ls`（跑 shell）。
- **易错点**：① VS Code 用错解释器（未指向 `.venv/bin/python`）导致补全和运行环境不一致；② 魔术命令 `%timeit` 必须独占一行，不能混进普通表达式。
- **Java 视角**：PyCharm/VS Code ≈ IntelliJ IDEA；Ruff ≈ Google Java Format + SpotBugs/PMD 合体；mypy ≈ 编译期类型检查；`%timeit` ≈ JMH 基准测试；pdb ≈ IDEA Debugger 的命令行版。
- **前置**：02（IDE 需指向虚拟环境的解释器）。

---

## 🎯 学习要点

- **只学一套主流组合，别在选型上纠结**：入门用 `pyenv + venv + VS Code`，做数据科学加 `conda`（省心处理 CUDA），工程化再上 `uv`（Rust 实现，比 pip 快 10-100×）。
- **亲手验证环境正确性**：跑一段脚本打印 `sys.version` / `sys.executable`，确认 `python` 和 `pip` 指向同一个 `.venv`，避免「装了却 import 不到」的经典坑。
- **把 `.gitignore` 当第一件事配好**：至少排除 `.venv/`、`__pycache__/`、`*.py[cod]`、`.ipynb_checkpoints/`，对标 Java 的 `target/` 不入库。
- **理解 GIL 但别过度担心**：同一时刻只有一个线程执行 Python 字节码，CPU 密集用 multiprocessing、I/O 密集用 asyncio；大模型的重计算由 PyTorch 的 C++ 后端处理并释放 GIL，GIL 不是瓶颈。
- **让 Ruff + mypy 替你把关**：开启保存时自动格式化和 import 排序，把 Java 里 IDEA 帮你做的事在 Python 里显式配起来，从第一天就写规范代码。
- **用 Jupyter 培养「交互式探索」习惯**：逐 Cell 执行、即时看图表和中间结果，这是后续观察 loss 曲线、调试张量形状的核心工作方式。

---

## 🔗 关联

- **下一模块**：[02-syntax-comparison](../02-syntax-comparison/) — 环境就绪后开始写第一行 Python，逐点对比 Java 语法。
- **本阶段总览**：[阶段一 README](../README.md)
- **配套实战**：[agent-course/Day-01](../../agent-course/Day-01-first-call.md) — 第一次调用大模型 API，需要本模块搭好的环境与依赖管理。
