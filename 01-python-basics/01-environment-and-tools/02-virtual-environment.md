# 虚拟环境
# Virtual Environment

## 1. 背景（Background）

> **为什么要学这个？**
>
> 在 Java 中，Maven/Gradle 通过 `pom.xml` / `build.gradle` 为每个项目管理独立的依赖，依赖下载到 `~/.m2/repository` 或 `~/.gradle/caches` 中共享但版本隔离。Python 的 pip 默认将包安装到**全局**——如果项目 A 需要 `numpy==1.24` 而项目 B 需要 `numpy==1.26`，全局安装就会冲突。
>
> **虚拟环境就是 Python 的"项目隔离"方案**，等同于给每个项目一个独立的 "Maven Local Repository"。
>
> **在整个体系中的位置：** 虚拟环境是 Python 项目管理的基石。在大模型开发中，不同项目可能依赖不同版本的 PyTorch、Transformers 等库，必须使用虚拟环境隔离。

## 2. 知识点（Key Concepts）

| 概念 | Java 对应 | Python 对应 | 说明 |
|------|-----------|-------------|------|
| 依赖隔离 | Maven 项目级 pom.xml | 虚拟环境 (venv/conda) | 每个项目独立依赖 |
| 依赖声明 | pom.xml / build.gradle | requirements.txt / pyproject.toml | 依赖清单 |
| 依赖锁定 | pom.xml 精确版本 | pip freeze / poetry.lock | 锁定精确版本 |
| 依赖仓库 | Maven Central | PyPI (pip) / conda-forge | 包仓库 |

**核心工具对比：**

| 工具 | 定位 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|--------|
| venv | 内置虚拟环境 | 无需额外安装 | 功能简单 | ⭐⭐⭐ |
| conda | 包 + 环境管理 | 可管理非 Python 依赖（CUDA） | 与 pip 混用易冲突 | ⭐⭐⭐⭐ |
| poetry | 现代项目管理 | 类似 Maven，功能完整 | 学习曲线 | ⭐⭐⭐⭐ |
| uv | 新一代包管理 | Rust 实现，极快 | 较新，生态未完全覆盖 | ⭐⭐⭐⭐⭐ |

## 3. 内容（Content）

### 3.1 venv（内置虚拟环境）

```bash
# 创建虚拟环境（类似 mvn archetype:generate 创建项目）
# Create virtual environment
python -m venv .venv

# 激活虚拟环境
# Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 激活后，命令行前缀会变化
# After activation, shell prompt changes
# (.venv) $ python --version

# 安装依赖
# Install dependencies
pip install numpy pandas torch

# 导出依赖清单（类似 mvn dependency:tree > deps.txt）
# Export dependency list
pip freeze > requirements.txt

# 从依赖清单安装（类似 mvn install）
# Install from dependency list
pip install -r requirements.txt

# 退出虚拟环境
# Deactivate virtual environment
deactivate
```

**venv 目录结构（类似 Java 项目的 target 目录，不要提交到 Git）：**

```
.venv/
├── bin/                 # 可执行文件（python, pip 等）
│   ├── python → python3.11
│   ├── pip
│   └── activate         # 激活脚本
├── lib/
│   └── python3.11/
│       └── site-packages/  # 安装的第三方包（类似 node_modules）
│           ├── numpy/
│           ├── pandas/
│           └── torch/
└── pyvenv.cfg           # 虚拟环境配置
```

### 3.2 conda 环境管理

```bash
# 安装 Miniconda（推荐，比 Anaconda 轻量）
# Install Miniconda (recommended, lighter than Anaconda)
brew install miniconda

# 创建环境并指定 Python 版本
# Create environment with specific Python version
conda create -n llm-dev python=3.11

# 激活环境
# Activate environment
conda activate llm-dev

# 安装包（conda 可以安装 CUDA 等非 Python 依赖）
# Install packages (conda can install non-Python dependencies like CUDA)
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy pandas matplotlib

# 导出环境
# Export environment
conda env export > environment.yml

# 从文件创建环境
# Create environment from file
conda env create -f environment.yml

# 查看所有环境
# List all environments
conda env list

# 删除环境
# Remove environment
conda env remove -n llm-dev

# 退出环境
# Deactivate environment
conda deactivate
```

### 3.3 requirements.txt 详解

```txt
# requirements.txt 示例
# requirements.txt example

# 精确版本（推荐用于生产）
# Exact version (recommended for production)
numpy==1.26.2
pandas==2.1.4

# 兼容版本范围
# Compatible version range
torch>=2.1.0,<2.3.0
transformers~=4.36.0  # 等价于 >=4.36.0,<4.37.0

# 从 Git 仓库安装
# Install from Git repository
# git+https://github.com/huggingface/transformers.git@main

# 开发依赖（可以拆分为 requirements-dev.txt）
# Dev dependencies (can be split into requirements-dev.txt)
pytest==7.4.3
black==23.12.1
```

**Java 工程师理解 requirements.txt：**

```xml
<!-- 这个 Maven 依赖声明... -->
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>32.1.3-jre</version>
</dependency>

<!-- 等价于 requirements.txt 中的 -->
<!-- guava==32.1.3 -->
```

### 3.4 现代工具 uv（推荐）

```bash
# 安装 uv（Rust 编写，速度极快）
# Install uv (written in Rust, extremely fast)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 初始化项目（类似 mvn archetype:generate）
# Initialize project
uv init my-llm-project
cd my-llm-project

# 添加依赖（类似 mvn add dependency）
# Add dependency
uv add numpy pandas torch

# 运行项目
# Run project
uv run python main.py

# 同步依赖（类似 mvn install）
# Sync dependencies
uv sync

# uv 速度对比 pip：安装同样的依赖
# uv speed comparison with pip:
# pip install: ~45 秒
# uv pip install: ~3 秒（15x 提速）
```

## 4. 详细推理（Deep Dive）

### 4.1 虚拟环境的工作原理

```
┌──── 系统 Python (/usr/bin/python3) ────┐
│                                          │
│  site-packages/ (全局安装的包)            │
│  ├── numpy 1.24.0                        │
│  ├── requests 2.31.0                     │
│  └── ...                                 │
│                                          │
│  创建 venv 时发生的事情：                  │
│  1. 复制 python 可执行文件（或创建符号链接）│
│  2. 创建独立的 site-packages 目录          │
│  3. 生成 activate 脚本修改 PATH            │
│                                          │
└──────────────────────────────────────────┘

┌──── 虚拟环境 (.venv/) ─────────────────┐
│                                          │
│  bin/python → 链接到系统 Python 解释器     │
│  lib/site-packages/ (隔离的包)            │
│  ├── numpy 1.26.2  ← 独立版本！          │
│  ├── torch 2.1.0                         │
│  └── ...                                 │
│                                          │
│  activate 脚本做了什么？                   │
│  1. 将 .venv/bin 加到 PATH 最前面         │
│  2. 设置 VIRTUAL_ENV 环境变量             │
│  3. 修改 shell 提示符                     │
│                                          │
└──────────────────────────────────────────┘
```

### 4.2 venv vs conda：何时选哪个？

```
选择决策树：
                    ┌─ 需要管理 CUDA 版本？ ─→ conda ✅
                    │
你的项目需求 ──────┼─ 纯 Python 项目？ ─→ venv + pip 或 uv ✅
                    │
                    ├─ 需要项目管理（构建/发布）？ ─→ poetry 或 uv ✅
                    │
                    └─ 需要复现科研环境？ ─→ conda + environment.yml ✅
```

**大模型开发推荐组合：**
- **入门阶段**：`conda`（自动处理 CUDA 依赖，最省心）
- **工程化阶段**：`uv`（速度快，与 pip 兼容，适合 CI/CD）

### 4.3 .gitignore 配置

```gitignore
# Python 虚拟环境（类似 Java 的 target/）
# Python virtual environment (like Java's target/)
.venv/
venv/
env/

# Python 编译缓存（类似 Java 的 .class 文件）
# Python compiled cache (like Java's .class files)
__pycache__/
*.py[cod]

# IDE
.idea/
.vscode/

# 环境变量
.env

# Jupyter
.ipynb_checkpoints/
```

## 5. 例题（Worked Examples）

### 例题 1：完整的项目环境搭建

**问题：** 创建一个大模型学习项目，要求使用 Python 3.11 + venv，安装 PyTorch 和 Transformers。

**解答：**

```bash
# 步骤 1：确保使用正确的 Python 版本
# Step 1: Ensure correct Python version
pyenv local 3.11.7
python --version  # Python 3.11.7

# 步骤 2：创建项目目录
# Step 2: Create project directory
mkdir llm-learning && cd llm-learning

# 步骤 3：创建虚拟环境
# Step 3: Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 步骤 4：升级 pip
# Step 4: Upgrade pip
pip install --upgrade pip

# 步骤 5：安装核心依赖
# Step 5: Install core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install numpy pandas matplotlib
pip install jupyter

# 步骤 6：导出依赖
# Step 6: Export dependencies
pip freeze > requirements.txt

# 步骤 7：验证安装
# Step 7: Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 步骤 8：初始化 Git
# Step 8: Initialize Git
git init
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

### 例题 2：conda 环境配置 CUDA

**问题：** 使用 conda 创建一个支持 GPU 的 PyTorch 环境。

**解答：**

```bash
# 创建 conda 环境
conda create -n torch-gpu python=3.11 -y
conda activate torch-gpu

# 安装支持 CUDA 的 PyTorch（conda 会自动安装对应的 CUDA Toolkit）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证 GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 使用 venv 创建一个名为 `.venv` 的虚拟环境，安装 `requests` 和 `flask`，然后导出 `requirements.txt`。对比 `requirements.txt` 和 Maven 的 `pom.xml`，列举 3 个异同点。

**练习 2：** 解释为什么 `.venv/` 不应该提交到 Git 仓库。Java 中有哪些类似的不应提交的目录？

### 进阶题

**练习 3：** 你发现运行 `pip install numpy` 后，Python 脚本中 `import numpy` 仍然报 `ModuleNotFoundError`。列举至少 3 种可能的原因，并给出排查命令。

> **参考答案：**
> 1. 没有激活虚拟环境 → `which python` 检查
> 2. pip 和 python 指向不同环境 → `which pip` vs `which python`
> 3. 安装到了不同版本 → `python -m pip install numpy` 确保一致
> 4. IDE 使用了不同的解释器 → 检查 IDE 的 Python Interpreter 设置

**练习 4：** 编写一个 shell 脚本 `setup.sh`，实现以下功能：
1. 检查 Python 版本是否 >= 3.10
2. 创建 venv 虚拟环境
3. 激活虚拟环境
4. 安装 requirements.txt 中的依赖
5. 运行一个验证脚本

> **参考答案：**
> ```bash
> #!/bin/bash
> set -e
> 
> # 检查 Python 版本 / Check Python version
> PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
> MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
> if [ "$MINOR" -lt 10 ]; then
>     echo "❌ 需要 Python 3.10+，当前为 $PYTHON_VERSION"
>     exit 1
> fi
> echo "✅ Python $PYTHON_VERSION"
> 
> # 创建并激活虚拟环境 / Create and activate venv
> python3 -m venv .venv
> source .venv/bin/activate
> echo "✅ 虚拟环境已激活"
> 
> # 安装依赖 / Install dependencies
> pip install --upgrade pip
> if [ -f "requirements.txt" ]; then
>     pip install -r requirements.txt
>     echo "✅ 依赖安装完成"
> fi
> 
> echo "🎉 环境配置完成！"
> ```
