# 🧠 Python 大模型学习路线

> **面向 Java 后端工程师的 Python + LLM 系统学习路线**
>
> 从 Python 基础到大模型工程化部署，8 个阶段、103 个知识点，每个知识点均以 Java 对比视角编写。

---

## 📋 学习路线总览

| 阶段 | 主题 | 知识点数 | 关键内容 |
|:----:|------|:-------:|---------|
| 一 | [Python 基础](./01-python-basics/) | 20 | 环境管理、语法对比、数据结构、OOP、高级特性 |
| 二 | [数据科学基础](./02-data-science-fundamentals/) | 13 | NumPy、Pandas、Matplotlib/Seaborn |
| 三 | [机器学习基础](./03-machine-learning-basics/) | 12 | 数学基础、经典算法、Sklearn 实战 |
| 四 | [深度学习基础](./04-deep-learning-basics/) | 15 | 神经网络理论、PyTorch、CNN、RNN |
| 五 | [NLP 基础](./05-nlp-fundamentals/) | 9 | 文本预处理、词嵌入、Seq2Seq/Attention |
| 六 | [大模型核心技术](./06-llm-core-technology/) | 12 | Transformer、预训练模型、训练技术 |
| 七 | [大模型应用实战](./07-llm-applications/) | 13 | HuggingFace、Prompt Engineering、RAG、微调、LangChain |
| 八 | [工程化与部署](./08-llm-engineering/) | 9 | 量化加速、模型服务、MLOps |

> 📄 完整知识点树请参考 [OUTLINE.md](./OUTLINE.md)

---

## 🎯 适合人群

- ☕ **Java 后端工程师**想转型 AI / 大模型方向
- 🐍 有编程基础，想系统学习 **Python + 深度学习 + LLM**
- 🏗️ 需要一份**结构化、可执行**的学习路线图

## 📝 文档结构

每个知识点文件统一采用以下格式：

```
## 1. 背景（Background）     — 为什么要学这个
## 2. 知识点（Key Concepts）  — 核心概念速览
## 3. 内容（Content）         — 代码示例 + Java 对比
## 4. 详细推理（Deep Dive）   — 底层原理分析
## 5. 例题（Examples）        — 典型应用
## 6. 习题（Exercises）       — 动手练习
```

## 🛠️ 推荐环境

| 工具 | 推荐 | 说明 |
|------|------|------|
| Python 版本 | 3.11+ | `pyenv` 管理多版本 |
| 包管理 | `uv` | 比 pip 快 10-100x |
| IDE | VS Code | 配合 Python + Jupyter 插件 |
| 交互式实验 | JupyterLab | 适合数据探索和模型实验 |
| 代码规范 | Ruff | 替代 flake8 + black + isort |

## 📂 目录结构

```
learn-ai/
├── OUTLINE.md                          # 知识点总大纲
├── 01-python-basics/                   # 阶段一：Python 基础
│   ├── 01-environment-and-tools/       #   环境与工具链
│   ├── 02-syntax-comparison/           #   语法对比（Java vs Python）
│   ├── 03-data-structures/             #   数据结构
│   ├── 04-oop-in-python/               #   面向对象编程
│   └── 05-advanced-features/           #   高级特性
├── 02-data-science-fundamentals/       # 阶段二：数据科学
│   ├── 01-numpy/
│   ├── 02-pandas/
│   └── 03-matplotlib/
├── 03-machine-learning-basics/         # 阶段三：机器学习
│   ├── 01-math-foundations/
│   ├── 02-classic-algorithms/
│   └── 03-sklearn-practice/
├── 04-deep-learning-basics/            # 阶段四：深度学习
│   ├── 01-neural-network-theory/
│   ├── 02-pytorch/
│   ├── 03-cnn/
│   └── 04-rnn/
├── 05-nlp-fundamentals/                # 阶段五：NLP 基础
│   ├── 01-text-preprocessing/
│   ├── 02-word-embeddings/
│   └── 03-seq2seq-and-attention/
├── 06-llm-core-technology/             # 阶段六：大模型核心
│   ├── 01-transformer/
│   ├── 02-pretrained-models/
│   └── 03-training-techniques/
├── 07-llm-applications/                # 阶段七：大模型应用
│   ├── 01-huggingface/
│   ├── 02-prompt-engineering/
│   ├── 03-rag/
│   ├── 04-fine-tuning/
│   └── 05-langchain/
└── 08-llm-engineering/                 # 阶段八：工程化部署
    ├── 01-model-optimization/
    ├── 02-model-serving/
    └── 03-mlops/
```

## 📜 License

本项目仅供个人学习使用。
