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

---

## 🗺️ 学习路线图

```
阶段一 Python 基础        ──→  阶段二 数据科学        ──→  阶段三 机器学习
  • 环境与工具链                 • NumPy ndarray            • 线性代数/概率/微积分
  • Java vs Python 语法          • Pandas DataFrame          • 线性/逻辑回归
  • 数据结构与 OOP               • Matplotlib 可视化         • 决策树/SVM/聚类
  • 装饰器/生成器/异步           • Seaborn 统计图            • Sklearn Pipeline
         │
         ▼
阶段四 深度学习基础       ──→  阶段五 NLP 基础        ──→  阶段六 大模型核心
  • 感知机/MLP/反向传播          • 分词/文本表示              • Self-Attention 数学推导
  • PyTorch Tensor/nn.Module     • Word2Vec/GloVe            • Multi-Head Attention
  • CNN 经典架构                 • Seq2Seq + Attention       • Transformer 从零实现
  • LSTM/GRU                     • 机器翻译实战              • BERT/GPT/T5/Scaling Laws
         │
         ▼
阶段七 大模型应用实战     ──→  阶段八 工程化与部署
  • HuggingFace 生态             • 量化 (GPTQ/GGUF/AWQ)
  • Prompt Engineering           • 推理加速 (vLLM/TGI)
  • RAG + 向量数据库             • API 服务 (FastAPI)
  • LoRA/QLoRA 微调              • Docker 容器化部署
  • LangChain Agent              • MLOps (W&B/CI-CD)
```

---

## 📝 文档结构

每个知识点文件统一采用 **6 段式模板**：

```
## 1. 背景（Background）     — 为什么要学 + Java 对比
## 2. 知识点（Key Concepts）  — 核心概念速览表
## 3. 内容（Content）         — 完整代码示例 + 注释
## 4. 详细推理（Deep Dive）   — 底层原理 + 数学推导
## 5. 例题（Worked Examples） — 实战案例
## 6. 习题（Exercises）       — 分层练习（基础/进阶）
```

### 特色

- 🔄 **Java 对比视角** — 每个概念都映射到 Java 工程师熟悉的模式（如 Agent ≈ Controller，Pipeline ≈ 责任链）
- 📐 **代码注释双语** — 中英文双语注释，含算法复杂度和 Shape 标注
- 🎯 **工程导向** — 不止理论，覆盖生产环境所需的优化、部署和监控实践
- 📊 **渐进式难度** — 从 `np.array` 到 `torch.einsum`，从线性回归到 LoRA 微调

---

## 🛠️ 推荐环境

| 工具 | 推荐 | 说明 |
|------|------|------|
| Python 版本 | 3.11+ | `pyenv` 管理多版本 |
| 包管理 | `uv` | 比 pip 快 10-100x |
| IDE | VS Code | 配合 Python + Jupyter 插件 |
| 交互式实验 | JupyterLab | 适合数据探索和模型实验 |
| 代码规范 | Ruff | 替代 flake8 + black + isort |
| GPU 环境 | CUDA 12.1+ | 深度学习阶段需要 |

### 核心依赖

```bash
# 数据科学
pip install numpy pandas matplotlib seaborn scikit-learn

# 深度学习
pip install torch torchvision

# 大模型
pip install transformers datasets peft trl accelerate

# RAG & Agent
pip install langchain langchain-openai chromadb faiss-cpu

# 模型服务
pip install vllm fastapi uvicorn

# 实验管理
pip install wandb mlflow
```

---

## 📂 目录结构

```
learn-ai/
├── OUTLINE.md                          # 知识点总大纲
├── 01-python-basics/                   # 阶段一：Python 基础
│   ├── 01-environment-and-tools/       #   环境与工具链
│   ├── 02-syntax-comparison/           #   语法对比（Java vs Python）
│   ├── 03-data-structures/             #   数据结构
│   ├── 04-oop-in-python/               #   面向对象编程
│   └── 05-advanced-features/           #   高级特性（装饰器/生成器/异步）
├── 02-data-science-fundamentals/       # 阶段二：数据科学
│   ├── 01-numpy/                       #   ndarray/索引/广播/线性代数/性能
│   ├── 02-pandas/                      #   DataFrame/IO/清洗/分组/合并
│   └── 03-matplotlib/                  #   基础绑图/子图/Seaborn
├── 03-machine-learning-basics/         # 阶段三：机器学习
│   ├── 01-math-foundations/            #   线性代数/概率/微积分
│   ├── 02-classic-algorithms/          #   回归/树/SVM/聚类/评估
│   └── 03-sklearn-practice/            #   Pipeline/特征工程/端到端
├── 04-deep-learning-basics/            # 阶段四：深度学习
│   ├── 01-neural-network-theory/       #   感知机/激活函数/反向传播/优化器
│   ├── 02-pytorch/                     #   Tensor/nn.Module/训练循环/GPU
│   ├── 03-cnn/                         #   卷积原理/经典架构/图像分类
│   └── 04-rnn/                         #   LSTM-GRU/序列预测
├── 05-nlp-fundamentals/                # 阶段五：NLP 基础
│   ├── 01-text-preprocessing/          #   分词/文本表示/清洗
│   ├── 02-word-embeddings/             #   Word2Vec/GloVe/上下文嵌入
│   └── 03-seq2seq-and-attention/       #   编解码器/注意力/机器翻译
├── 06-llm-core-technology/             # 阶段六：大模型核心
│   ├── 01-transformer/                 #   Self-Attention/MHA/位置编码/架构/实现
│   ├── 02-pretrained-models/           #   BERT/GPT/T5/Scaling Laws
│   └── 03-training-techniques/         #   分布式/混合精度/RLHF/DPO
├── 07-llm-applications/                # 阶段七：大模型应用
│   ├── 01-huggingface/                 #   Transformers/Datasets/Trainer
│   ├── 02-prompt-engineering/          #   基础/CoT/ReAct/Self-Consistency
│   ├── 03-rag/                         #   RAG原理/向量数据库/生产实践
│   ├── 04-fine-tuning/                 #   LoRA-QLoRA/指令微调
│   └── 05-langchain/                   #   LCEL/Agent/完整应用
└── 08-llm-engineering/                 # 阶段八：工程化部署
    ├── 01-model-optimization/          #   量化/推理加速/知识蒸馏
    ├── 02-model-serving/               #   vLLM-TGI/FastAPI/Docker
    └── 03-mlops/                       #   实验管理/评估监控/CI-CD
```

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/qian-lou/learn-ai.git
cd learn-ai

# 建议从阶段一开始，按顺序学习
# 如果你已有 Python 基础，可以直接跳到阶段四（深度学习）
```

---

## 📜 License

本项目仅供个人学习使用。
