# 🚀 Python 大模型学习路线（Java 工程师版）

> **目标读者**：有扎实 Java 后端经验的工程师，零基础过渡到 Python 大模型领域。
> **设计理念**：以 Java 对比切入 Python，循序渐进从基础到大模型实战。

---

## 📋 学习路线总览

| 阶段 | 目录 | 预估周期 | 核心目标 |
|------|------|----------|----------|
| 一 | `01-python-basics/` | 2-3 周 | Python 基础语法，Java→Python 快速过渡 |
| 二 | `02-data-science-fundamentals/` | 2-3 周 | 数据科学三件套：NumPy、Pandas、Matplotlib |
| 三 | `03-machine-learning-basics/` | 3-4 周 | 数学基础 + 经典 ML 算法 + Scikit-learn |
| 四 | `04-deep-learning-basics/` | 3-4 周 | 神经网络原理 + PyTorch + CNN/RNN |
| 五 | `05-nlp-fundamentals/` | 2-3 周 | NLP 基础：文本处理、词嵌入、Seq2Seq |
| 六 | `06-llm-core-technology/` | 3-4 周 | Transformer + BERT/GPT + 训练技术 |
| 七 | `07-llm-applications/` | 3-4 周 | HuggingFace + Prompt + RAG + LoRA + LangChain |
| 八 | `08-llm-engineering/` | 2-3 周 | 量化优化 + 服务部署 + MLOps |

---

## 📁 知识点树（文件夹 ↔ 文件映射）

### 阶段一：Python 基础（Java 工程师快速过渡）
```
01-python-basics/
├── 01-environment-and-tools/
│   ├── 01-python-installation-and-version.md    # Python 安装与版本管理
│   ├── 02-virtual-environment.md                # 虚拟环境（venv/conda）
│   └── 03-ide-and-toolchain.md                  # IDE 与开发工具链
├── 02-syntax-comparison/
│   ├── 01-variables-and-types.md                # 变量与类型系统（Java vs Python）
│   ├── 02-control-flow.md                       # 控制流（if/for/while 对比）
│   ├── 03-functions-and-scope.md                # 函数与作用域
│   └── 04-modules-and-packages.md               # 模块与包管理（Maven vs pip）
├── 03-data-structures/
│   ├── 01-list-and-tuple.md                     # 列表与元组（ArrayList 对比）
│   ├── 02-dict-and-set.md                       # 字典与集合（HashMap 对比）
│   ├── 03-string-processing.md                  # 字符串处理
│   └── 04-comprehensions.md                     # 推导式（List/Dict/Set）
├── 04-oop-in-python/
│   ├── 01-class-and-object.md                   # 类与对象（Java 对比）
│   ├── 02-inheritance-and-polymorphism.md        # 继承与多态
│   ├── 03-magic-methods.md                      # 魔术方法（__init__, __str__ 等）
│   └── 04-dataclass-and-enum.md                 # dataclass 与枚举
└── 05-advanced-features/
    ├── 01-decorators.md                         # 装饰器（AOP 对比）
    ├── 02-generators-and-iterators.md           # 生成器与迭代器（Stream 对比）
    ├── 03-context-managers.md                   # 上下文管理器（try-with-resources 对比）
    ├── 04-type-hints.md                         # 类型注解（泛型对比）
    └── 05-async-programming.md                  # 异步编程（CompletableFuture 对比）
```

### 阶段二：Python 数据科学基础
```
02-data-science-fundamentals/
├── 01-numpy/
│   ├── 01-ndarray-basics.md                     # ndarray 基础与创建
│   ├── 02-indexing-and-slicing.md               # 索引与切片
│   ├── 03-broadcasting.md                       # 广播机制
│   ├── 04-linear-algebra-ops.md                 # 线性代数运算
│   └── 05-performance-optimization.md           # 性能优化（向量化 vs 循环）
├── 02-pandas/
│   ├── 01-series-and-dataframe.md               # Series 与 DataFrame
│   ├── 02-data-io.md                            # 数据读写（CSV/JSON/SQL）
│   ├── 03-data-cleaning.md                      # 数据清洗（缺失值/重复值）
│   ├── 04-groupby-and-aggregation.md            # 分组与聚合（SQL 对比）
│   └── 05-merge-and-join.md                     # 合并与连接（JOIN 对比）
└── 03-matplotlib/
    ├── 01-basic-plotting.md                     # 基础绑图（折线/柱状/散点）
    ├── 02-subplot-and-layout.md                 # 子图与布局
    └── 03-seaborn-advanced.md                   # Seaborn 高级可视化
```

### 阶段三：机器学习基础
```
03-machine-learning-basics/
├── 01-math-foundations/
│   ├── 01-linear-algebra.md                     # 线性代数（矩阵/向量/特征值）
│   ├── 02-probability-and-statistics.md         # 概率与统计（分布/贝叶斯）
│   └── 03-calculus-and-optimization.md          # 微积分与优化（梯度下降）
├── 02-classic-algorithms/
│   ├── 01-linear-regression.md                  # 线性回归
│   ├── 02-logistic-regression.md                # 逻辑回归
│   ├── 03-decision-tree-and-random-forest.md    # 决策树与随机森林
│   ├── 04-svm.md                                # 支持向量机
│   ├── 05-clustering.md                         # 聚类算法（K-Means/DBSCAN）
│   └── 06-model-evaluation.md                   # 模型评估与调优
└── 03-sklearn-practice/
    ├── 01-sklearn-pipeline.md                   # Scikit-learn Pipeline
    ├── 02-feature-engineering.md                # 特征工程
    └── 03-end-to-end-project.md                 # 端到端 ML 项目实战
```

### 阶段四：深度学习基础
```
04-deep-learning-basics/
├── 01-neural-network-theory/
│   ├── 01-perceptron-and-mlp.md                 # 感知机与多层感知机
│   ├── 02-activation-functions.md               # 激活函数（ReLU/Sigmoid/Tanh）
│   ├── 03-backpropagation.md                    # 反向传播算法
│   └── 04-optimization-algorithms.md            # 优化算法（SGD/Adam/学习率调度）
├── 02-pytorch/
│   ├── 01-tensor-basics.md                      # Tensor 基础与自动求导
│   ├── 02-nn-module.md                          # nn.Module 模型构建
│   ├── 03-training-loop.md                      # 训练循环与 DataLoader
│   ├── 04-gpu-acceleration.md                   # GPU 加速与混合精度
│   └── 05-model-save-and-load.md                # 模型保存与加载
├── 03-cnn/
│   ├── 01-convolution-theory.md                 # 卷积原理
│   ├── 02-classic-architectures.md              # 经典架构（LeNet/ResNet/VGG）
│   └── 03-image-classification-practice.md      # 图像分类实战
└── 04-rnn/
    ├── 01-rnn-and-bptt.md                       # RNN 原理与 BPTT
    ├── 02-lstm-and-gru.md                       # LSTM 与 GRU
    └── 03-sequence-prediction-practice.md       # 序列预测实战
```

### 阶段五：自然语言处理
```
05-nlp-fundamentals/
├── 01-text-preprocessing/
│   ├── 01-tokenization.md                       # 分词（中英文对比）
│   ├── 02-text-representation.md                # 文本表示（Bag-of-Words/TF-IDF）
│   └── 03-text-cleaning-pipeline.md             # 文本清洗流水线
├── 02-word-embeddings/
│   ├── 01-word2vec.md                           # Word2Vec（CBOW/Skip-gram）
│   ├── 02-glove-and-fasttext.md                 # GloVe 与 FastText
│   └── 03-contextual-embeddings.md              # 上下文嵌入（ELMo 概述）
└── 03-seq2seq-and-attention/
    ├── 01-encoder-decoder.md                    # 编码器-解码器架构
    ├── 02-attention-mechanism.md                # 注意力机制原理
    └── 03-machine-translation-practice.md       # 机器翻译实战
```

### 阶段六：大模型核心技术
```
06-llm-core-technology/
├── 01-transformer/
│   ├── 01-self-attention.md                     # 自注意力机制详解
│   ├── 02-multi-head-attention.md               # 多头注意力
│   ├── 03-positional-encoding.md                # 位置编码
│   ├── 04-transformer-architecture.md           # Transformer 完整架构
│   └── 05-transformer-from-scratch.md           # 从零实现 Transformer
├── 02-pretrained-models/
│   ├── 01-bert.md                               # BERT 详解
│   ├── 02-gpt-series.md                         # GPT 系列（GPT-1/2/3/4）
│   ├── 03-t5-and-others.md                      # T5 及其他模型
│   └── 04-scaling-laws.md                       # 缩放定律（Scaling Laws）
└── 03-training-techniques/
    ├── 01-pretraining-strategies.md             # 预训练策略（MLM/CLM/Span）
    ├── 02-distributed-training.md               # 分布式训练（DDP/FSDP/DeepSpeed）
    └── 03-rlhf.md                               # RLHF 人类反馈强化学习
```

### 阶段七：大模型应用实战
```
07-llm-applications/
├── 01-huggingface/
│   ├── 01-transformers-library.md               # Transformers 库入门
│   ├── 02-tokenizers.md                         # Tokenizers 分词器
│   ├── 03-datasets-and-evaluate.md              # Datasets 与 Evaluate 库
│   └── 04-model-hub.md                          # Model Hub 使用
├── 02-prompt-engineering/
│   ├── 01-prompt-basics.md                      # Prompt 基础与原则
│   ├── 02-few-shot-and-cot.md                   # Few-shot 与 Chain-of-Thought
│   └── 03-prompt-optimization.md                # Prompt 优化与评估
├── 03-rag/
│   ├── 01-vector-database.md                    # 向量数据库（FAISS/Milvus/Chroma）
│   ├── 02-embedding-models.md                   # Embedding 模型选择
│   ├── 03-rag-pipeline.md                       # RAG 流水线构建
│   └── 04-advanced-rag.md                       # 高级 RAG 技术
├── 04-fine-tuning/
│   ├── 01-full-fine-tuning.md                   # 全参数微调
│   ├── 02-lora-and-qlora.md                     # LoRA 与 QLoRA
│   ├── 03-peft-library.md                       # PEFT 库实战
│   └── 04-instruction-tuning.md                 # 指令微调（SFT）
└── 05-langchain/
    ├── 01-langchain-basics.md                   # LangChain 核心概念
    ├── 02-chains-and-agents.md                  # Chains 与 Agents
    ├── 03-memory-and-tools.md                   # Memory 与工具集成
    └── 04-full-application.md                   # 完整应用开发实战
```

### 阶段八：大模型部署与工程化
```
08-llm-engineering/
├── 01-model-optimization/
│   ├── 01-quantization.md                       # 模型量化（INT8/INT4/GPTQ/AWQ）
│   ├── 02-pruning-and-distillation.md           # 剪枝与知识蒸馏
│   └── 03-inference-optimization.md             # 推理优化（KV Cache/FlashAttention）
├── 02-model-serving/
│   ├── 01-vllm.md                               # vLLM 部署
│   ├── 02-triton-inference-server.md            # Triton 推理服务器
│   ├── 03-api-design.md                         # API 设计（OpenAI-compatible）
│   └── 04-scaling-and-monitoring.md             # 扩缩容与监控
└── 03-mlops/
    ├── 01-experiment-tracking.md                # 实验追踪（MLflow/W&B）
    ├── 02-data-pipeline.md                      # 数据流水线
    └── 03-cicd-for-ml.md                        # ML 项目 CI/CD
```

---

## 📖 每个知识点文件结构

每个 `.md` 文件统一采用以下结构：

```markdown
# [知识点标题]

## 1. 背景（Background）
> 为什么要学这个？它在整个体系中的位置是什么？
> 对于 Java 工程师，相当于什么概念？

## 2. 知识点（Key Concepts）
> 核心概念的定义与要点速览。

## 3. 内容（Content）
> 详细的知识内容，包含代码示例和图示说明。

## 4. 详细推理（Deep Dive）
> 原理层面的深入推导，数学公式或源码级分析。

## 5. 例题（Worked Examples）
> 完整的解题过程，附带代码实现。

## 6. 习题（Exercises）
> 由易到难的练习题，附带参考答案。
```

---

## 🎯 学习建议

1. **不要跳阶段**：即使你有 Java 经验，也要认真过一遍 Python 基础
2. **每阶段写项目**：每完成一个阶段，用一个小项目来巩固
3. **对比思维**：用 Java 知识来理解 Python 概念，事半功倍
4. **动手实践**：每个习题都要亲手敲代码运行
5. **关注工程化**：大模型不只是算法，部署和工程化同样重要
