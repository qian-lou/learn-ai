# 🧠 Python 大模型学习路线

> **面向 Java 后端工程师的 Python + LLM 系统学习路线**
>
> 从 Python 基础到大模型工程化部署，8 个阶段、107 个知识点，每个知识点均以 Java 对比视角编写。

---

## 📋 学习路线总览

| 阶段 | 主题 | 知识点数 | 关键内容 |
|:----:|------|:-------:|---------|
| 一 | [Python 基础](./01-python-basics/) | 20 | 环境管理、语法对比、数据结构、OOP、高级特性 |
| 二 | [数据科学基础](./02-data-science-fundamentals/) | 14 | NumPy、Pandas、Matplotlib/Seaborn |
| 三 | [机器学习基础](./03-machine-learning-basics/) | 13 | 数学基础、经典算法、Sklearn 实战 |
| 四 | [深度学习基础](./04-deep-learning-basics/) | 15 | 神经网络理论、PyTorch、CNN、RNN |
| 五 | [NLP 基础](./05-nlp-fundamentals/) | 9 | 文本预处理、词嵌入、Seq2Seq/Attention |
| 六 | [大模型核心技术](./06-llm-core-technology/) | 13 | Transformer、预训练模型、训练技术 |
| 七 | [大模型应用实战](./07-llm-applications/) | 14 | HuggingFace、Prompt Engineering、RAG、微调、LangChain |
| 八 | [工程化与部署](./08-llm-engineering/) | 9 | 量化加速、模型服务、MLOps |

> 📄 完整知识点树请参考 [OUTLINE.md](./OUTLINE.md)
>
> 🔧 关键库锁定版本与迁移点见 [VERSIONS.md](./VERSIONS.md)——「2026 现代写法」跑不通时先对版本。

---

## 🧠 知识点地图

> 下面把 8 阶段 107 个知识点压缩成「每阶段最该掌握的核心簇」。
> 每条只讲**一件事**：这个点你真正要拿走的是什么。带 ⭐ 的是四个**能力跃迁里程碑**——过了它，你就换了一个段位。

### 阶段一 · Python 基础 · 20 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| 环境与工具链 | 直接上 `uv` + `pyenv`，别再用 pip 装环境；2026 年 `uv` 已是事实标准 |
| Java→Python 语法映射 | 动态类型、缩进即作用域、一切皆对象——把「编译期报错」的安全感换成「运行时 + 类型注解」 |
| 数据结构 | `list/dict/set` 对标 `ArrayList/HashMap/HashSet`，但推导式让你一行写完一个 for 循环 |
| OOP 与 dataclass | 没有 `private`，用约定（`_name`）代替；`@dataclass` ≈ Lombok 的 `@Data` |
| 装饰器/生成器/上下文管理器 | 装饰器 ≈ AOP，生成器 ≈ 惰性 Stream，`with` ≈ try-with-resources |
| 类型注解 + 异步 | `type hints` 是给人和 IDE 看的（运行时不强制）；`async/await` ≈ 单线程事件循环版的 `CompletableFuture` |

### 阶段二 · 数据科学基础 · 14 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| ndarray 与向量化 | 一切性能的起点：用数组运算替代 Python for 循环，快 10-100 倍 |
| 广播机制 | 不同形状的数组如何自动对齐——理解它才能读懂后面所有张量代码 |
| Pandas DataFrame | 内存里的「SQL 表 + Excel」，`groupby` ≈ `GROUP BY`，`merge` ≈ `JOIN` |
| 数据清洗 | 真实数据 80% 时间花在缺失值/重复值/类型转换上 |
| 可视化 | Matplotlib 画一切，Seaborn 一行出统计图——探索数据的眼睛 |

### 阶段三 · 机器学习基础 · 13 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| 数学三件套 | 线性代数（矩阵运算）+ 概率（分布/贝叶斯）+ 微积分（梯度）——深度学习的地基，够用即可 |
| 梯度下降 | 所有模型训练的通用引擎：沿梯度反方向一步步逼近最优 |
| 线性/逻辑回归 | 最简单的可解释模型，逻辑回归其实是「单层神经网络」 |
| 树模型与集成 | 决策树/随机森林/GBDT——表格数据上至今仍打得过深度学习 |
| 聚类与评估 | 无监督的 K-Means/DBSCAN；混淆矩阵/AUC/交叉验证是你的「单元测试」 |
| Sklearn Pipeline | 把预处理+模型串成一条链（≈ 责任链模式），杜绝数据泄漏 |

### 阶段四 · 深度学习基础 · 15 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| 感知机→MLP | 神经网络 = 线性层 + 非线性激活的堆叠 |
| 反向传播 | 链式法则自动求梯度——理解它，才不会把训练当黑盒 |
| 优化器与调度 | 从 SGD 到 Adam/AdamW；学习率是最重要的超参数 |
| PyTorch 核心 | `Tensor`（带 autograd 的 ndarray）+ `nn.Module`（≈ 可组合的 Bean）+ 训练循环 |
| GPU 与混合精度 | `.to(device)` + AMP，是从「跑得动」到「跑得快」的分水岭 |
| CNN | 卷积=局部权重共享；ResNet 的残差连接让网络能堆到几百层 |
| RNN/LSTM/GRU | 处理序列的经典方案，也是理解「为什么需要 Attention」的反面教材 |

### 阶段五 · NLP 基础 · 9 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| 分词与文本表示 | 从 Bag-of-Words/TF-IDF 到子词分词——把文字变成模型能吃的数字 |
| 词嵌入 | Word2Vec/GloVe：用向量距离表达语义，「国王-男+女≈女王」 |
| 上下文嵌入 | 同一个词在不同句子里向量不同——通往预训练模型的思想跳板 |
| Seq2Seq | 编码器-解码器结构，机器翻译的经典范式 |
| ⭐ 注意力机制 | **里程碑一**：搞懂 Attention 就搞懂了 LLM 的心脏——让模型自己决定「该看哪里」 |

### 阶段六 · 大模型核心技术 · 13 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| Self-Attention 数学推导 | `softmax(QKᵀ/√d_k)·V`——把这个公式的每个符号讲清楚，你就入门了 |
| 多头注意力 | 并行开多个「视角」看同一句话，再拼接 |
| 位置编码 | Attention 本身无序，靠位置编码注入顺序信息（正弦式 → 现代 RoPE） |
| ⭐ 从零实现 Transformer | **里程碑二**：亲手拼出 Encoder/Decoder，从此不再有黑盒 |
| 预训练模型谱系 | BERT（双向理解）/ GPT（单向生成）/ T5（统一 text-to-text）三条路线 |
| Scaling Laws | 参数量、数据量、算力与效果的幂律关系——大模型「大」的理论依据 |
| 训练技术 | 预训练策略（MLM/CLM）+ 分布式（DDP/FSDP/DeepSpeed）+ 对齐（RLHF/DPO） |

### 阶段七 · 大模型应用实战 · 14 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| HuggingFace 生态 | `transformers`+`datasets`+`Trainer`：加载/微调开源模型的标准工作台 |
| Prompt Engineering | CoT（思维链）/ReAct（推理+行动）/Self-Consistency——不改权重就提升效果 |
| RAG | 检索增强生成：给模型外挂知识库，解决「幻觉」和「知识过时」 |
| 向量数据库 | Chroma/FAISS/pgvector——语义检索的存储底座 |
| ⭐ LoRA/QLoRA 微调 | **里程碑三**：只训练 <1% 参数就能定制大模型，单卡消费级 GPU 可跑 |
| LangChain / Agent | LCEL 编排 + Agent（≈ 会调用工具的 Controller）搭出完整应用 |

### 阶段八 · 工程化与部署 · 9 点

| 核心簇 | 一句话抓住重点 |
|--------|----------------|
| 量化 | GPTQ/AWQ/GGUF——把模型压到 4-bit，显存砍半、精度几乎不掉 |
| 推理加速 | KV Cache + FlashAttention + PagedAttention 是吞吐量的三大杠杆 |
| 知识蒸馏 | 用大模型教小模型，换取更低的部署成本 |
| ⭐ vLLM 服务部署 | **里程碑四**：用 vLLM/TGI 起高并发推理服务，FastAPI 包成 OpenAI 兼容接口，Docker 交付 |
| MLOps | 实验追踪（W&B/MLflow）+ 评估监控 + CI/CD——让模型可复现、可观测、可迭代 |

### 🧗 四个能力跃迁里程碑

```
理解 self-attention  ──▶  能从零写 Transformer  ──▶  能做 LoRA 微调  ──▶  能部署 vLLM 服务
   (阶段五收尾)              (阶段六中段)              (阶段七核心)          (阶段八落地)
   "看懂论文"                "读懂源码"                "定制模型"            "交付生产"
```

> 判断自己学到哪：能对着公式讲清 Attention → 里程碑一；能不看参考实现敲出 Transformer → 里程碑二；能跑通一次 QLoRA 并解释每个超参 → 里程碑三；能把一个模型做成扛得住并发的线上服务 → 里程碑四。

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

- 🔄 **Java 对比视角** — 多数概念映射到 Java 工程师熟悉的模式（如 Agent ≈ Controller，Pipeline ≈ 责任链）；基础阶段对比最充分，进阶主题侧重底层原理
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
uv add numpy pandas matplotlib seaborn scikit-learn

# 深度学习
uv add torch torchvision

# 大模型
uv add transformers datasets peft trl accelerate

# RAG & Agent
uv add langchain langchain-openai chromadb faiss-cpu

# 模型服务
uv add vllm fastapi uvicorn

# 实验管理
uv add wandb mlflow
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
│   └── 03-matplotlib/                  #   基础绘图/子图/Seaborn
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
│   └── 03-training-techniques/         #   预训练策略/分布式训练/RLHF+DPO
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

> 🤖 **想直接做 Agent？** 本仓库另有一条平行的 [`agent-course/`](./agent-course/)——70 天每日课程，专攻工具调用 / RAG / 多智能体 / MCP-A2A / 可观测性与生产部署。本主课程讲**大模型底层原理**（怎么造与理解模型），agent-course 讲**Agent 工程实战**（怎么用模型交付产品），两条线可并行，遇到底层概念回查本主课程即可。

---

## 📜 License

本项目仅供个人学习使用。
