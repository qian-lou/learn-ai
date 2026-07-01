# 阶段三：机器学习基础

> **预估周期**：3-4 周
> **核心目标**：数学基础 + 经典 ML 算法 + Scikit-learn 实战
> **学完能做到**：手推梯度下降与交叉熵、理解主流算法的偏差-方差取舍、用 Pipeline + 交叉验证跑通一个端到端项目并选对评估指标

---

## 📋 模块大纲

### [01-math-foundations](./01-math-foundations/) — 数学基础

机器学习的数学三大支柱：线性代数（矩阵/向量运算，神经网络的语言）、概率统计（损失函数与贝叶斯的根基）、微积分与优化（梯度下降是一切训练的引擎）。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [linear-algebra](./01-math-foundations/01-linear-algebra.md) | 线性代数（矩阵/向量/特征值） |
| 02 | [probability-and-statistics](./01-math-foundations/02-probability-and-statistics.md) | 概率与统计（分布/贝叶斯） |
| 03 | [calculus-and-optimization](./01-math-foundations/03-calculus-and-optimization.md) | 微积分与优化（梯度下降） |

---

### [02-classic-algorithms](./02-classic-algorithms/) — 经典算法

从最简单的线性回归到无监督聚类，覆盖监督/无监督两大范式，并配一套完整的评估-调优方法论。这些算法是理解深度学习动机的阶梯，也是结构化数据场景的主力。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [linear-regression](./02-classic-algorithms/01-linear-regression.md) | 线性回归 |
| 02 | [logistic-regression](./02-classic-algorithms/02-logistic-regression.md) | 逻辑回归 |
| 03 | [decision-tree-and-random-forest](./02-classic-algorithms/03-decision-tree-and-random-forest.md) | 决策树与随机森林 |
| 04 | [svm](./02-classic-algorithms/04-svm.md) | 支持向量机 |
| 05 | [clustering](./02-classic-algorithms/05-clustering.md) | 聚类算法（K-Means/DBSCAN） |
| 06 | [model-evaluation](./02-classic-algorithms/06-model-evaluation.md) | 模型评估与调优 |

---

### [03-sklearn-practice](./03-sklearn-practice/) — Scikit-learn 实战

把前两个模块的理论落到工程：用 Pipeline 封装预处理防止数据泄露，用特征工程逼近数据上限，最后串成一个从 EDA 到落盘部署的完整项目。

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [sklearn-pipeline](./03-sklearn-practice/01-sklearn-pipeline.md) | Scikit-learn Pipeline |
| 02 | [feature-engineering](./03-sklearn-practice/02-feature-engineering.md) | 特征工程 |
| 03 | [end-to-end-project](./03-sklearn-practice/03-end-to-end-project.md) | 端到端 ML 项目实战 |

---

## 🎯 阶段学习要点

- **数学不是背公式，是建立直觉**：能一句话说清「梯度下降为什么朝负梯度走」「交叉熵为什么等价于最大似然」「特征值/SVD 在 PCA 与 LoRA 里各扮演什么角色」，比会推导更重要。
- **贯穿全阶段的一条主线**：`y = Wx + b` → 加 Sigmoid 变逻辑回归 → 堆叠+激活变神经网络。线性回归和逻辑回归就是神经网络的最小细胞，务必吃透。
- **偏差-方差权衡是评判一切模型的标尺**：训练分高、验证分低 = 过拟合（高方差）；两者都低 = 欠拟合（高偏差）。每学一个算法都问：它的哪个超参在这条轴上滑动（如 SVM 的 `C`/`gamma`、树的 `max_depth`）。
- **评估指标先于算法**：不平衡数据上 Accuracy 是陷阱，先想清楚该用 F1 还是 AUC-ROC、该保 Recall 还是 Precision，再动手建模。
- **Pipeline + 交叉验证是防作弊底线**：标准化/PCA/编码的 `fit` 只能在训练折上做，任何在全量数据上 `fit` 再切分的写法都会造成数据泄露、指标虚高。
- **先手写一遍再调库**：每个算法先用 NumPy 实现核心几行（sigmoid、MSE 梯度、Gini），再用 sklearn 一行搞定——手写建立理解，调库建立生产力。

---

## 🔗 关联

- **上一阶段**：[阶段二 · Python 数据科学基础](../02-data-science-fundamentals/) — NumPy/Pandas 是本阶段所有代码的底座
- **下一阶段**：[阶段四 · 深度学习基础](../04-deep-learning-basics/) — 把线性回归/逻辑回归/梯度下降升级为多层网络与反向传播
- **Agent 课程**：本阶段的线性代数（余弦相似度、向量点积）直接支撑 [agent-course/Day-16 · Embedding 基础](../agent-course/Day-16-embedding-basics.md)
