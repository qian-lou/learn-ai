# 02-classic-algorithms — 经典算法

> **所属阶段**：阶段三 · 机器学习基础
> **学习目标**：掌握经典机器学习算法的原理与实现，理解监督/无监督两大范式与偏差-方差取舍
> **预估时长**：8-10 天（算法多，建议每个先手写核心再调库）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [linear-regression](./01-linear-regression.md) | 线性回归 | 最小二乘/MSE、正规方程解析解、梯度下降数值解、L1(Lasso)/L2(Ridge) 正则、R² 评估 |
| 02 | [logistic-regression](./02-logistic-regression.md) | 逻辑回归 | Sigmoid 映射概率、二元交叉熵损失、决策边界、Softmax 多分类、`predict_proba` 与 AUC |
| 03 | [decision-tree-and-random-forest](./03-decision-tree-and-random-forest.md) | 决策树与随机森林 | Gini/信息增益分裂、剪枝防过拟合、Bagging（Bootstrap+投票）、特征重要性、无需缩放 |
| 04 | [svm](./04-svm.md) | 支持向量机 | 最大间隔、Hinge Loss 与软间隔 C、对偶问题与 KKT、核技巧（线性/RBF/poly）、OvO/OvR 多分类、大数据禁区 |
| 05 | [clustering](./05-clustering.md) | 聚类算法 | KMeans（Lloyd 迭代+k-means++）、DBSCAN（密度可达+噪声）、层次聚类、GMM/EM、选 K（肘部/轮廓系数） |
| 06 | [model-evaluation](./06-model-evaluation.md) | 模型评估与调优 | 混淆矩阵、Precision/Recall/F1、AUC-ROC、交叉验证（StratifiedKFold）、GridSearchCV、过拟合检测 |

---

## 🔑 知识点详解

### 01 · 线性回归

- **核心概念**：用一条超平面 `y = Wx + b` 拟合连续目标，最小化均方误差 MSE。
- **关键公式/API**：正规方程解析解 `w = (XᵀX)⁻¹Xᵀy`（一步到位，无需迭代）；`LinearRegression`、`Ridge(alpha)`、`Lasso(alpha)`。
- **易错点**：① 特征数 D 远大于样本 N 时严重过拟合，必须上正则（Ridge 压权重，Lasso 出稀疏解）；② `XᵀX` 不可逆时正规方程失效，改用伪逆或梯度下降；③ R²=0 不代表模型没用，它意味着效果等同于「直接用均值预测」。
- **Java 视角**：正规方程是「一次解析求解」，梯度下降是「迭代逼近」——类比精确公式解 vs. 数值逼近，大数据下后者更省内存。
- **前置**：math-foundations 的梯度下降、矩阵求逆/伪逆。

### 02 · 逻辑回归

- **核心概念**：线性回归外套一个 Sigmoid 把输出压到 (0,1) 当概率，是最基础的分类器，也是神经网络输出层的原型。
- **关键公式/API**：`σ(z) = 1/(1+e⁻ᶻ)`；二元交叉熵 `L = -[y·log(p)+(1-y)·log(1-p)]`；`LogisticRegression(max_iter=1000)`、`clf.predict_proba(X)[:,1]` 取正类概率。
- **易错点**：① 名字叫「回归」实为分类，别用它做连续值预测；② `p` 接近 0/1 时 `log` 溢出，须 `np.clip(p, 1e-15, 1-1e-15)`；③ sklearn 1.5+ 多分类默认即 multinomial，`multi_class` 参数已废弃，别再传。
- **Java 视角**：BERT/GPT 的分类头 `Linear(d, num_classes) → Softmax → CE Loss` 本质就是一层逻辑回归——学好它等于看懂了大模型的分类出口。
- **前置**：01 线性回归、math-foundations 的 Sigmoid/交叉熵。

### 03 · 决策树与随机森林

- **核心概念**：树按特征阈值层层二分，用纯度（Gini/熵）挑最优分裂点；随机森林用 Bagging 集成多棵树投票降方差。
- **关键公式/API**：Gini `1 - Σpₖ²`（越小越纯）；`DecisionTreeClassifier(max_depth, min_samples_leaf)`、`RandomForestClassifier(n_estimators)`、`.feature_importances_`。
- **易错点**：① 单棵树极易过拟合，靠 `max_depth`/`min_samples_leaf` 剪枝控制；② 决策树/树集成**无需特征缩放**（分裂只看阈值大小顺序），这是相对 SVM/线性模型的一大便利；③ 特征重要性会偏向高基数（取值多）的特征，解读需谨慎。
- **Java 视角**：一棵树就是一串嵌套 `if-else` 规则，可读性极强；随机森林类比「多个专家独立投票再取多数」，用冗余换稳健。
- **前置**：math-foundations 的信息熵。

### 04 · 支持向量机（SVM）

- **核心概念**：在两类之间找「间隔最大」的超平面，只有边界附近的支持向量决定模型；核技巧让它无需显式升维就能处理非线性。
- **关键公式/API**：软间隔目标 `min ½‖w‖² + CΣξᵢ`，等价于 Hinge Loss + L2 正则；RBF 核 `K = exp(-γ‖x-z‖²)`；`SVC(kernel='rbf', C, gamma)`，`LinearSVC` 用于高维稀疏。
- **易错点**：① **必须先标准化**再喂 SVM（对尺度极敏感），且缩放要包进 Pipeline 防泄露；② `C` 与 `gamma` 强耦合，须二维网格联合搜索，不能各调各的；③ 核 SVM 训练对样本数超线性、核矩阵 O(N²) 内存，N 超十万级基本 OOM，大数据换 `LinearSVC` 或 Nyström 近似核。
- **Java 视角**：Hinge Loss「间隔外损失为 0」类比熔断阈值——只有越界请求才触发惩罚/告警，安全区内完全不计成本，与逻辑回归「每个点都持续施力」形成对比。
- **前置**：01/02 回归打底、math-foundations 的范数与凸优化、拉格朗日/KKT（对偶部分为选读进阶）。

### 05 · 聚类算法

- **核心概念**：无监督地把样本按相似度分组。KMeans 靠质心与簇内平方和，DBSCAN 靠密度，GMM 靠概率软分配，层次聚类给树状结构。
- **关键公式/API**：KMeans 目标 `J = ΣΣ‖x-μₖ‖²`（inertia）；轮廓系数 `s = (b-a)/max(a,b) ∈ [-1,1]`；`KMeans(n_clusters, init='k-means++', n_init='auto')`、`DBSCAN(eps, min_samples)`、`silhouette_score`。
- **易错点**：① KMeans 必须预设 K 且只收敛到局部最优，靠 k-means++ 初始化 + 多次 `n_init` 缓解；② DBSCAN 的 `eps` 对数据尺度极敏感，**别硬编码**，用 k-距离图找膝点或先标准化；③ 评估 DBSCAN 的轮廓系数要剔除噪声点（标签 -1）；④ sklearn 1.4+ `n_init` 默认已改为 `'auto'`（k-means++ 下等于 1），写死 `n_init=10` 语义会变。
- **Java 视角**：KMeans 分配步是天然数据并行的（N 点算 argmin 互不依赖），对应 Spark MLlib 的 map-reduce；Java 中并行务必按 P3C 显式 `ThreadPoolExecutor`，禁用 `Executors` 工厂防 OOM。
- **前置**：math-foundations 的欧氏距离/范数、高斯分布（GMM 用）。

### 06 · 模型评估与调优

- **核心概念**：准确率不是万能指标，评估要匹配业务代价，并用交叉验证得到稳健估计。
- **关键公式/API**：Precision `TP/(TP+FP)`、Recall `TP/(TP+FN)`、F1 `2PR/(P+R)`；`classification_report`、`confusion_matrix`、`roc_auc_score`、`cross_val_score(scoring=...)`、`GridSearchCV`。
- **易错点**：① 不平衡数据上 Accuracy 会骗人（全猜多数类也能 99%），改用 F1/AUC-ROC；② 分类交叉验证要用 `StratifiedKFold` 保持各折类别比例；③ AUC-ROC 衡量排序能力、与阈值无关，Precision/Recall 依赖阈值，别混用。
- **Java 视角**：选指标 = 选 SLA。医疗诊断保 Recall（不能漏诊）、垃圾邮件保 Precision（不能误杀），正如不同服务对「延迟」和「可用性」的权衡取舍不同。
- **前置**：01-05 各算法（评估的对象）。

---

## 🎯 学习要点

- **先理解算法数学原理，再用 sklearn 快速实现**：每个算法先用 NumPy 写核心几行（sigmoid、Gini、MSE 梯度），建立理解后再一行调库。
- **模型评估是实际项目中最关键的环节**：动手前先问「这是不平衡数据吗？该保 Recall 还是 Precision？」，指标选错，模型调得再好也白费。
- **从简单模型开始，逐步理解复杂模型的动机**：线性回归 → 逻辑回归 → SVM/树，每一步都在解决前一步的短板（非线性、过拟合、可解释）。
- **牢记「偏差-方差」这根主轴**：把每个超参归位——`C` 大/`max_depth` 深/`gamma` 大 → 低偏差高方差（易过拟合），反之易欠拟合。
- **标准化与数据泄露是两条工程红线**：SVM/KMeans/线性模型都需缩放且必须包进 Pipeline，`fit` 只在训练折上做。
- **对结构化表格数据，树集成（RF/XGBoost）常胜过深度学习**：别一上来就上神经网络，先用随机森林拿基线，往往又快又强。

---

## 🔗 关联

- **上一模块**：[01-math-foundations](../01-math-foundations/) — 梯度下降、交叉熵、范数、凸优化是本模块所有算法的数学地基
- **下一模块**：[03-sklearn-practice](../03-sklearn-practice/) — 用 Pipeline + 特征工程把这些算法组装成端到端项目
- **本阶段总览**：[阶段三 README](../README.md)
- **Agent 课程**：聚类用于训练数据去重与多样性分析，与 [agent-course/Day-18 · Chunking](../../agent-course/Day-18-chunking.md)、[Day-19 · Embedding ETL](../../agent-course/Day-19-embedding-etl.md) 的数据处理思路相通
