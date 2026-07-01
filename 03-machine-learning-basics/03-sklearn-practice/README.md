# 03-sklearn-practice — Scikit-learn 实战

> **所属阶段**：阶段三 · 机器学习基础
> **学习目标**：使用 Scikit-learn 构建端到端机器学习项目，把算法理论落成可复现、防泄露的工程流水线
> **预估时长**：4-5 天（重在跑通完整闭环，而非记 API）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [sklearn-pipeline](./01-sklearn-pipeline.md) | Scikit-learn Pipeline | `Pipeline` 串联预处理+模型、`ColumnTransformer` 分列处理、`__` 命名引用内部超参、与 GridSearchCV 无缝集成、joblib 序列化 |
| 02 | [feature-engineering](./02-feature-engineering.md) | 特征工程 | StandardScaler/MinMaxScaler 缩放、OneHot/Ordinal 编码、PCA 降维（保留 95% 方差）、SelectKBest/VarianceThreshold 特征选择、目标变量 log 变换 |
| 03 | [end-to-end-project](./03-end-to-end-project.md) | 端到端 ML 项目实战 | 完整七步流程：问题定义 → EDA → 预处理 → 特征工程 → 多模型对比 → 交叉验证调优 → 评估落盘部署 |

---

## 🔑 知识点详解

### 01 · Scikit-learn Pipeline

- **核心概念**：把「预处理 + 模型」封装成一个对象，`fit` 时预处理只学训练集统计量，从机制上杜绝数据泄露。
- **关键 API**：`Pipeline([('scaler', ...), ('clf', ...)])`；跨列不同处理用 `ColumnTransformer`；网格搜索用双下划线引用步骤内参数，如 `'clf__max_depth'`、`'pca__n_components'`；`joblib.dump(pipe, 'model.pkl')` 落盘。
- **易错点**：① `GridSearchCV` 的参数名必须带步骤前缀（`步骤名__参数名`），漏了会报 invalid parameter；② 只有除最后一步外的步骤需实现 `transform`，最后一步是 estimator；③ 把标准化留在 Pipeline 外、在全量数据上先 `fit_transform` 再切分，是最常见的泄露写法。
- **Java 视角**：Pipeline 就是责任链模式（Chain of Responsibility），每个 Transformer 处理后把数据传给下一环，最后交给 Estimator——和 Servlet Filter 链、Spring 拦截器链同构。
- **前置**：feature-engineering 的各类预处理器、classic-algorithms 的估计器。

### 02 · 特征工程

- **核心概念**：「数据和特征决定 ML 的上限，算法只是逼近它」——缩放、编码、降维、选择四件套把原始数据整理成模型友好的形式。
- **关键 API**：`StandardScaler`（z-score，`(x-μ)/σ`）、`MinMaxScaler`（缩到 [0,1]）、`OneHotEncoder(handle_unknown='ignore')`、`OrdinalEncoder`（有序类别）、`PCA(n_components=0.95)`（按方差比例自动定维）、`SelectKBest(f_classif, k=...)`、`np.log1p`/`np.expm1`（长尾目标变换）。
- **易错点**：① 缩放器必须「训练集 `fit_transform`，测试集只 `transform`」，测试集 `fit` 即泄露；② `LabelEncoder` 只用于目标 `y`，输入的有序特征用 `OrdinalEncoder`、无序用 `OneHotEncoder`；③ 目标变量长尾（幂律）分布会拖垮最小二乘，先 `log1p` 变换、预测后 `expm1` 还原。
- **Java 视角**：特征工程类比 DTO/VO 转换层——把杂乱的原始数据字段清洗、映射、组装成服务层能直接消费的规整结构。
- **前置**：阶段二 Pandas 数据清洗、math-foundations 的 PCA（特征值分解）。

### 03 · 端到端 ML 项目实战

- **核心概念**：一个完整项目 = 问题定义 → EDA → 预处理 → 特征工程 → 多模型对比 → 交叉验证调优 → 评估 → 部署，与大模型微调项目的骨架一致。
- **关键 API**：`train_test_split(stratify=y)`、`cross_val_score(model, X, y, cv=5, scoring='f1')`、`classification_report`、`joblib.dump/load`；用 dict 装多个 Pipeline 批量比模型。
- **易错点**：① 分类切分务必 `stratify=y` 保持类别比例；② 交叉验证在训练集内做，测试集只在最后评估一次，反复看测试集调参也是变相泄露；③ 高偏差（欠拟合）与高方差（过拟合）的治理手段相反，先诊断再对症，别盲目加正则或加特征。
- **Java 视角**：ML 项目流程与后端项目同构——EDA≈需求调研、特征工程≈数据建模、GridSearch≈参数调优、joblib 保存≈打包制品、Flask/FastAPI 部署≈发布服务。工具不同，流程一致。
- **前置**：01 Pipeline、02 特征工程、classic-algorithms 全部（尤其 model-evaluation）。

---

## 🎯 学习要点

- **Pipeline 模式保证数据预处理和模型训练的一致性**：训练怎么处理，推理就自动怎么处理，杜绝「训练/线上特征不一致」这类生产事故。
- **特征工程是提升模型效果的关键手段**：同一算法，好特征能带来的提升往往超过换更复杂的模型，优先把时间花在数据上。
- **通过端到端项目串联所有知识点**：拿 `load_breast_cancer` 或 `make_regression` 完整跑一遍七步流程，比孤立学十个 API 更有价值。
- **牢记数据泄露的三种典型形态**：全量 `fit` 缩放、测试集参与 `fit`、反复用测试集调参——三条都要在 Pipeline + 交叉验证的框架下堵死。
- **先建基线再迭代**：先用逻辑回归/随机森林拿一个能跑的 baseline，再逐步加特征、调参、换模型，每步都用交叉验证量化增益。
- **模型要能落盘复现**：`joblib.dump` 保存整个 Pipeline（含预处理），加载后能直接 `predict`，这才是可交付的制品。

---

## 🔗 关联

- **上一模块**：[02-classic-algorithms](../02-classic-algorithms/) — 提供 Pipeline 里可插拔的各种估计器与评估方法
- **下一模块**：（本阶段末模块）进入 [阶段四 · 深度学习基础](../../04-deep-learning-basics/)，把 sklearn 的建模范式升级到 PyTorch 训练循环
- **本阶段总览**：[阶段三 README](../README.md)
- **Agent 课程**：端到端流程、joblib 制品化与「训练/推理一致性」思想，直通 [agent-course/Day-19 · Embedding ETL](../../agent-course/Day-19-embedding-etl.md) 的数据流水线设计
