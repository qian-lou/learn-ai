# 阶段二：Python 数据科学基础

> **预估周期**：2-3 周
> **核心目标**：掌握数据科学三件套 NumPy / Pandas / Matplotlib，建立"向量化 + 结构化 + 可视化"的数据处理能力，为机器学习与大模型开发打底
> **前置**：[阶段一 · Python 基础](../01-python-basics/README.md)

---

## 🧭 本阶段总览

数据科学三件套构成一条完整的数据流水线，也是通往 PyTorch 与大模型工程的必经之路：

```
NumPy  ──向量化数值计算──▶  Pandas  ──结构化清洗/分析──▶  Matplotlib
（张量与广播的雏形）        （像写 SQL 一样处理数据）        （把结果画出来看懂）
        │                          │                            │
        └──────▶ 直接支撑 PyTorch Tensor / Embedding / 模型评估 ◀──────┘
```

- **NumPy** 教会你"永远不写逐元素循环"的向量化思维，其 `ndarray` 与广播机制是 PyTorch `Tensor` 的直系前身。
- **Pandas** 让你像写 SQL 一样清洗、分组、关联结构化数据——这是训练前 80% 工作量所在。
- **Matplotlib/Seaborn** 把训练曲线、混淆矩阵、特征分布可视化，是调优与汇报的必备手段。

---

## 📋 模块大纲

### [01-numpy](./01-numpy/) — NumPy 数值计算

高性能数值计算库，AI 开发的数学运算基石；`ndarray` 与广播是 PyTorch `Tensor` 的前身。

| 序号 | 文件 | 主题 | 核心要点 |
|------|------|------|---------|
| 01 | [ndarray-basics](./01-numpy/01-ndarray-basics.md) | ndarray 基础与创建 | 创建/属性/`reshape`；向量化替代 for 循环 |
| 02 | [indexing-and-slicing](./01-numpy/02-indexing-and-slicing.md) | 索引与切片 | 布尔/花式索引、`np.where`；视图 vs 副本陷阱 |
| 03 | [broadcasting](./01-numpy/03-broadcasting.md) | 广播机制 | 从右向左对齐三规则、`keepdims`、`np.newaxis` |
| 04 | [linear-algebra-ops](./01-numpy/04-linear-algebra-ops.md) | 线性代数运算 | `@`/`solve`/SVD/`einsum`；LoRA 理论基础 |
| 05 | [performance-optimization](./01-numpy/05-performance-optimization.md) | 性能优化 | 向量化 vs 循环（100-1000x）、`out=` 原地运算 |

---

### [02-pandas](./02-pandas/) — Pandas 数据处理

结构化数据处理核心工具，对标 Java 中的 JDBC + Stream 操作，是训练前数据清洗与分析的主力。

| 序号 | 文件 | 主题 | 核心要点 |
|------|------|------|---------|
| 01 | [series-and-dataframe](./02-pandas/01-series-and-dataframe.md) | Series 与 DataFrame | `loc`/`iloc`、布尔过滤、`apply` 派生列 |
| 02 | [data-io](./02-pandas/02-data-io.md) | 数据读写（CSV/JSON/SQL/Parquet） | `usecols/dtype/chunksize`；Parquet 谓词下推 |
| 03 | [data-cleaning](./02-pandas/03-data-cleaning.md) | 数据清洗（缺失值/重复值/异常值） | `fillna/dropna/drop_duplicates/clip`/IQR |
| 04 | [groupby-and-aggregation](./02-pandas/04-groupby-and-aggregation.md) | 分组与聚合（SQL 对比） | Split-Apply-Combine、`agg` vs `transform` |
| 05 | [merge-and-join](./02-pandas/05-merge-and-join.md) | 合并与连接（JOIN 对比） | `merge` 四种 `how`、`concat` 纵横拼接 |

---

### [03-matplotlib](./03-matplotlib/) — Matplotlib 数据可视化

数据可视化工具，掌握图表绘制与统计图高级美化，服务于模型评估与结果呈现。

| 序号 | 文件 | 主题 | 核心要点 |
|------|------|------|---------|
| 01 | [basic-plotting](./03-matplotlib/01-basic-plotting.md) | 基础绘图（折线/柱状/散点/直方图） | 四类图语义、Figure/Axes、`savefig` |
| 02 | [subplot-and-layout](./03-matplotlib/02-subplot-and-layout.md) | 子图与布局 | `subplots`/`twinx` 双轴/`GridSpec` |
| 03 | [seaborn-advanced](./03-matplotlib/03-seaborn-advanced.md) | Seaborn 高级可视化 | `heatmap`/`boxplot`/`violinplot`；长格式 |

---

## 🎯 阶段学习要点

- **向量化思维是本阶段的灵魂**：从此告别 Java 式逐元素 for 循环，任何数组/DataFrame 运算都用向量化表达，写完自查能否消掉循环。
- **吃透广播机制**：能手推两个 shape 能否广播及结果 shape——这是后续 PyTorch、Attention、BatchNorm 的直接前置。
- **建立 SQL 映射直觉**：Pandas 的过滤/选列/排序/分组/连接逐一对上 WHERE/SELECT/ORDER BY/GROUP BY/JOIN，用已有 SQL 经验加速上手。
- **数据清洗成套路**：`isnull().sum()` → 填充/删除 → 去重 → 异常值裁剪 → 类型/字符串规整，练成一条可复用的清洗管线。
- **掌握 ML 三张必备图**：训练/验证曲线、混淆矩阵、特征相关性矩阵，覆盖模型调优与汇报的核心可视化需求。
- **打通一条端到端小流水线**：用一份真实 CSV 走通"读入(Pandas) → 清洗 → 向量化统计(NumPy) → 可视化(Matplotlib/Seaborn)"，把三件套串成肌肉记忆。

---

## 🔗 关联

- **上一阶段**：[阶段一 · Python 基础](../01-python-basics/README.md)
- **下一阶段**：[阶段三 · 机器学习基础](../03-machine-learning-basics/README.md)（本阶段三件套是 scikit-learn 建模的直接工具）
- **Agent 课程衔接**：本阶段能力支撑 [agent-course · Day 15 数据查询 Agent](../agent-course/Day-15-data-query-agent.md)、[Day 16 Embedding 与余弦相似度](../agent-course/Day-16-embedding-basics.md)、[Day 19 嵌入 ETL Pipeline](../agent-course/Day-19-embedding-etl.md)。
