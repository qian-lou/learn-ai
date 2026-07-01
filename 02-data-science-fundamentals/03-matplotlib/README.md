# 03-matplotlib — 数据可视化

> **所属阶段**：阶段二 · 数据科学基础
> **学习目标**：掌握数据可视化工具，能制作训练曲线、混淆矩阵、特征分布等专业图表
> **预估时长**：3-4 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [basic-plotting](./01-basic-plotting.md) | 基础绘图 | `plot` 折线（训练曲线）、`scatter` 散点、`hist` 直方图、`bar` 柱状；`label/legend/savefig` |
| 02 | [subplot-and-layout](./02-subplot-and-layout.md) | 子图与布局 | `subplots(r,c)` 网格、`sharey`/`twinx` 双轴、`GridSpec` 不规则布局、`tight_layout` |
| 03 | [seaborn-advanced](./03-seaborn-advanced.md) | Seaborn 高级可视化 | `heatmap`（混淆矩阵/相关性）、`boxplot`/`violinplot` 分布、`pairplot`；长格式数据 |

---

## 🔑 知识点详解

### 01 · 基础绘图
- **核心概念**：一张图对应一个 `Figure`（画布），图上一块绘图区是 `Axes`（坐标系）；四类基础图各有语义——折线看趋势、散点看关系、直方图看分布、柱状看对比。
- **关键 API**：`plt.plot(x, y, 'b-o', label=…)`、`plt.scatter(x, y, c=…, cmap=…)`、`plt.hist(data, bins=…)`、`plt.bar(names, vals)`；收尾三件套 `legend()` / `tight_layout()` / `savefig("f.png", dpi=150)`。
- **易错点**：① 脚本环境务必 `savefig` 在 `show()` **之前**——`show()` 后画布可能被清空，导致保存出空白图；② `hist` 的 `bins` 太少会掩盖分布形状，太多则噪声化，需按数据量调；③ 中文标签不设字体会显示成方框，需配置 `rcParams["font.sans-serif"]`。
- **Java 视角**：`pyplot` 的 `plt.*` 是隐式操作"当前图"的全局状态机（类似单例上下文），适合快速出图；复杂图应转 OOP 的 `fig, ax`（见 02）。
- **前置**：NumPy 01（绘图数据基本是 ndarray）。

### 02 · 子图与布局
- **核心概念**：真实分析要在一张图里并排多个视图；`fig, axes = plt.subplots(r, c)` 是推荐的 OOP 风格入口，每个 `ax` 独立控制。
- **关键 API**：`fig, axes = plt.subplots(2, 2, figsize=…)`；子图上用 `axes[i,j].plot(...)`、`.set_title(...)`；`twinx()` 造双 Y 轴（Loss + Accuracy 同图）；`GridSpec(2,2, height_ratios=…)` + `fig.add_subplot(gs[0,:])` 做跨行跨列不规则布局。
- **易错点**：① `subplots` 返回的 `axes` 类型随网格变化——`1×1` 是单个 Axes、`1×N`/`N×1` 是一维数组、`M×N` 是二维数组，取子图前先想清维度；② 忘记 `tight_layout()` 会导致标题/标签重叠；③ 双轴图的图例分散在两个 Axes 上，需用 `fig.legend()` 或手动合并句柄。
- **Java 视角**：`GridSpec` 的 `height_ratios`/跨格 ≈ HTML/CSS Grid 的 `grid-template-rows` 与 `colspan/rowspan`——声明式排版而非逐个摆放。
- **前置**：01（先会画单图）。

### 03 · Seaborn 高级可视化
- **核心概念**：Seaborn 构建在 Matplotlib 之上，面向**统计图表**开箱即用、默认更美观；混淆矩阵、相关性矩阵、分布对比一行搞定。
- **关键 API**：`sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")`（混淆矩阵）；`sns.heatmap(df.corr(), cmap="RdBu_r", vmin=-1, vmax=1, center=0)`（相关性）；`sns.boxplot/violinplot(data=df, x=…, y=…)`；`sns.set_theme(style="whitegrid")` 全局风格。
- **易错点**：① 相关性热力图务必设 `vmin=-1, vmax=1, center=0`，否则色标不对称会误导正负相关；② 混淆矩阵计数用 `fmt="d"`（整数），比例用 `fmt=".2f"`，用错会显示成科学计数法或截断；③ `boxplot/catplot` 期望**长格式（long-form）**数据（一列类别 + 一列数值），宽表要先 `melt`。
- **Java 视角**：Seaborn ≈ 在 Matplotlib 上封装的"高层报表组件库"——你给 DataFrame 和列名，它负责分组统计与美化，类似给模板引擎喂结构化数据。
- **前置**：01、02；数据侧依赖 [Pandas 模块](../02-pandas/README.md)（`df.corr()`、长格式构造）。

---

## 🎯 学习要点

- **分清两种 API 风格**：单图快速探查用 `pyplot`（`plt.*`），多子图/复杂布局一律用 OOP 的 `fig, ax`——后者可控性强、不依赖全局状态。
- **吃透 Figure/Axes 两层架构**：一切设置都问"作用在整张 Figure 还是某个 Axes 上"（`fig.suptitle` vs `ax.set_title`），这是看懂所有 Matplotlib 代码的钥匙。
- **练熟四类基础图的语义**：趋势→折线、关系→散点、分布→直方图、对比→柱状；选错图型比画错更糟。
- **掌握三张 ML 必备图**：训练/验证曲线（`plot` 双线）、混淆矩阵（`heatmap`+`fmt="d"`）、特征相关性矩阵（`heatmap(df.corr())`），这是模型调优与汇报的标配。
- **Seaborn 认长格式**：`boxplot/violinplot/catplot` 传 `data=df, x=, y=`，遇宽表先 `pd.melt` 转长格式再画。
- **出图规范化**：加 `label/legend/title/xlabel/ylabel`，`tight_layout()` 防重叠，`savefig(dpi=150)` 在 `show()` 前保存；中文场景先配字体。

---

## 🔗 关联

- **上一模块**：[02-pandas](../02-pandas/README.md)（可视化的数据来自 Pandas）
- **下一模块**：[阶段三 · 03-machine-learning-basics](../../03-machine-learning-basics/README.md)（可视化服务于模型评估与调优）
- **本阶段总览**：[阶段二 · 数据科学基础](../README.md)
- **Agent 课程衔接**：混淆矩阵、指标分布图是 [agent-course · Day 49 Eval 入门](../../agent-course/Day-49-eval-intro.md) / [Day 50 编写 Evals](../../agent-course/Day-50-writing-evals.md) 中呈现评估结果的常用手段。
