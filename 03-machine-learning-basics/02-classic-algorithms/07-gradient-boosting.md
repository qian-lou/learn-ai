# 梯度提升
# Gradient Boosting (GBDT / XGBoost / LightGBM)

## 1. 背景（Background）

> **为什么要学这个？**
>
> 随机森林是"并行造很多棵树投票"，梯度提升是"串行造树、每棵专门纠正上一步的错误"。在结构化表格数据上，XGBoost / LightGBM 长期霸榜 Kaggle，工业界的风控、CTR、排序模型也大量用它。前几篇（决策树、`end-to-end-project`）反复把 Boosting 捧为"结构化数据王者"却没讲机制——本篇补上这块最大的空洞。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| Boosting | 串行拟合残差/负梯度，逐步降**偏差** |
| Bagging 对比 | 并行独立造树，降**方差**（随机森林） |
| Shrinkage | 学习率 `learning_rate`，给每棵树的贡献打折防过拟合 |
| 二阶泰勒展开 | XGBoost 用一阶+二阶梯度更精确地找分裂 |
| leaf-wise vs level-wise | LightGBM 按增益最大叶子分裂，XGBoost 按层分裂 |
| 早停 | `early_stopping` + `eval_set`，自动定 `n_estimators` |

## 3. 内容（Content）

```python
# sklearn 自带的现代梯度提升实现（直方图算法，无需装第三方库即可运行）
# / sklearn's built-in histogram-based gradient boosting, runs out of the box
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 树集成无需特征缩放（分裂只看阈值顺序），这是相对 SVM/线性模型的便利
gb = HistGradientBoostingClassifier(
    learning_rate=0.05,      # shrinkage：每棵树贡献打 5% 折
    max_iter=1000,           # 树的上限，交给早停自动收敛
    early_stopping=True,     # 内部划验证集，验证指标不再改善就停
    n_iter_no_change=30,     # 连续 30 轮不改善即早停
    random_state=42)
gb.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])
print(f"HistGB AUC: {auc:.4f}, 实际迭代 / actual iters: {gb.n_iter_}")
# 预期输出约: HistGB AUC: 0.9878, 实际迭代: 147（<1000，被早停提前截断）
```

### 3.1 Boosting vs Bagging：串行降偏差 vs 并行降方差 / Serial Bias-Reduction vs Parallel Variance-Reduction

两者都是"多棵树集成"，但方向截然相反，这决定了各自的调参哲学：

| 维度 | Bagging（随机森林） | Boosting（GBDT） |
|------|--------------------|------------------|
| 造树方式 | **并行**，各树独立、互不依赖 | **串行**，第 $m$ 棵树依赖前 $m-1$ 棵的结果 |
| 每棵树目标 | 各自拟合原始标签 | 拟合当前集成的**残差/负梯度** |
| 主要降低 | **方差**（多个高方差树平均抵消） | **偏差**（逐步逼近真实函数） |
| 单棵树倾向 | 深、强（低偏差高方差） | 浅、弱（`max_depth` 常 3~6） |
| 加树的风险 | 几乎不过拟合，越多越稳 | **会过拟合**，需 shrinkage + 早停 |

一句话直觉：随机森林是"三个臭皮匠取平均，把随机误差磨平"；梯度提升是"专家团队接力，后一个专门修正前一个的系统性错误"。因为 Boosting 主动降偏差，所以它的基学习器要**弱而浅**——太强会一步吃掉太多残差，反而不稳。

### 3.2 手写梯度提升核心：拟合负梯度 + shrinkage / Hand-Written Core

梯度提升的通用框架（Gradient Boosting Machine）：把预测函数写成 $F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta\, h_m(x)$，每一步让新树 $h_m$ 去拟合损失对**当前预测的负梯度**。对最常见的平方损失 $L=\tfrac12(y-F)^2$，负梯度恰好就是**残差** $y-F$——这就是"拟合残差"这一说法的由来。

```python
import numpy as np

# 一个最小回归树桩(depth=1): 遍历切分点选平方误差最小者 / Minimal regression stump
# 时间 O(N·U) 空间 O(1)，U 为候选阈值数 / U = number of candidate thresholds
def fit_stump(x: np.ndarray, r: np.ndarray):
    best_sse, thr, pl, pr = np.inf, None, r.mean(), r.mean()
    for t in np.unique(x)[:-1]:                 # 每个候选阈值把样本二分
        L, R = r[x <= t], r[x > t]
        pred_l, pred_r = L.mean(), R.mean()     # 叶子输出=该侧残差均值
        sse = ((L - pred_l) ** 2).sum() + ((R - pred_r) ** 2).sum()
        if sse < best_sse:
            best_sse, thr, pl, pr = sse, t, pred_l, pred_r
    return thr, pl, pr

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, 1.0, 10.0, 10.0])           # 一个阶跃，便于心算核对
F = np.full_like(y, y.mean())                  # 初始预测 = 均值 5.5
lr = 0.5                                        # shrinkage 学习率
# 梯度提升主循环 / GBM main loop: 时间 O(M·N·U) 空间 O(N)
for m in range(3):
    residual = y - F                           # 平方损失的负梯度 = 残差
    thr, pl, pr = fit_stump(x, residual)       # 新树拟合残差
    update = np.where(x <= thr, pl, pr)
    F = F + lr * update                        # 累加"打折后"的树输出
    print(f"iter{m}: F={np.round(F, 4)}")
# 预期输出（每步把差距缩掉一半，因 lr=0.5）:
#   iter0: F=[3.25 3.25 7.75 7.75]
#   iter1: F=[2.125 2.125 8.875 8.875]
#   iter2: F=[1.5625 1.5625 9.4375 9.4375]  → 正逐步逼近 [1,1,10,10]
```

关键点：(1) `residual = y - F` 就是负梯度，换成别的损失只需换这一行梯度公式；(2) `lr`（shrinkage）让每棵树只补一部分差距，步子小、树多，泛化更好——这是"learning_rate 小 + n_estimators 大"这一黄金搭配的数学根源。

### 3.3 从 GBDT 到 XGBoost：二阶泰勒 + 正则 + 缺失值 + 列采样 / GBDT → XGBoost

原始 GBDT 只用一阶梯度（残差）。XGBoost 把损失在当前预测处做**二阶泰勒展开**，同时用上一阶梯度 $g_i$ 和二阶梯度（Hessian）$h_i$：

$$\mathcal{L}^{(m)} \approx \sum_i \bigl[g_i\, h_m(x_i) + \tfrac12 h_i\, h_m(x_i)^2\bigr] + \Omega(h_m),\qquad \Omega = \gamma T + \tfrac12\lambda\|w\|^2$$

由此可解析地推出每个叶子的**最优权重** $w_j^* = -\dfrac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda}$，以及分裂的**增益公式**（对左右子节点的 $\tfrac{(\sum g)^2}{\sum h + \lambda}$ 求差再减 $\gamma$）。这带来四项决定性改进：

| 改进 | 作用 |
|------|------|
| **二阶泰勒展开** | 用曲率 $h_i$ 更精确定位最优叶子权重，收敛更快更稳 |
| **正则项** $\Omega$ | $\lambda$（`reg_lambda`）压叶子权重、$\gamma$（`gamma`）罚叶子数，内建防过拟合 |
| **缺失值处理** | 每个分裂学一个"默认方向"，缺失样本自动走该方向，无需手工填充 |
| **列采样** | `colsample_bytree` 像随机森林那样随机选特征子集，进一步去相关、防过拟合 |

### 3.4 LightGBM：leaf-wise + 直方图 + 原生类别特征 / LightGBM Innovations

LightGBM 与 XGBoost 目标函数同源，但在"怎么快速造树"上做了三处工程革新：

- **直方图算法（Histogram）**：把连续特征分到固定 bin（如 255 个），分裂时只需在直方图上扫 bin 而非扫每个样本值，把找分裂的复杂度从 $O(N\cdot D)$ 降到 $O(\text{bins}\cdot D)$，内存也大幅下降。
- **leaf-wise 生长**：每次在**所有叶子里**挑增益最大的那个分裂（best-first），而非像 XGBoost 默认那样逐层（level-wise）铺满。相同叶子数下 leaf-wise 损失更低，但树会长得不对称、更深，**更容易过拟合**——必须用 `num_leaves` 和 `max_depth` 联合约束。
- **原生类别特征**：`categorical_feature` 直接接收整数编码的类别列，内部用特殊的排序分裂法（依据每个类别的梯度统计排序后二分），**免去 one-hot**，高基数类别列上又快又准。

**XGBoost(level-wise) vs LightGBM(leaf-wise) 取舍**：

| 场景 | 更合适 |
|------|--------|
| 数据量大（百万级+）、追求训练速度 | **LightGBM**（直方图 + leaf-wise 快数倍） |
| 数据较小、想要更稳更不易过拟合的默认行为 | **XGBoost**（level-wise 天然更规整） |
| 有大量高基数类别特征 | **LightGBM**（原生类别支持免 one-hot） |
| 需要极致精调、生态/部署成熟度 | 两者都成熟，XGBoost 历史更久、文档更全 |

> 补充：XGBoost 也支持 `tree_method='hist'`（自 2.0 起为默认），两者的算法差距已远小于早年，实践中常两个都试、按验证集选。

## 4. 详细推理（Deep Dive）

### 4.1 sklearn 接口与早停 / sklearn API & Early Stopping

XGBoost、LightGBM 都提供了 sklearn 风格接口（`fit`/`predict`/`predict_proba`），可直接塞进 `Pipeline`、`GridSearchCV`。核心工程动作是**早停**：给一个验证集 `eval_set`，验证指标连续若干轮不再改善就停，从而无需手调 `n_estimators`。

```python
# ⚠️ 需先安装 / requires: pip install xgboost lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# 再从训练集切出内部验证集用于早停 / carve out a validation set for early stopping
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

# XGBoost 2.x：early_stopping_rounds 移到了构造器里 / moved into the constructor since 2.0
xgb = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    early_stopping_rounds=50, eval_metric="auc")
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print(f"XGB 最佳树数 / best_iteration: {xgb.best_iteration}")

# LightGBM：早停用 callbacks 传入 / early stopping via callbacks
import lightgbm as lgb
lgbm = LGBMClassifier(
    n_estimators=2000, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
lgbm.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
```

### 4.2 关键超参的偏差-方差定位 / Bias-Variance Map of Hyperparameters

把每个超参归位到"偏差-方差"这根主轴上，调参就有了方向感：

| 超参 | 调大的效果 | 偏差-方差 |
|------|-----------|-----------|
| `learning_rate` | 每棵树补更多残差，收敛快 | ↑ 大 → 低偏差**高方差**（须配少树/早停） |
| `n_estimators` | 更多树，拟合更充分 | ↑ 大 → 低偏差高方差（靠早停封顶） |
| `max_depth` / `num_leaves` | 单棵树更复杂 | ↑ 大 → 低偏差**高方差**（GBDT 常 3~6 层） |
| `subsample` | 每棵树只用部分样本 | ↓ 小 → 高偏差**低方差**（去相关、防过拟合） |
| `colsample_bytree` | 每棵树只用部分特征 | ↓ 小 → 高偏差低方差 |
| `reg_lambda` / `reg_alpha` | L2/L1 正则强度 | ↑ 大 → **高偏差低方差** |

黄金搭配：**`learning_rate` 调小 + `n_estimators` 调大 + 早停**，几乎总能换来更好的泛化，代价是训练更久。

### 4.3 特征重要性：split count vs gain / Feature Importance

树集成天然给出特征重要性，但**默认口径要看清**：

```python
# HistGradientBoosting 无内建 feature_importances_，用排列重要性（更可靠）
# / permutation importance is model-agnostic and less biased
from sklearn.inspection import permutation_importance
r = permutation_importance(gb, X_te, y_te, n_repeats=10, random_state=42)
print("排列重要性 top: / permutation importance:", r.importances_mean.argsort()[::-1][:3])

# XGBoost/LightGBM 的 feature_importances_ 默认口径不同，需显式指定：
#   XGBoost:  importance_type='gain'（分裂带来的平均增益，推荐）vs 'weight'（被选次数）
#   LightGBM: importance_type='gain' vs 'split'
```

要点：**`'weight'`/`'split'`（被选次数）会偏向高基数、连续型特征**（它们能提供更多切分点），解读模型贡献时优先看 **`'gain'`**（平均增益）；跨模型对比或要严谨结论时，用与模型无关的**排列重要性**。

### 4.4 复杂度标注 / Complexity Annotation

```
# 设 N=样本数, D=特征维, M=树数(n_estimators), depth=树深, bins=直方图桶数
# 传统精确 GBDT 训练:  Time: O(M · D · N·log N)   ← 每次分裂对特征排序主导
# 直方图 GBDT(LightGBM/HistGB): Time: O(M · D · (N + bins·2^depth))  ← 扫桶而非扫样本
# 预测(单样本):        Time: O(M · depth)          ← 沿 M 棵树各走 depth 步
# 空间:                O(N·D) 数据 + O(M·2^depth) 存树结构
```

关键结论：直方图算法把每次分裂从"对 N 个值排序"降到"扫 bins 个桶"，这是 LightGBM 在大数据上快数倍的根源；预测极快，只与树数×树深有关，与训练集大小无关。

## 5. 例题（Worked Examples）

### 例题 1：随机森林基线 → 梯度提升调优的完整对照 / RF Baseline → Boosting Tuned

工程实践的标准动作：先用随机森林拿一个又快又强的基线，再上梯度提升 + 早停争取更优。本例用 sklearn 自带的乳腺癌数据集，梯度提升实现用**开箱即用**的 `HistGradientBoostingClassifier`（若装了 xgboost/lightgbm，把注释里的两行换上即可，超参一一对应）。

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. 数据 / Data
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 随机森林基线（Bagging，无需调参就很强）/ RF baseline
# Time: O(M · D · N·log N)（M 棵树可并行）Space: O(M · N)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
auc_rf = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])

# 3. 梯度提升 + 早停（Boosting，串行降偏差）/ Boosting with early stopping
#    小 learning_rate + 大 max_iter + 早停 = 泛化黄金搭配
# ⚠️ 换 xgboost: from xgboost import XGBClassifier
#       gb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
#                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
#                          early_stopping_rounds=30, eval_metric='auc')
#       gb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)  # 真实项目用独立验证集
gb = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=1000, early_stopping=True,
    n_iter_no_change=30, random_state=42)
gb.fit(X_tr, y_tr)
auc_gb = roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])

print(f"RF   基线 AUC / baseline: {auc_rf:.4f}")
print(f"GB   调优 AUC / tuned:    {auc_gb:.4f}, 实际迭代: {gb.n_iter_}")
# 预期输出约:
#   RF   基线 AUC: 0.9937
#   GB   调优 AUC: 0.9878, 实际迭代: 147
```

要点：(1) 本数据集样本少、噪声低，RF 基线已极强，梯度提升未必反超——**这正是"先跑基线"的价值：知道努力的天花板在哪**；(2) 早停让实际迭代（147）远小于上限（1000），无需手调树数；(3) 真实项目里 `eval_set` 必须是**独立验证集**，不能拿测试集早停（会泄漏）。

### 例题 2：learning_rate 与 n_estimators 的此消彼长 / The lr–n_estimators Trade-off

验证 3.2 的黄金搭配：固定其他超参，看学习率越小是否需要越多的树、以及泛化是否更好。

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=4000, n_features=30, n_informative=10,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for lr in (0.3, 0.1, 0.03):
    gb = HistGradientBoostingClassifier(
        learning_rate=lr, max_iter=1000, early_stopping=True,
        n_iter_no_change=30, random_state=42)
    gb.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])
    print(f"lr={lr:<4}  实际树数={gb.n_iter_:<4}  测试 AUC={auc:.4f}")
# 预期趋势: lr 越小 → 早停时树数越多（步子小需更多步补齐）
#   典型输出: lr=0.3 树≈78, lr=0.1 树≈181, lr=0.03 树≈379；三者 AUC 相当(≈0.989)
#   干净数据上差异在噪声内；数据越噪、越易过拟合，小 lr 的泛化优势越明显
```

结论：小学习率把"一步大跳"拆成"多步小走"，用更多的树换更细的逼近，每步过拟合更少。干净数据上几种设置精度相当，但**数据越噪、越易过拟合，小 lr + 大树数 + 早停的泛化优势越明显**——这就是竞赛里 `learning_rate` 常压到 0.01~0.05 的原因，代价只是训练时间。

## 6. 习题（Exercises）

### 基础题
**练习 1**：用一句话说清 Bagging（随机森林）和 Boosting（梯度提升）在"降什么、怎么造树"上的本质区别；并解释为什么梯度提升的基学习器要用**浅**树。
*参考答案*：
- **Bagging**：并行造独立的树、各自拟合原标签，靠平均**降方差**；**Boosting**：串行造树、每棵拟合上一步的残差/负梯度，靠逐步逼近**降偏差**。
- 梯度提升主动降偏差，靠"很多棵弱树接力"实现。基学习器若太深太强，单棵就吃掉大部分残差，步子过猛、方差飙升、易过拟合；浅树（`max_depth` 3~6）弱而稳，配 shrinkage 慢慢补，才能稳定泛化。

### 进阶题
**练习 2**：某 CTR 模型有大量**高基数类别特征**（如 user_id、city），训练集千万级。请说明为什么优先选 LightGBM 而非 XGBoost，并指出用 leaf-wise 生长时最需要设的**两个**防过拟合超参。
*参考答案*：
- **选 LightGBM 的理由**：① 直方图算法在千万级样本上找分裂远快于逐样本扫描；② 原生 `categorical_feature` 支持免去高基数列的 one-hot（否则维度爆炸、稀疏且慢）；③ leaf-wise 在相同叶子数下损失更低。
- **两个关键防过拟合超参**：`num_leaves`（直接约束单棵树复杂度，leaf-wise 下比 `max_depth` 更本质）和 `min_child_samples`/`min_data_in_leaf`（叶子最少样本数，防止为个别样本切出过细的叶子）。此外配合 `learning_rate` 调小 + 早停。
```python
# ⚠️ 需 pip install lightgbm
# LGBMClassifier(num_leaves=63, min_child_samples=100,
#                learning_rate=0.03, n_estimators=2000)  # 再配 eval_set 早停
```
