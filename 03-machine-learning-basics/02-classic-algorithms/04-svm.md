# 支持向量机
# Support Vector Machine (SVM)

## 1. 背景（Background）

> **为什么要学这个？**
>
> SVM 通过最大化间隔找到最优决策边界，核技巧可以处理非线性分类。虽然深度学习时代 SVM 用得少了，但其"间隔最大化"思想影响了 contrastive learning。

## 2. 知识点（Key Concepts）

| 概念 | 说明 |
|------|------|
| 最大间隔 | 找到离两类最远的超平面 |
| 支持向量 | 距超平面最近的样本 |
| 核技巧 | 映射到高维空间处理非线性 |
| C 参数 | 正则化（软间隔） |

## 3. 内容（Content）

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ⚠️ SVM 对特征缩放敏感！必须标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 线性 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
print(f"Linear SVM: {svm_linear.score(X_test, y_test):.4f}")

# RBF 核（非线性）
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale')
svm_rbf.fit(X_train, y_train)
print(f"RBF SVM: {svm_rbf.score(X_test, y_test):.4f}")
```

### 3.1 硬间隔、软间隔与 Hinge Loss / Hard Margin, Soft Margin & Hinge Loss

记决策超平面为 $w^\top x + b = 0$，标签 $y_i \in \{-1, +1\}$。SVM 把样本到超平面的**几何间隔**写成 $\gamma_i = y_i (w^\top x_i + b) / \|w\|$。

**硬间隔（Hard Margin）**：假设数据严格线性可分，要求所有点都在间隔之外，目标是最大化最小间隔 $1/\|w\|$，等价于：

$$\min_{w,b} \tfrac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \ge 1,\ \forall i$$

一旦有一个噪声点跨过边界，硬间隔就**无解**——这正是现实中几乎不能直接用它的原因。

**软间隔（Soft Margin）**：引入松弛变量 $\xi_i \ge 0$ 允许少量越界，用参数 $C$ 权衡"间隔宽度"与"违例程度"：

$$\min_{w,b,\xi} \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^{N}\xi_i \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \xi_i \ge 0$$

把约束里的 $\xi_i$ 解出来（$\xi_i = \max(0,\,1 - y_i(w^\top x_i + b))$）并代回，软间隔等价于一个**无约束的正则化经验风险**：

$$\min_{w,b}\ \underbrace{\sum_{i=1}^{N}\max\bigl(0,\ 1 - y_i(w^\top x_i + b)\bigr)}_{\text{Hinge Loss 合页损失}} + \underbrace{\tfrac{1}{2C}\|w\|^2}_{\text{L2 正则}}$$

合页损失的几何含义：分类正确且**间隔 ≥ 1** 时损失为 0（点落在"安全区"，不贡献梯度）；只有间隔不足或分错的点才有损失。这就是 SVM"只被边界附近的支持向量决定"的根源。与逻辑回归的 log loss 对比：log loss 对所有点都有非零梯度（即便分对也会被继续拉远），hinge loss 在间隔之外**完全平坦**——这是两者最本质的区别。

### 3.2 对偶问题与 KKT 条件 / Dual Problem & KKT Conditions

对软间隔的拉格朗日函数求 $w,b,\xi$ 的偏导并令其为零，可消去原变量，得到只含拉格朗日乘子 $\alpha_i$ 的**对偶问题**：

$$\max_{\alpha}\ \sum_{i=1}^{N}\alpha_i - \tfrac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_j\, y_i y_j\, (x_i^\top x_j) \quad \text{s.t.}\quad \sum_i \alpha_i y_i = 0,\ \ 0 \le \alpha_i \le C$$

对偶形式有两个决定性优点：(1) 数据仅以**内积 $x_i^\top x_j$** 出现 → 可直接替换成核函数（见 3.3）；(2) 约束 $0 \le \alpha_i \le C$ 把 $C$ 的作用变成对乘子的**上界封顶**，直观可解释。

由 KKT 互补松弛条件可读出**三类样本**（$f(x)=w^\top x + b$）：

| $\alpha_i$ 取值 | 样本位置 | 角色 |
|----------------|----------|------|
| $\alpha_i = 0$ | $y_i f(x_i) > 1$，在间隔外侧 | 非支持向量，删掉不影响模型 |
| $0 < \alpha_i < C$ | $y_i f(x_i) = 1$，恰在间隔边界上 | **自由支持向量**，用于解出 $b$ |
| $\alpha_i = C$ | $y_i f(x_i) \le 1$，在间隔内或被分错 | **越界支持向量** |

最终 $w = \sum_i \alpha_i y_i x_i$ 只对 $\alpha_i>0$ 的支持向量求和，因此模型的存储与预测成本只与支持向量数量相关，而非全样本。

### 3.3 核技巧的数学直觉 / The Kernel Trick

核技巧的核心：在对偶问题里，凡是出现内积 $x_i^\top x_j$ 的地方，都可替换为 $K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)$——**只要能直接算出高维空间的内积，就无需显式构造高维映射 $\phi$**。这让"先升维再线性分类"的代价从指数级降到 $O(D)$。

| 核 / Kernel | 公式 | 直觉与适用 |
|------------|------|-----------|
| 线性 / Linear | $K=x^\top z$ | 不升维。高维稀疏特征（文本 TF-IDF）天然近似线性可分，首选 |
| 多项式 / Poly | $K=(\gamma\, x^\top z + r)^d$ | 显式构造 $d$ 阶交互特征。`degree=d` 越大越易过拟合，对缩放极敏感 |
| 高斯 / RBF | $K=\exp(-\gamma\|x-z\|^2)$ | 映射到**无穷维**空间；可看作以每个支持向量为中心的相似度"钟形"叠加，最通用的默认核 |

RBF 的 $\gamma$ 几何直觉：$\gamma$ 控制每个支持向量影响的"半径"。$\gamma$ 大 → 半径小、边界贴着样本扭曲 → 高方差易过拟合；$\gamma$ 小 → 半径大、边界平滑 → 高偏差易欠拟合。它与 $C$ 共同决定模型复杂度，必须联合调参。

### 3.4 SVC 关键超参与调参 / Key Hyperparameters of `SVC`

```python
from sklearn.model_selection import GridSearchCV

# 三个最关键的超参 / The three pivotal hyperparameters
#   C      : 正则化强度的倒数。大 → 重视分对（低偏差高方差）；小 → 重视间隔（高偏差低方差）
#   gamma  : 仅 RBF/poly 有效。'scale'(默认)=1/(n_features*X.var())，数据驱动，通常先用它
#   kernel : 'linear' / 'rbf' / 'poly'。线性不可分时上 RBF
# C 与 gamma 强耦合，须二维网格联合搜索，且必须在标准化之后 / Search jointly, after scaling
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf'],
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"最优参数 / Best params: {grid.best_params_}")
```

调参口诀：先固定 `kernel='rbf'`、`gamma='scale'`，对 $C$ 做粗扫定量级；再在最优 $C$ 附近联合细扫 $(C,\gamma)$。**`probability=True` 会触发内部交叉验证、训练显著变慢**，只在确实需要 `predict_proba` 时开启。

### 3.5 多分类策略 OvO / OvR / Multiclass Strategies

SVM 本是二分类器，sklearn 用两种策略扩展到 $K$ 类：

| 策略 | 训练分类器数 | 单个分类器规模 | sklearn 位置 |
|------|-------------|----------------|--------------|
| OvR (One-vs-Rest) | $K$ | 用全部 $N$ 样本 | `LinearSVC` 默认 |
| OvO (One-vs-One) | $K(K-1)/2$ | 仅用两类的样本，更小更快 | **`SVC` 默认** |

`SVC` 默认 OvO，因为核 SVM 复杂度对样本数是超线性的，把样本拆小后总成本反而更低（`decision_function_shape='ovr'` 只改变输出形状，不改变内部训练策略）。

### 3.6 SVM vs 逻辑回归 / SVM vs Logistic Regression

| 维度 | SVM (hinge) | 逻辑回归 (log loss) |
|------|-------------|---------------------|
| 决策依据 | 仅边界附近的支持向量 | 全部样本共同决定 |
| 间隔外的点 | 损失为 0，不影响 | 仍贡献梯度，被持续拉远 |
| 概率输出 | 无（需额外 Platt 校准） | 天然输出校准良好的概率 |
| 非线性 | 核技巧，开箱即用 | 需手工造特征 |
| 大数据扩展 | 差（超线性，见 3.7） | 好（SGD 线性可扩展） |

经验法则：**特征维度 ≫ 样本数**（如文本）选线性 SVM/逻辑回归；中等样本 + 非线性边界选 RBF-SVM；需要概率或海量数据选逻辑回归。

### 3.7 为何大数据集慎用 SVM / Why Avoid SVM on Big Data

核 SVM 训练要求解一个**稠密二次规划（QP）**，需构造 $N\times N$ 的核矩阵（Gram matrix）：

- **时间复杂度**：训练介于 $O(N^2 \cdot D)$ 到 $O(N^3)$（取决于支持向量比例与求解器收敛），预测 $O(n_{SV}\cdot D)$。
- **空间复杂度**：核矩阵 $O(N^2)$。$N=10^5$ 时核矩阵约 $10^{10}$ 个浮点数（约 80 GB），直接 OOM。

因此核 `SVC` 通常只适合 $N \lesssim 10^4 \sim 10^5$。大数据替代方案：(1) 线性问题用 `LinearSVC`/`SGDClassifier(loss='hinge')`（基于 liblinear/SGD，对样本近似线性可扩展）；(2) 非线性用 **Nyström / `RBFSampler` 近似核映射** + 线性模型，把 $O(N^2)$ 降到 $O(N)$。

## 4. 详细推理（Deep Dive）

```
SVM 核函数:
  linear: K(x,y) = x·y               → 线性可分
  rbf:    K(x,y) = exp(-γ||x-y||²)   → 最常用
  poly:   K(x,y) = (x·y + c)^d       → 多项式

SVM vs 深度学习:
  小数据 + 高维特征: SVM 可能更好
  大数据 + 复杂模式: 深度学习完胜
```

### 4.1 从原问题到对偶问题的完整推导 / Primal → Dual Derivation

软间隔原问题（见 3.1）含不等式约束，引入乘子 $\alpha_i \ge 0$（对应间隔约束）和 $\mu_i \ge 0$（对应 $\xi_i \ge 0$），写出拉格朗日函数：

$$L(w,b,\xi,\alpha,\mu) = \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i\bigl[y_i(w^\top x_i + b) - 1 + \xi_i\bigr] - \sum_i \mu_i \xi_i$$

对原变量求偏导并令其为零（**驻点条件**，stationarity，KKT 的一部分）：

$$\frac{\partial L}{\partial w}=0 \Rightarrow w = \sum_i \alpha_i y_i x_i; \qquad \frac{\partial L}{\partial b}=0 \Rightarrow \sum_i \alpha_i y_i = 0; \qquad \frac{\partial L}{\partial \xi_i}=0 \Rightarrow \alpha_i + \mu_i = C$$

把这三式代回 $L$，原变量与 $\xi_i$ 全部消去（注意 $\mu_i\ge0$ 与 $\alpha_i+\mu_i=C$ 合起来恰好给出 $\alpha_i \le C$），得到 3.2 的对偶目标。**为什么对偶等价于原问题**：QP 是凸的且满足 Slater 条件，强对偶成立，对偶最优 = 原问题最优。求解器（如 SMO，Sequential Minimal Optimization）每次只挑两个 $\alpha$ 解析更新，是 libsvm 的核心算法。

由偏置不能从 $w$ 直接得到，需用任一自由支持向量（$0<\alpha_i<C$，此时 $y_if(x_i)=1$）反解：$b = y_i - \sum_j \alpha_j y_j K(x_j, x_i)$。

### 4.2 Mercer 条件：什么样的函数能当核 / When is $K$ a Valid Kernel

核技巧成立的前提是存在某个 $\phi$ 使 $K(x,z)=\phi(x)^\top\phi(z)$。**Mercer 定理**给出可操作判据：$K$ 是合法核 $\iff$ 对任意有限样本集，其核矩阵 $G_{ij}=K(x_i,x_j)$ **对称半正定（PSD）**。RBF 核可证恒为 PSD（对任意 $\gamma>0$、任意数据），这是它能无脑当默认核的理论保障；而自定义相似度函数若不满足 PSD，对偶 QP 可能非凸、解不可靠。

### 4.3 复杂度标注 / Complexity Annotation

```
# 设 N=样本数, D=特征维, n_SV=支持向量数, K=类别数
# 核 SVC 训练 (SMO 求解 QP):   Time: O(N^2·D) ~ O(N^3)   Space: O(N^2)   ← 核矩阵主导
# 线性 SVC (liblinear 坐标下降): Time: O(N·D)              Space: O(N·D)
# 预测 (核):                    Time: O(n_SV·D) / 样本     Space: O(n_SV·D)
# 多分类 OvO 总训练:            ≈ K(K-1)/2 个子问题, 每个仅含两类样本 → 常比 OvR 更省
```

关键结论：训练成本对 $N$ **超线性**，对 $D$ 仅线性。所以"高维少样本"是 SVM 的最佳赛道，"海量样本"是它的禁区（见 3.7）。

## 5. 例题（Worked Examples）

### 例题 1：线性与高斯核 (RBF) 支持向量机分类对比 / Linear vs RBF Kernel SVM Comparison

支持向量机（SVM）能够通过核技巧（Kernel Trick）处理非线性边界。本例演示线性核与 RBF 高斯核在环状数据集上的分类表现对比。

```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 制造环形非线性数据集 / Generate non-linear dataset
# Time: O(N), Space: O(N)
X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# 2. 训练线性核 SVM / Train Linear SVM
# Time: O(N^2 * D) 到 O(N^3 * D) 之间 / Computational scale.
# Space: O(N)
linear_svm = SVC(kernel='linear')
linear_svm.fit(X, y)

# 3. 训练高斯核 RBF SVM / Train RBF SVM
rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X, y)

print(f"线性核精度 / Linear Kernel Accuracy: {accuracy_score(y, linear_svm.predict(X)):.4f}")
print(f"高斯核精度 / RBF Kernel Accuracy:   {accuracy_score(y, rbf_svm.predict(X)):.4f}")
```

线性核在环形数据上精度约 0.5（近乎瞎猜），RBF 核接近 1.0——直观印证了"升维 + 核技巧"对非线性边界的威力。

### 例题 2：RBF-SVM 超参网格搜索与决策边界可视化 / Grid Search & Decision Boundary Visualization

本例演示工程中的标准流程：标准化 → 联合网格搜索 $(C,\gamma)$ → 在测试集评估 → 把决策边界画出来理解模型。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

# 1. 月牙形非线性数据 / Two-moons non-linear data
# X shape: [300, 2], y shape: [300]
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 用 Pipeline 把标准化与 SVC 绑定，避免在 CV 中泄漏测试集统计量
#    / Pipeline ties scaling to SVC, preventing leakage across CV folds
pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

# 3. 联合搜索 C 与 gamma（注意 Pipeline 命名前缀 svc__）/ Joint grid over (C, gamma)
# Time: O(|grid| * cv * QP_cost), QP_cost ≈ O(N^2·D)
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_tr, y_tr)

print(f"最优参数 / Best params: {grid.best_params_}")
print(f"测试集精度 / Test accuracy: {grid.score(X_te, y_te):.4f}")

# 4. 可视化决策边界（2026 sklearn 推荐 API）/ Plot boundary with modern API
disp = DecisionBoundaryDisplay.from_estimator(
    grid.best_estimator_, X, response_method="predict",
    alpha=0.3, cmap="coolwarm", grid_resolution=200)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", s=20)
disp.ax_.set_title(f"RBF-SVM  best={grid.best_params_}")
plt.show()
```

要点：(1) 用 `Pipeline` 把标准化包进交叉验证，否则每折都会用到测试集的均值/方差，造成**数据泄漏**导致评估虚高；(2) `gamma` 过大时边界会贴着噪声点剧烈扭曲（过拟合），网格搜索能自动避开；(3) `DecisionBoundaryDisplay` 是 1.1+ 推荐的边界绘制 API，取代了手写 `np.meshgrid + contourf`。

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释 SVM 优化目标中软间隔（Soft Margin）参数 $C$ 的作用。当 $C$ 很大或很小时，模型分别倾向于什么状态？
*参考答案*：
参数 $C$ 是对误分类误差的惩罚权重：
- $C$ 很大：对误分类惩罚重，模型会倾向于减小误分类样本，使得决策边界间隔窄，容易**过拟合**。
- $C$ 很小：对误分类惩罚轻，允许更多样本误分以换取更宽的间隔，容易**欠拟合**。

### 进阶题
**练习 2**：在金融反欺诈中，类别往往极度不平衡（如 99.9% 正常，0.1% 欺诈）。在此场景下训练 SVM 模型，如何配置超参数来平衡类别权重，防止模型偏向大类？
*参考答案*：
应使用 `class_weight='balanced'` 参数，或者手动指定各类别的权重字典，使少数类的误分类惩罚权重成比例放大。
```python
# SVC(kernel='rbf', class_weight='balanced')
```