# 聚类算法
# Clustering Algorithms

## 1. 背景（Background）

> **为什么要学这个？**
>
> 聚类是无监督学习的代表。在大模型开发中，聚类用于数据去重（MinHash + 聚类）、数据多样性分析、Embedding 可视化（t-SNE + KMeans）。

## 2. 知识点（Key Concepts）

| 算法 | 特点 |
|------|------|
| KMeans | 快速，需指定 K |
| DBSCAN | 自动确定簇数，处理噪声 |
| 层次聚类 | 树状结构，无需指定 K |

## 3. 内容（Content）

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 生成聚类数据
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)

# ============================================================
# KMeans
# ============================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)
print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
print(f"聚类中心: {kmeans.cluster_centers_.shape}")

# 肘部法则选择 K
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)
# 画图找"肘部"拐点

# ============================================================
# DBSCAN（基于密度）
# ============================================================
# ⚠️ 此处硬编码 eps=0.5 仅为演示 API 调用；在这份 make_blobs 数据上它其实
#    会得到 6 簇 + 67 个噪声点（真实只有 4 簇），是反模式。
#    正确做法（数据驱动定 eps）见 3.4 节与例题 2。
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"DBSCAN 找到 {n_clusters} 个簇")
```

### 3.1 KMeans：Lloyd 迭代与初始化 / KMeans: Lloyd's Algorithm & Initialization

KMeans 优化目标是**簇内平方和**（Within-Cluster Sum of Squares, WCSS / inertia）：

$$J = \sum_{k=1}^{K}\sum_{x\in C_k}\|x-\mu_k\|^2$$

**Lloyd 迭代**交替执行两步，每步都不增大 $J$：

1. **分配步（Assignment）**：固定质心，把每个点归到最近质心 → 给定 $\mu$ 时这是 $J$ 的最优分配。
2. **更新步（Update）**：固定分配，把质心移到簇内均值 $\mu_k = \frac{1}{|C_k|}\sum_{x\in C_k}x$ → 给定分配时均值是使簇内平方和最小的点。

两步都单调下降且状态有限，故必收敛，但只保证**局部最优**——初值差会陷入坏解。

**k-means++ 初始化**（sklearn 默认 `init='k-means++'`）：先随机选第一个中心，之后每个新中心以正比于 $D(x)^2$（到已选最近中心距离的平方）的概率被选中，让初始中心尽量散开。它把期望逼近比从普通随机的无界改善到 $O(\log K)$，显著减少坏局部最优。

```python
# 2026 sklearn 1.4+：n_init 默认值已改为 'auto'
#   配 k-means++ 时 'auto' = 1（一次足够好），配随机初始化时 = 10
#   不传参时默认值已由 10 变为 'auto'；显式写死 n_init=10 会保持旧行为
#   （k-means++ 下比 'auto' 多跑 9 次），新代码统一用 'auto'
kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', random_state=42)
```

### 3.2 选 K：肘部法 vs 轮廓系数 / Choosing K: Elbow vs Silhouette

KMeans 必须预设 $K$，两种数据驱动方法：

| 方法 | 指标 | 判读 | 局限 |
|------|------|------|------|
| 肘部法 Elbow | inertia 随 $K$ 单调降 | 找下降由陡变缓的"拐点" | 拐点常模糊、主观 |
| 轮廓系数 Silhouette | $s=\frac{b-a}{\max(a,b)}\in[-1,1]$ | 取 $s$ **最大**的 $K$ | $O(N^2)$，大数据慢 |

轮廓系数对单点：$a$ 是它到**同簇**其他点的平均距离，$b$ 是到**最近邻簇**所有点的平均距离。$s\to1$ 说明簇内紧、簇间远；$s<0$ 说明该点很可能被分错。相比肘部法的主观拐点，轮廓系数给出可比较的标量，工程上更可靠。

### 3.3 层次聚类 / Hierarchical (Agglomerative) Clustering

自底向上：每点先成一簇，反复合并最近的两簇，得到一棵**树状图（dendrogram）**；在任意高度横切即得对应簇数，**无需预设 $K$**。簇间距离由 `linkage` 定义：

| linkage | 簇间距离 | 特点 |
|---------|----------|------|
| `ward`（默认） | 合并后 inertia 增量最小 | 倾向等体积球状簇，最常用 |
| `complete` | 两簇最远点对距离 | 紧凑、抗链式效应 |
| `single` | 两簇最近点对距离 | 能抓非凸形状，但易"链式"串簇 |
| `average` | 所有跨簇点对平均 | 折中 |

```python
from sklearn.cluster import AgglomerativeClustering
# 指定簇数横切；改 distance_threshold 可改为按距离阈值横切
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg = agg.fit_predict(X)
```

代价是 $O(N^2)$ 空间、$O(N^2\log N)$ 时间，只适合中小数据；优点是产出层次结构，便于探索性分析。

### 3.4 DBSCAN：密度可达与调参 / DBSCAN: Density-Reachability & Tuning

DBSCAN 用两个参数刻画"密度"：`eps`（邻域半径）与 `min_samples`（成为核心点所需的邻域内最少点数）。

- **核心点（core）**：`eps` 邻域内点数 ≥ `min_samples`。
- **密度直达**：$q$ 在核心点 $p$ 的 `eps` 邻域内，则 $p\to q$ 直达。
- **密度可达 / 相连**：直达关系的传递闭包；一条密度相连链上的点构成一个簇。
- **噪声点**：不属于任何簇，标签为 **-1**——这是 DBSCAN 相对 KMeans 的杀手锏（KMeans 强行把噪声塞进簇）。

它能识别**任意形状**簇（环形、月牙），无需预设簇数。但 `eps` 极敏感：

> ⚠️ **在 `make_blobs` 上必须数据驱动调 `eps`**。`make_blobs` 的坐标尺度随 `cluster_std`、`center_box` 变化，硬编码 `eps=0.5` 在某些尺度下会把所有点连成一坨或全判为噪声。标准做法用 **k-距离图**：对每点取第 `k=min_samples` 近邻的距离，升序排序画出，曲线的"膝点"即合理 `eps`（见例题 2）。也可先 `StandardScaler` 标准化再选 `eps`。

```python
# min_samples 经验值：≥ 维度 D + 1，常取 2*D；噪声多则调大
dbscan = DBSCAN(eps=0.8, min_samples=2 * X.shape[1])
```

### 3.5 高斯混合 GMM 与 EM / Gaussian Mixture & EM

GMM 假设数据由 $K$ 个高斯分布**按权重 $\pi_k$ 混合**生成：$p(x)=\sum_k \pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k)$。与 KMeans 的硬分配不同，GMM 给出**软分配**——每点属于各簇的后验概率（responsibility）$\gamma_{ik}$，且通过协方差矩阵 $\Sigma_k$ 支持**椭球形、不同朝向/体积**的簇。

用 **EM 算法**最大化对数似然，交替两步：

- **E 步**：固定参数，算责任度 $\gamma_{ik}=\dfrac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$（软归属）。
- **M 步**：固定责任度，用加权样本更新 $\pi_k,\mu_k,\Sigma_k$。

可证 KMeans 是 GMM 在"各向同性、等协方差、硬分配（$\gamma$ 退化为 0/1）"下的特例。

```python
from sklearn.mixture import GaussianMixture
# covariance_type: 'full'(各簇任意椭球) / 'tied' / 'diag' / 'spherical'(退化近 KMeans)
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X)
proba = gmm.predict_proba(X)   # shape: [N, K]，软分配概率
# 用 BIC 选 K：在多个 K 上取 BIC 最小者 / Select K by minimizing BIC
print(f"BIC: {gmm.bic(X):.2f}")
```

### 3.6 算法选型对比 / Algorithm Comparison

| 算法 | 需预设 K | 簇形状 | 抗噪声 | 软分配 | 复杂度 | 适用场景 |
|------|:-------:|--------|:------:|:------:|--------|----------|
| KMeans | 是 | 凸/球状 | 差 | 否 | $O(N\!K\!D\!I)$ | 大数据、球状簇、要快 |
| 层次聚类 | 否* | 取决 linkage | 差 | 否 | $O(N^2\log N)$ | 中小数据、需层次结构 |
| DBSCAN | 否 | **任意** | **强** | 否 | $O(N\log N)$** | 任意形状、有噪声、密度均匀 |
| GMM | 是 | **椭球** | 中 | **是** | $O(N\!K\!D^2\!I)$ | 椭球簇、要概率/软分配 |

\* 层次聚类需选横切高度（等价于选 K）；\*\* DBSCAN 配空间索引为 $O(N\log N)$，最坏 $O(N^2)$。
口诀：**要快且球状 → KMeans；任意形状带噪声 → DBSCAN；要概率/椭球 → GMM；要层次探索 → 层次聚类**。

## 4. 详细推理（Deep Dive）

```
聚类在 LLM 中的应用:
  - 训练数据去重: 对 embedding 聚类，同簇视为重复
  - 数据配比: 不同簇代表不同领域，确保训练数据多样
  - 评估: 对模型输出 embedding 可视化（t-SNE + 颜色标注）
```

### 4.1 KMeans 为何必然收敛 / Why KMeans Converges

把目标 $J=\sum_k\sum_{x\in C_k}\|x-\mu_k\|^2$ 看成关于"分配 $z$"与"质心 $\mu$"两组变量的函数，Lloyd 迭代是**坐标下降（block coordinate descent）**：

- **固定 $\mu$ 优化 $z$**（分配步）：每点独立取最近质心，是该子问题的全局最优 → $J$ 不增。
- **固定 $z$ 优化 $\mu$**（更新步）：对 $\mu_k$ 求导 $\partial J/\partial\mu_k = -2\sum_{x\in C_k}(x-\mu_k)=0$，解得 $\mu_k$ = 簇均值，是该子问题的全局最优 → $J$ 不增。

每轮 $J$ 单调非增且有下界 0，而可能的分配只有有限种，故有限步内收敛。但 $J$ **非凸**，收敛点依赖初值——这正是需要 k-means++ 与多次 `n_init` 的根本原因。

### 4.2 EM 推导直觉：为什么 E/M 交替能提升似然 / EM Intuition

直接最大化 GMM 的对数似然 $\log\sum_k\pi_k\mathcal{N}(x\mid\theta_k)$ 因"log 套 sum"无闭式解。EM 的技巧：引入隐变量 $z$（点属于哪个高斯），用 Jensen 不等式对似然构造一个**下界 $Q$（ELBO）**，再交替"贴紧下界"和"抬高下界"：

- **E 步**：用当前参数算后验 $\gamma_{ik}=p(z=k\mid x_i)$，使下界在当前参数处与真实似然**相切**（差值即 KL 散度被消为 0）。
- **M 步**：对这个下界（变成对完整数据似然的加权求和，有闭式解）最大化，更新参数。

因为下界处处 ≤ 似然且在当前点相切，最大化下界必然使**真实似然单调不减**。与 KMeans 同样只保证局部最优、对初值敏感（sklearn 用 k-means++ 初始化 GMM）。

### 4.3 复杂度标注 / Complexity Annotation

```
# N=样本数, K=簇数, D=维度, I=迭代轮数
# KMeans (Lloyd):        Time: O(N·K·D·I)        Space: O(N·D + K·D)   ← 大数据首选
# 轮廓系数 silhouette:    Time: O(N^2·D)          Space: O(N^2)         ← 大数据须采样
# 层次聚类 (ward):        Time: O(N^2·log N)      Space: O(N^2)         ← N 上万即吃力
# DBSCAN (带 KD-Tree):    Time: O(N·log N)        Space: O(N)           ← 最坏 O(N^2)
# GMM-EM (full 协方差):   Time: O(N·K·D^2·I)      Space: O(N·K + K·D^2) ← D 大时 Σ 求逆贵
```

关键取舍：KMeans/DBSCAN 对 $N$ 近线性、可扩展；层次聚类与轮廓系数因 $O(N^2)$ 只适合中小数据；GMM 的 `full` 协方差对维度 $D$ 是平方级（协方差求逆），高维时改用 `diag`。

### 4.4 与 Java 实现思路的对照 / Note on Java Implementation

KMeans 的分配步是天然数据并行的：N 个点到 K 个质心的距离计算互不依赖，Java 中可用 `parallelStream()` 或 `ThreadPoolExecutor`（按 P3C 显式构造，禁用 `Executors` 工厂以防 OOM）切分点集并行算 argmin，更新步再做一次 reduce 求均值——这正是 Spark MLlib `KMeans` 的 map-reduce 思路。sklearn 内部则用 Cython + BLAS 把分配步向量化，对应"绝不在张量上写显式循环"的原则。

## 5. 例题（Worked Examples）

### 例题 1：利用肘部法 (Elbow Method) 确定 K-Means 聚类的最佳簇数 / Selecting optimal K via Elbow Method

K-Means 算法需要预先设定聚类簇数 $K$。本例题演示如何计算不同 $K$ 值下的手肘指标（SSE），找出拐点。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. 制造高斯聚类数据集 / Generate dataset
# Time: O(N * D), Space: O(N * D)
X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)

# 2. 循环计算不同 K 值的样本平方距离和 (Inertia) / Calculate inertia for each K
# Time: O(K * Iterations * N * C), Space: O(N * D)
inertias = []
k_range = range(1, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 打印 SSE 折线变化 / Print SSE results
for k, sse in zip(k_range, inertias):
    print(f"K={k}, SSE (Inertia) = {sse:.4f}")
```

SSE 在 K=4 后下降明显放缓，拐点对应真实簇数 4。配合 `silhouette_score` 在 K=4 取最大值可交叉确认。

### 例题 2：用 k-距离图数据驱动地确定 DBSCAN 的 eps / Data-Driven `eps` via k-Distance Graph

承接 3.4 的警示：在 `make_blobs` 上**绝不能硬编码 `eps`**。本例先用 k-距离图定出 `eps`，再聚类，并用轮廓系数评估、与 KMeans 对比。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# 1. 数据（不标准化，保留原始尺度以凸显 eps 的尺度依赖）
# X shape: [500, 2]
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# 2. k-距离图：取第 k=min_samples 近邻的距离并升序排序，"膝点"即合理 eps
#    / k-distance graph: sort each point's distance to its k-th neighbor
# Time: O(N·log N) with KD-Tree, Space: O(N)
min_samples = 2 * X.shape[1]                       # 经验值 = 2*D
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
dists, _ = nbrs.kneighbors(X)                       # dists shape: [N, min_samples]
k_dist = np.sort(dists[:, -1])                       # 每点到第 k 近邻的距离，升序

plt.plot(k_dist)
plt.xlabel("points sorted"); plt.ylabel(f"{min_samples}-th NN distance")
plt.title("k-distance graph: pick eps at the knee")
plt.show()

# 3. 从膝点读出 eps（这里用 90% 分位数做稳健的自动估计）
#    / Read eps from the knee (90th percentile as a robust auto-estimate)
eps = np.quantile(k_dist, 0.90)
print(f"数据驱动的 eps / Data-driven eps = {eps:.3f}")

# 4. 聚类与评估 / Cluster & evaluate
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print(f"簇数 / clusters = {n_clusters}, 噪声点 / noise = {n_noise}")

# 轮廓系数仅在非噪声点上计算（噪声标签 -1 不是真正的簇）
# Silhouette computed on non-noise points only
mask = labels != -1
if n_clusters >= 2 and mask.sum() > n_clusters:
    print(f"轮廓系数 / Silhouette = {silhouette_score(X[mask], labels[mask]):.4f}")
```

要点：(1) 把 `eps=0.5` 写死在不同 `cluster_std` 下会全连一簇或全判噪声，k-距离图让 `eps` 跟着数据尺度走；(2) 评估 DBSCAN 的轮廓系数要**剔除噪声点**（标签 -1），否则把噪声当簇会拉低指标；(3) `make_blobs` 是凸球状簇，这种场景 KMeans 往往更稳更快，DBSCAN 的优势要在环形/月牙等非凸数据上才显现——选算法永远先看数据形状。

## 6. 习题（Exercises）

### 基础题
**练习 1**：解释 K-Means 聚类与 DBSCAN 聚类算法在算法原理和“簇形状”支持上的核心差异。
*参考答案*：
- **K-Means**：基于距离和质心分配，默认假设簇为凸集（球状分布），需要预设定 $K$。
- **DBSCAN**：基于核心点及密度可达性，不需要设定 $K$，可以识别任意形状（如环状、星形）的簇，并可自动过滤噪声点。

### 进阶题
**练习 2**：K-Means 容易陷入局部最优。在实际工程中，如何通过参数初始化优化这一缺陷？请说明 `init='k-means++'` 参数的工作机制。
*参考答案*：
使用 `init='k-means++'`。机制是：首先随机挑选第一个聚类中心，接着计算其他样本点到已选聚类中心的最短距离，以正比于这个距离平方的概率去挑选下一个聚类中心。这确保了初始聚类中心彼此相距尽可能远，能大大加快收敛并避免陷入次优局部解。