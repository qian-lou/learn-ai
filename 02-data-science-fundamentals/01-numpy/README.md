# 01-numpy — NumPy 数值计算

> **所属阶段**：阶段二 · 数据科学基础
> **学习目标**：掌握 NumPy 高性能数值计算，理解向量化编程思想，为 PyTorch 张量操作打底
> **预估时长**：4-5 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [ndarray-basics](./01-ndarray-basics.md) | ndarray 基础与创建 | `array/zeros/ones/arange/linspace` 创建；`shape/dtype/ndim/size` 属性；`reshape(-1,…)` 变形；向量化替代 for 循环 |
| 02 | [indexing-and-slicing](./02-indexing-and-slicing.md) | 索引与切片 | 基础索引、切片、布尔索引 `a[a>0]`、花式索引 `a[[0,2]]`、`np.where`；视图 vs 副本陷阱 |
| 03 | [broadcasting](./03-broadcasting.md) | 广播机制 | 从右向左对齐的三条广播规则、`keepdims`、`np.newaxis` 升维、Attention Mask 广播 |
| 04 | [linear-algebra-ops](./04-linear-algebra-ops.md) | 线性代数运算 | `@` 矩阵乘、`inv/det/solve`、特征分解、SVD（LoRA 基础）、`einsum`、`norm` |
| 05 | [performance-optimization](./05-performance-optimization.md) | 性能优化 | 向量化 vs 循环（100-1000x）、`np.where/maximum`、轴向聚合、`out=` 原地运算避免临时数组 |
| 06 | [numerical-stability](./06-numerical-stability.md) | 数值精度与稳定性 | `float16/bfloat16` 位宽与动态范围、softmax 减最大值防 `nan`、`logsumexp`、`np.logaddexp` |

---

## 🔑 知识点详解

### 01 · ndarray 基础与创建
- **核心概念**：`ndarray` 是**同构、定长、内存连续**的多维数组，等价于 PyTorch 的 `Tensor`（90% API 一致）。
- **关键 API**：`np.array / np.zeros((r,c)) / np.arange(s,e,step) / np.linspace(s,e,n)`；变形用 `a.reshape(-1, 4)`（`-1` 自动推断该维度）。
- **易错点**：① `reshape` 前后元素总数必须相等，否则报错；② 默认整型是 `int64`、浮点是 `float64`，训练场景通常要显式指定 `dtype=np.float32`（内存减半）。
- **Java 视角**：相当于 `Apache Commons Math + Stream API + SIMD` 的合体——Java 的 `int[][]` 是行指针数组（内存分散），而 ndarray 是单块连续内存（缓存友好）。
- **前置**：无（本模块起点）。

### 02 · 索引与切片
- **核心概念**：一套语法覆盖标量取值、切片、条件筛选，是数据预处理与 Mask 操作的基础。
- **关键 API**：布尔索引 `a[mask]`、条件替换 `a[a<0]=0`（向量化 ReLU）、三元向量化 `np.where(cond, x, y)`。
- **易错点**：① **切片返回视图**（共享内存），改视图会污染原数组，需要独立数据时显式 `.copy()`；② **布尔/花式索引返回副本**，对副本赋值不影响原数组——两者行为相反，最易踩坑。
- **Java 视角**：视图 ≈ `subList()` 返回的可回写视图；副本 ≈ `new ArrayList<>(...)`。误判视图/副本 = Java 里误改共享引用导致的隐蔽 bug。
- **前置**：01（ndarray 与 shape）。

### 03 · 广播机制
- **核心概念**：不同 shape 的数组无需手动扩展即可逐元素运算——理解广播是读懂大模型代码的门槛。
- **关键规则**：**从右向左对齐**——① 维数不足在左侧补 1；② 对应维相等则兼容；③ 某维为 1 则被拉伸；④ 都不为 1 且不等则报错。升维用 `a[:, np.newaxis, :]`。
- **易错点**：① 沿轴聚合后需 `keepdims=True` 才能广播回原 shape（`mean(axis=1)` 得 `[100]` 无法与 `[100,50]` 广播，`keepdims=True` 得 `[100,1]` 才行）；② `[3]+[4]`、`[2,3]+[3,2]` 这类都不为 1 又不相等会直接报错。
- **Java 视角**：广播是"隐式循环"，把手写双层 for 交给底层 C 完成——你只需描述形状，无需写循环体。
- **前置**：01、02。

### 04 · 线性代数运算
- **核心概念**：矩阵乘法、SVD、特征分解是深度学习的数学骨架；`einsum` 是描述任意张量运算的通用语言。
- **关键 API**：`A @ B`（矩阵乘，等价 `np.matmul`）、`np.linalg.solve(A,b)`（解方程优于先求逆再乘）、`np.linalg.svd`、`np.linalg.eigh`（对称矩阵专用，比 `eig` 快且返回实数）、`np.einsum('ij,jk->ik', A, B)`。
- **易错点**：① `A @ B`（矩阵乘）与 `A * B`（逐元素乘）完全不同，混用是高频 bug；② 解线性方程组用 `solve` 而非 `inv(A) @ b`——后者数值不稳定且更慢；③ `@` 要求内侧维度对齐 `(m,k)@(k,n)`。
- **Java 视角**：`einsum` 之于张量运算，类似 SQL 之于数据查询——用声明式下标表达式（`'bhsd,bhtd->bhst'` 就是 Attention 的 QKᵀ）替代命令式循环。
- **前置**：01、03；数学上依赖阶段〇/线代基础（矩阵乘、特征值）。

### 05 · 性能优化（向量化 vs 循环）
- **核心概念**：**永远不要对 ndarray 写逐元素 for 循环**——向量化把循环下沉到 C/BLAS，快 100-1000 倍，是 Java 转 Python 最关键的思维转变。
- **关键 API**：`np.where/np.maximum`（条件与 ReLU）、`arr.sum(axis=…) / mean / std / argmax`（轴向聚合）、`np.multiply(a, a, out=buf)`（`out=` 复用缓冲区，避免临时数组）。
- **易错点**：① `sum(a.tolist())` 用了 Python 内建 `sum`，比 `np.sum(a)` 慢百倍；② `a**2 + b**2` 会产生多个临时数组，大数组场景内存翻几倍，用 `out=` 原地算；③ 记不清 `axis`：`axis=0` 沿行方向压缩（得到每列结果），`axis=1` 沿列方向压缩（得到每行结果）。
- **Java 视角**：向量化 ≈ 用 SIMD/并行流一次处理一批，而不是 `for(i…) a[i]+b[i]`；思考"维度和形状"而非"循环和索引"。
- **前置**：01-04（是前四点的综合运用）。

### 06 · 数值精度与稳定性
- **核心概念**：会向量化不等于写对——照直写的 `np.exp(x)/np.exp(x).sum()` 在真实 logits 上会溢出成 `nan`；数值稳定性是 softmax/交叉熵/Attention 能跑起来的地基。
- **关键 API**：softmax 减最大值 `np.exp(x - x.max())`、`logsumexp = m + np.log(np.exp(x-m).sum())`、成对稳定加法 `np.logaddexp(a, b)`、位宽查询 `np.finfo(np.float16).max`。
- **易错点**：① `exp(大 logits)` 溢出 `inf`，`inf/inf` 得 `nan`——必须先减最大值（分子分母同乘常数，结果不变）；② 沿轴 softmax 时减最大值要配 `keepdims=True` 才能广播；③ 训练用 `bfloat16` 而非 `float16`——前者指数 8 位、动态范围同 `float32`（不易溢出），后者指数仅 5 位、上限约 65504。
- **Java 视角**：类似 `int` 溢出回绕——浮点溢出不报错而是无声变 `inf/nan`，比 Java 更隐蔽，必须在算法层面（减最大值）预防而非事后 `catch`。
- **前置**：01-05；数学上依赖 softmax/log 恒等式。

---

## 🎯 学习要点

- **向量化是铁律**：任何对 ndarray 的逐元素 for 循环都应改写成数组运算或 ufunc；写完自查"有没有循环还能消掉"。
- **吃透广播**：能手推任意两个 shape 能否广播、结果 shape 是多少；这是后续 PyTorch、Attention、BatchNorm 的直接前置。
- **区分视图与副本**：切片/`reshape`/`.T` 是视图，布尔/花式索引/`.copy()` 是副本——修改前先判断会不会污染原数组。
- **`axis` 与 `keepdims` 成对练**：每次聚合都问自己"沿哪个轴、结果 shape 是什么、要不要 `keepdims` 广播回去"。
- **线代记 `@`/`solve`/`svd`/`einsum` 四件套**：`@` 做前向传播，`solve` 解方程，`svd` 理解 LoRA，`einsum` 读懂大模型代码。
- **dtype 意识**：训练默认 `float32`，索引/标签用 `int64`，Mask 用 `bool_`；大数组优先考虑降精度省内存。

---

## 🔗 关联

- **上一模块**：[阶段一 · 01-python-basics](../../01-python-basics/README.md)（Python 语法与环境）
- **下一模块**：[02-pandas](../02-pandas/README.md)（Pandas 底层正是 ndarray）
- **本阶段总览**：[阶段二 · 数据科学基础](../README.md)
- **Agent 课程衔接**：向量化与线代直接支撑 [agent-course · Day 16 Embedding 原理与余弦相似度](../../agent-course/Day-16-embedding-basics.md)（相似度计算就是 `einsum`/矩阵乘 + `norm`）。
