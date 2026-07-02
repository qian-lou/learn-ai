# 高效/长上下文注意力架构 / Efficient Attention Architectures

## 1. 背景（Background）

> **为什么要学这个？**
>
> 上一节（01-self-attention）我们算过一笔账：注意力矩阵是 `N×N`，`N=128K` 时高达 **164 亿**个元素，显存直接爆炸。那节结尾抛出一句「这就是为什么需要 FlashAttention、稀疏注意力等优化」——本节就来兑现「稀疏注意力」这半句，把 O(N²) **主线**补完整。
>
> 关键要先分清两条不同的路：
> - **改算法**（本节）：动手改注意力的**连接结构**——每个 token 不再看全部 N 个 token，只看一个子集，于是矩阵天生就稀疏、少算了。属于**架构层**优化，代价是**近似**（可能损失一点质量）。
> - **改访存**（FlashAttention，阶段八 `02-inference-acceleration`）：算法一字不改，softmax(QKᵀ)V 该算的照算、结果**精确**，只是用分块 + 不落地中间矩阵的方式重排 GPU 内存访问。属于**内核层**优化。
>
> 对于 Java 工程师来说，这就像优化一个 O(N²) 的全连接广播：**改算法**是把全连接图剪成「只连邻居 + 少数枢纽节点」的稀疏图；**改访存**是图不动，只把嵌套循环改成缓存友好的分块遍历（类似矩阵乘法的 tiling）。两者正交，真实系统里常叠加使用。
>
> **在整个体系中的位置：** 本节是 O(N²) 主线中「稀疏/滑窗/线性注意力」这一族的总纲，也解释了 `02-pretrained-models/03-t5-and-others.md` 里 Mistral「Sliding Window Attention」到底是什么。

## 2. 知识点（Key Concepts）

| 方案 | 代表模型 | 连接结构 | 时间复杂度 | 精确/近似 |
|------|---------|---------|-----------|----------|
| 滑动窗口 / 局部 | Mistral (SWA) | 每个 token 只看左右 `w` 个邻居 | O(N·w) | 近似（截断远距离） |
| 稀疏 + 全局 token | Longformer / BigBird | 局部窗口 + 少量全局枢纽 token | O(N·w) | 近似 |
| 线性 / 低秩 | Linformer / Performer | 重排结合律，去掉 N×N | O(N) | 近似（低秩/核近似） |
| FlashAttention | —（内核，非架构） | 不改连接，全连接照旧 | O(N²) 计算 / O(N) 访存 | **精确** |

**一句话公式对比：**
```
标准注意力:   softmax(QKᵀ/√d)·V              → N×N 稠密矩阵, O(N²·d)
滑动窗口:     只在 |i−j| ≤ w 的带状区域算       → O(N·w·d)
线性注意力:   φ(Q)·(φ(K)ᵀ·V)  （先算右边 d×d）  → O(N·d²)
```

> 核心洞察：O(N²) 来自「每个 token × 每个 token」。要降它，要么**减少每个 token 看的对象**（稀疏/滑窗），要么**用结合律绕开 N×N 这个中间量**（线性）。

## 3. 内容（Content）

### 3.1 滑动窗口注意力（Sliding Window Attention, SWA）

直觉：语言里绝大多数依赖是**局部**的——第 1000 个词和第 3 个词直接相关的情况很少。那就规定每个 token 只关注它左边（因果场景）`w` 个邻居，注意力矩阵从满的 `N×N` 退化成一条**带状**（band）区域。

```python
import torch

def sliding_window_mask(seq_len: int, window: int) -> torch.Tensor:
    """构造因果滑动窗口掩码：位置 i 只能看 [i-window+1, i] / Causal SWA mask.

    Args:
        seq_len: 序列长度 N / Sequence length.
        window: 窗口大小 w（含自身）/ Window size (inclusive of self).

    Returns:
        布尔掩码，True=可见 / Bool mask, True means attend. Shape: [N, N]
    """
    # 时间 O(N²) 空间 O(N²)（仅构造掩码；真实核只算带内 O(N·w)）
    idx = torch.arange(seq_len)                       # Shape: [N]
    row = idx.unsqueeze(1)                            # Shape: [N, 1]
    col = idx.unsqueeze(0)                            # Shape: [1, N]
    causal = col <= row                               # 下三角：不看未来
    within = col > (row - window)                     # 只看最近 window 个
    return causal & within                            # Shape: [N, N]

# 验证：N=6, w=3，位置 3 应能看到列 {1,2,3}
mask = sliding_window_mask(6, 3)
print(mask.int())
# tensor([[1, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0],
#         [0, 1, 1, 1, 0, 0],   ← 位置 3：列 0 被窗口截断，只看 1/2/3
#         [0, 0, 1, 1, 1, 0],
#         [0, 0, 0, 1, 1, 1]])
```

把这个掩码接到 01 节 `SelfAttention` 的 `scores.masked_fill(mask == 0, -inf)` 上，注意力就限制在带内。**复杂度直觉**：每行只有 `w` 个非 `-inf`，有效计算量是 `O(N·w·d)`；`w` 是常数（Mistral 取 4096），于是对 N 线性。

**代价与补偿——感受野靠层数叠加**：单层只能看 `w` 个邻居，但和 CNN 一样，堆 `L` 层后信息可跨层传播，等效感受野约 `L·w`。Mistral-7B：`w=4096`、`L=32` → 理论感受野 `≈131072 ≈ 128K`，这正是它宣传的长上下文来源。远距离信息不是「直接看到」，而是「逐层接力传过来」，这就是近似的代价。

### 3.2 稀疏 + 全局 token（Longformer / BigBird）

纯滑窗有个硬伤：分类任务里 `[CLS]` 这种要**汇总全局**的 token，只看邻居就废了；某些关键词（问题里的实体）也需要被所有位置看到。Longformer/BigBird 的补丁是给少数 token 开「全局通道」：

```
连接结构 = 局部窗口注意力  +  少量全局 token（全看 & 被全看）  (+ BigBird 再加随机连接)

           普通 token j        全局 token g
普通 i    只在 |i−j|≤w 时连    总是连 g（能看到枢纽）
全局 g    连所有 i             连所有（枢纽看全局）

矩阵形态：一条带状对角线 + 几行几列「十字」全黑
```

复杂度：设全局 token 数为 `g`（常数，如 8~64），代价是 `O(N·w) + O(N·g) = O(N·w)`——仍对 N 线性。BigBird 额外加少量随机边，用图论里「随机图直径小」的性质保证任意两 token 间有短路径，理论上能逼近全连接注意力的表达力，同样保持线性。

> **Java 类比**：这就是「局部消息总线 + 几个全局广播节点」。大部分节点只和邻居通信（省带宽），少数枢纽节点（如注册中心）与所有节点互联，保证全局信息一跳可达。

### 3.3 线性 / 低秩注意力（Linformer / Performer）

前两种是「让矩阵变稀疏」，线性注意力换了个思路：**用矩阵乘法结合律，根本不生成 N×N 那个中间量**。

标准注意力必须先算 `QKᵀ`（这就是 `N×N`），因为 softmax 是非线性的、卡在中间拆不开。若把 softmax 替换成一个可分解的核 `φ`，使 `softmax(QKᵀ)V ≈ φ(Q)·(φ(K)ᵀ·V)`，就能先算右边：

```
标准:    (Q Kᵀ) V     形状 (N×d)(d×N)(N×d)，中间 N×N     → O(N²·d)
线性:    Q (Kᵀ V)     先算 KᵀV 得 d×d，再左乘 Q          → O(N·d²)
                       ↑ 结合律换括号位置，N×N 消失了
```

`d`（特征维）是常数，于是对 N **线性**。

- **Performer**：用随机特征映射 `φ` **无偏近似** softmax 核（FAVOR+ 算法），数学上有误差界。
- **Linformer**：观察到注意力矩阵近似**低秩**，先用投影把 K、V 的长度维从 `N` 压到常数 `k`（如 256），`QKᵀ` 变成 `N×k`，也是 O(N)。

**代价**：softmax 的尖锐、精确对齐能力被核近似/低秩投影**磨平**了。经验上线性注意力在超长序列上省得最狠，但在需要精确检索（如「大海捞针」式定位某个远处 token）的任务上质量掉得明显——这就是为什么主流大模型至今仍以「稀疏/滑窗 + 精确注意力」为主，纯线性方案更多用于特定长序列场景。

## 4. 详细推理（Deep Dive）

### 4.1 与 FlashAttention 划界（兑现「O(N²) 主线」闭环）

这是本节最容易混淆、也最该讲清的一点：

```
            改的是什么         结果精确?   矩阵是否稠密?   复杂度           所属层/章节
─────────────────────────────────────────────────────────────────────────────────────
稀疏/滑窗    连接结构(算法)     近似        稀疏(带状)     O(N·w) 计算       架构层 · 本节
线性注意力   softmax→核(算法)   近似        无 N×N         O(N·d²) 计算      架构层 · 本节
FlashAttn    内存访问(内核)     精确        仍稠密         O(N²) 计算/O(N)访存 内核层 · 阶段八 02
```

一句话记牢：**稀疏注意力「少算了」（换来近似），FlashAttention「没少算、只是算得聪明」（保持精确）。** 前者改的是「谁连谁」，后者改的是「怎么把该算的搬进 SRAM 算」。

为什么 FlashAttention 计算量仍是 O(N²)？因为它一个 token 对都没跳过——它把 N×N 分成小块（tile），逐块在片上 SRAM 里算完 softmax 再累加，**从不把完整的 N×N 矩阵写回显存**，于是显存占用（HBM 访问量）从 O(N²) 降到 O(N)，但乘加次数一分不少、结果与朴素实现逐位相等。

**二者正交、可叠加**：真实系统里，Mistral 既用 SWA（稀疏、架构层）压掉大部分 token 对，又在剩下的带状注意力里用 FlashAttention 内核（访存优化）跑得更快。一个减少「要算的对数」，一个加速「每对怎么算」，互不冲突。

### 4.2 到底该选哪种？一个决策直觉

```
需求                                  推荐
────────────────────────────────────────────────────────────
序列不算长(N≤8K)、要精确、有好 GPU   标准注意力 + FlashAttention（不必近似）
超长上下文、依赖偏局部(代码/长文)     滑动窗口(SWA) + FlashAttention
要全局汇总(长文档分类/QA)             局部窗口 + 全局 token(Longformer/BigBird)
序列极长、能容忍近似、检索需求弱      线性/低秩(Performer/Linformer)
```

黄金法则：**能不近似就别近似**。近似是拿质量换显存/速度，只在标准注意力真的装不下（N 太大）时才动用，且优先用「局部性最符合数据」的那种（语言天然局部 → 滑窗最常用）。

## 5. 例题（Worked Examples）

### 例题：数一数滑动窗口到底省了多少

```python
import torch

def count_attention_pairs(seq_len: int, window: int) -> tuple[int, int]:
    """统计标准因果注意力 vs 滑动窗口的有效 token 对数量.

    Args:
        seq_len: 序列长度 N / Sequence length.
        window: 滑动窗口 w / Sliding window size.

    Returns:
        (标准对数, 滑窗对数) / (dense pairs, windowed pairs).
    """
    # 时间 O(1) 空间 O(1)（用等差数列/带宽公式直接算，不建矩阵）
    dense = seq_len * (seq_len + 1) // 2              # 因果下三角：N(N+1)/2
    # 滑窗：前 w 行是完整三角，其余每行恰好 w 个
    if seq_len <= window:
        windowed = dense
    else:
        head = window * (window + 1) // 2             # 前 window 行的三角
        windowed = head + (seq_len - window) * window # 剩余行每行 window 个
    return dense, windowed

for n, w in [(4096, 4096), (32768, 4096), (131072, 4096)]:
    d, s = count_attention_pairs(n, w)
    print(f"N={n:>6}, w={w}: 标准={d/1e6:8.1f}M 对, 滑窗={s/1e6:7.1f}M 对, "
          f"省 {(1 - s/d) * 100:4.1f}%")
# N=  4096, w=4096: 标准=     8.4M 对, 滑窗=    8.4M 对, 省  0.0%   ← N≤w，无差别
# N= 32768, w=4096: 标准=   536.9M 对, 滑窗=  125.8M 对, 省 76.6%
# N=131072, w=4096: 标准= 8590.0M 对, 滑窗=  528.5M 对, 省 93.8%   ← N 越大省越狠
```

解读：`N=w` 时滑窗退化成标准注意力（没截断，不省）；`N` 相对 `w` 越大，标准注意力按 `N²` 涨、滑窗按 `N·w` 线性涨，节省比例趋近 `1 − w/N`。`N=128K` 时省掉 **93.8%** 的计算——这就是长上下文能跑起来的直接原因。

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 3.1 的 `sliding_window_mask` 生成 `N=8, w=3` 的掩码，手工核对第 5 行（i=5）可见的列集合，并解释为什么第一行永远只有 1 个可见位置。

> **答案：** i=5 可见列为 `{3,4,5}`（`i-w+1=3` 到 `i`）。第一行（i=0）只能看自己，因为因果掩码禁止看未来、窗口内也没有更早的 token。

**练习 2：** 在 01 节 `SelfAttention.forward` 里把因果掩码换成滑动窗口掩码（`sliding_window_mask(N, w)` 再 `unsqueeze` 对齐维度），跑一遍确认输出形状仍是 `[B, N, D]`，并观察远距离 token 的注意力权重是否被置 0。

### 进阶题

**练习 3：** 用结合律验证线性注意力的等价性：对随机 `Q, K, V ∈ ℝ^{N×d}`（**不加 softmax**），分别按 `(Q Kᵀ) V` 和 `Q (Kᵀ V)` 计算，确认结果在数值误差内相等，并比较两种顺序的 FLOPs。

> **提示：** 无 softmax 时结合律严格成立；加了 softmax 才需要 Performer 那样的核近似。

*参考答案*：
```python
import torch

N, d = 2048, 64
Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
left = (Q @ K.T) @ V            # 中间 N×N，O(N²·d)
right = Q @ (K.T @ V)          # 中间 d×d，O(N·d²)
print(torch.allclose(left, right, atol=1e-3))  # True：结合律严格成立
# FLOPs 比 ≈ N/d = 2048/64 = 32×，右式在 N≫d 时大幅更省
```
要点：正是 softmax 这个非线性「胶水」卡在 `QKᵀ` 中间，才逼得标准注意力必须落地 `N×N`；线性注意力的全部戏法就是找一个可分解的核替掉 softmax，把括号挪到右边。

**练习 4：** 解释为什么 Mistral 同时使用 SWA 和 FlashAttention 不矛盾——它们各自优化了 O(N²) 的哪个部分？若只能留一个来处理 `N=1M` 的超长上下文，你留哪个，为什么？

> **答案：** SWA 是架构层，把「要算的 token 对」从 `O(N²)` 砍到 `O(N·w)`（近似）；FlashAttention 是内核层，把「每对怎么算」的访存从 `O(N²)` 降到 `O(N)`（精确）。二者正交、叠加。`N=1M` 时必须留 SWA——单靠 FlashAttention 计算量仍是 `O(N²)≈10¹²` 对，纯访存优化救不了平方级计算爆炸；只有先用稀疏把对数降成线性，超长上下文才可行。
