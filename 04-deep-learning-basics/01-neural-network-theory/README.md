# 01-neural-network-theory — 神经网络理论

> **所属阶段**：阶段四 · 深度学习基础
> **学习目标**：从感知机到反向传播，理解神经网络的数学原理与训练机制
> **预估时长**：5-7 天（本子模块）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [perceptron-and-mlp](./01-perceptron-and-mlp.md) | 感知机与多层感知机 | 感知机线性模型、XOR 不可分、隐藏层+非线性突破限制、通用近似定理，MLP 即 Transformer 的 FFN |
| 02 | [activation-functions](./02-activation-functions.md) | 激活函数 | Sigmoid/Tanh/ReLU/LeakyReLU/GELU/SiLU/Softmax 的公式、导数与适用时代，梯度消失成因，温度参数 T |
| 03 | [backpropagation](./03-backpropagation.md) | 反向传播算法 | 链式法则、计算图、动态图 vs 静态图、autograd 机制、梯度累积/裁剪/检查点 |
| 04 | [optimization-algorithms](./04-optimization-algorithms.md) | 优化算法 | SGD→Momentum→Adagrad→RMSprop→Adam→AdamW 演进，Warmup+Cosine 调度，GPT/LLaMA 超参配方 |

---

## 🔑 知识点详解

### 01 · 感知机与多层感知机
- **核心概念**：单个神经元 = 加权求和后过激活函数；MLP = 神经元分层堆叠，靠"线性变换 + 非线性激活"的交替获得任意函数拟合能力。
- **关键公式**：单层 `z = w·x + b, y = f(z)`；L 层前向 `a⁽ˡ⁾ = f(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)`；Transformer FFN `FFN(x) = GELU(xW₁+b₁)W₂+b₂`（隐藏层通常为 4×d_model）。
- **易错点**：① 去掉非线性激活后，多层 MLP 会塌缩成单层线性变换（`W₂W₁x = W'x`），层数白加；② 通用近似定理只保证"一层够用"，不保证参数量可控——深层用多项式级参数即可，浅层可能需指数级。
- **Java 视角**：MLP 的多层处理链 ≈ 多层 Service 调用链，每层做一次"变换 + 校验（激活）"再传给下一层。
- **前置**：无（本阶段第一课）。

### 02 · 激活函数
- **核心概念**：激活函数是引入非线性的"开关"，其形状与导数决定了梯度能否顺畅回传。
- **关键公式**：`ReLU=max(0,x)`；`Sigmoid=1/(1+e⁻ˣ)`，导数 `σ(1-σ)∈(0,0.25]`；`GELU=x·Φ(x)`；`SiLU=x·σ(x)`；`Softmax(xᵢ)=eˣⁱ/Σeˣʲ`，温度版 `softmax(x/T)`。
- **易错点**：① Sigmoid 导数最大仅 0.25，深层连乘导致梯度消失（10 层 ≈ 0.25¹⁰）；② ReLU 负半轴梯度恒为 0，学习率过大会造成"神经元死亡"（永不再激活）；③ Softmax 直接对大 logits 求 exp 会溢出，须减去 max（PyTorch 已内置）。
- **Java 视角**：激活函数 ≈ 数据管道中的 Transformer 变换器——不改结构，只对值做非线性映射。
- **前置**：01（激活函数是 MLP 的核心组件）。

### 03 · 反向传播算法
- **核心概念**：反向传播 = 链式法则 + 动态规划，在计算图上从损失端反向一次遍历即可求出所有参数的梯度。
- **关键公式/API**：链式法则 `∂L/∂w = ∂L/∂y·∂y/∂z·∂z/∂w`；PyTorch 中 `loss.backward()`（仅对标量）、`x.grad`、`optimizer.zero_grad()`、`torch.no_grad()`、`.detach()`。
- **易错点**：① PyTorch 梯度默认**累积**，每步必须 `zero_grad()`（除非刻意做梯度累积）；② `backward()` 只能对标量调用，向量需先 `.sum()` 或传 `gradient` 参数；③ 数值微分需 O(参数量) 次前向，绝不用于训练——autograd 一次前向+一次反向即可。
- **Java 视角**：前向 = Controller→Service→DAO 调用链；反向 = 异常/贡献度从 DAO→Service→Controller 逐层回传；autograd ≈ AOP 运行时自动织入梯度逻辑。
- **前置**：01、02（需先理解前向传播与激活函数导数）。

### 04 · 优化算法
- **核心概念**：优化器决定"沿梯度走多大步"；Adam 系用一阶矩（动量）+ 二阶矩（自适应学习率）自动调节步长，AdamW 把权重衰减从梯度更新中解耦。
- **关键公式/API**：SGD `θ=θ-lr·g`；Adam `m=β₁m+(1-β₁)g, v=β₂v+(1-β₂)g², θ=θ-lr·m̂/(√v̂+ε)`；`optim.AdamW(params, lr, betas, weight_decay)` + `SequentialLR/CosineAnnealingLR/OneCycleLR`。
- **易错点**：① Adam 的 `weight_decay` 实为 L2 正则、会被自适应缩放稀释，大模型应改用 AdamW；② 大模型必须 Warmup——训练初期 Adam 矩估计不准，直接用大 lr 会发散；③ AdamW 优化器状态（m、v 各一份 FP32）约占参数量 2 倍显存，7B 模型仅优化器状态就 ~56GB。
- **Java 视角**：优化器 ≈ 策略模式（Strategy），SGD/Adam/AdamW 是可替换的参数更新策略。
- **前置**：03（优化器消费反向传播算出的梯度）。

---

## 🎯 学习要点

- **手推一遍反向传播**：拿 3.2 节的两层 Sigmoid 网络，用纸笔从 `∂L/∂a2` 逐层推到 `∂L/∂W1`，再用 `loss.backward()` 对照验证——这是本阶段最该内化的能力。
- **验证"线性塌缩"**：搭一个去掉激活函数的多层网络，确认它在 XOR 上必然失败，理解非线性为何不可或缺。
- **背下激活函数导数表**：Sigmoid、Tanh、ReLU、GELU 各自的输出范围与导数，这是判断梯度消失/爆炸的基础。
- **理解 Adam→AdamW 的一处修正**：能说清"权重衰减解耦"到底改了哪一步公式，以及为什么大模型都用 AdamW。
- **记住大模型标准配方**：AdamW + Warmup(5-10%) + Cosine Decay + 梯度裁剪(max_norm=1.0)，并能对照 GPT-3/LLaMA 的具体超参。
- **算一次显存账**：给定参数量，估算 FP32 下"参数 + 梯度 + Adam 状态"的总显存，理解为什么需要混合精度与 ZeRO。

---

## 🔗 关联

- **上一模块**：[阶段三 · 机器学习基础](../../03-machine-learning-basics/README.md)（线性/逻辑回归是单层网络的特例）
- **下一模块**：[02-pytorch](../02-pytorch/README.md)（用 PyTorch 把本章理论落地为可训练代码）
- **本阶段总览**：[阶段四 · 深度学习基础](../README.md)
