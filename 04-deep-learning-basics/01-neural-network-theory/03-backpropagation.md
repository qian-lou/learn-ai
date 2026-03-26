# 反向传播算法 / Backpropagation

## 1. 背景（Background）

> **为什么要学这个？**
>
> 反向传播（Backpropagation, BP）是训练神经网络的核心算法。它利用微积分的**链式法则（Chain Rule）**，从输出层开始反向计算每个参数对损失函数的梯度，然后用这些梯度更新参数。对于 Java 工程师来说，可以把反向传播理解为一种**反向依赖计算**——类似于构建一个依赖图（DAG），然后从末端反向遍历计算每个节点的"贡献度"。
>
> 好消息是：PyTorch 的 `autograd` 引擎会**自动完成整个反向传播过程**，你不需要手动推导梯度。但理解原理至关重要，它能帮你诊断训练问题（梯度消失/爆炸）、理解为什么某些操作不能求导、以及如何自定义反向传播逻辑。
>
> **在整个体系中的位置：** 反向传播是连接"前向传播"和"参数优化"的桥梁。前向传播计算损失，反向传播计算梯度，优化器利用梯度更新参数。理解反向传播，才能理解大模型训练中的梯度累积、梯度裁剪、混合精度等技术。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | Java 类比 |
|------|------|-----------|
| 链式法则 (Chain Rule) | 复合函数求导法则 | 责任链模式中各层的影响传导 |
| 计算图 (Computational Graph) | 记录前向计算的 DAG | 依赖注入容器中的 Bean 依赖图 |
| 梯度 (Gradient) | 损失对参数的偏导数 | 每个参数对 "误差" 的贡献度 |
| 前向传播 (Forward Pass) | 输入→输出的计算 | Controller → Service → DAO 调用链 |
| 反向传播 (Backward Pass) | 输出→输入的梯度计算 | 异常从 DAO → Service → Controller 的传播 |
| 自动微分 (Autograd) | PyTorch 自动计算梯度 | 类似 AOP 在运行时自动织入逻辑 |
| 计算图模式 | 动态图（PyTorch） vs 静态图（TensorFlow） | 动态代理 vs 静态代理 |

**核心要点：**
- 反向传播 = **链式法则** + **动态规划**（避免重复计算）
- PyTorch 使用**动态计算图**：每次 `forward()` 构建新图，`backward()` 计算梯度后释放图
- `requires_grad=True` 标记需要梯度的张量（叶子节点）
- `.backward()` 只能对**标量**调用（非标量需要传入 `gradient` 参数）

## 3. 内容（Content）

### 3.1 链式法则回顾

```
链式法则（微积分基础）：

如果 y = f(g(x))，则：
  dy/dx = dy/dg · dg/dx = f'(g(x)) · g'(x)

多变量链式法则：
  如果 L = L(y)，y = f(z)，z = w·x + b
  则：
    ∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w
    ∂L/∂b = ∂L/∂y · ∂y/∂z · ∂z/∂b

计算方向：
  前向：x → z → y → L（求损失值）
  反向：L → y → z → x（求梯度）
```

### 3.2 手推反向传播示例

```python
import numpy as np

# ============================================================
# 两层网络的手动反向传播
# Manual backpropagation for a 2-layer network
# ============================================================

# 网络结构: Input(2) → Hidden(2) → Output(1)
# Network: Input(2) → Hidden(2) → Output(1)

# 初始化参数 / Initialize parameters
W1 = np.array([[0.15, 0.20], [0.25, 0.30]])  # Shape: [2, 2]
b1 = np.array([0.35, 0.35])                   # Shape: [2]
W2 = np.array([[0.40, 0.45]])                  # Shape: [1, 2]
b2 = np.array([0.60])                          # Shape: [1]

# 输入和目标 / Input and target
x = np.array([0.05, 0.10])
y_true = np.array([0.01])

# ============= 前向传播 / Forward Pass =============
# 第一层 / Layer 1
z1 = W1 @ x + b1         # [0.15*0.05+0.20*0.10+0.35, 0.25*0.05+0.30*0.10+0.35]
                           # = [0.3775, 0.3925]
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid: [0.5933, 0.5969]

# 第二层 / Layer 2
z2 = W2 @ a1 + b2        # [0.40*0.5933+0.45*0.5969+0.60] = [1.1059]
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid: [0.7514]

# 损失 / Loss (MSE)
loss = 0.5 * (y_true - a2) ** 2  # 0.5 * (0.01 - 0.7514)^2 = 0.2748

print(f"前向传播 / Forward:")
print(f"  z1={z1}, a1={a1}")
print(f"  z2={z2}, a2={a2}")
print(f"  loss={loss}")

# ============= 反向传播 / Backward Pass =============
# 链式法则，从后向前计算
# Chain rule, compute from back to front

# ∂L/∂a2 = -(y - a2) = a2 - y
dL_da2 = a2 - y_true     # [0.7414]

# ∂a2/∂z2 = a2 * (1 - a2)  (Sigmoid 导数)
da2_dz2 = a2 * (1 - a2)  # [0.1868]

# ∂L/∂z2 = ∂L/∂a2 · ∂a2/∂z2
dL_dz2 = dL_da2 * da2_dz2  # [0.1385]

# ∂L/∂W2 = ∂L/∂z2 · a1^T
dL_dW2 = dL_dz2 * a1     # Shape: [1, 2]

# ∂L/∂b2 = ∂L/∂z2
dL_db2 = dL_dz2

# 继续向前传播梯度 / Continue propagating gradient backward
# ∂L/∂a1 = W2^T · ∂L/∂z2
dL_da1 = W2.T @ dL_dz2   # Shape: [2, 1]

# ∂L/∂z1 = ∂L/∂a1 · a1(1-a1)
da1_dz1 = a1 * (1 - a1)
dL_dz1 = dL_da1.flatten() * da1_dz1

# ∂L/∂W1 = ∂L/∂z1 · x^T
dL_dW1 = np.outer(dL_dz1, x)

# ∂L/∂b1 = ∂L/∂z1
dL_db1 = dL_dz1

print(f"\n反向传播 / Backward:")
print(f"  dL/dW2 = {dL_dW2}")
print(f"  dL/dW1 = {dL_dW1}")
```

### 3.3 PyTorch Autograd 自动微分

```python
import torch

# ============================================================
# PyTorch autograd 自动完成反向传播
# PyTorch autograd does backpropagation automatically
# ============================================================

# 1. 标量函数求导 / Scalar function derivative
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + 3 * x[1]   # y = x₀² + 3x₁
y.backward()                # 自动反向传播
print(f"x = {x.data}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad}")   # [2*x₀, 3] = [4.0, 3.0]


# ============================================================
# 2. 理解计算图 / Understanding the Computational Graph
# ============================================================
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# 前向传播自动构建计算图
# Forward pass automatically builds computational graph
c = a * b         # c = 6.0,   grad_fn=MulBackward0
d = c + a         # d = 8.0,   grad_fn=AddBackward0
e = d ** 2        # e = 64.0,  grad_fn=PowBackward0

e.backward()
print(f"de/da = {a.grad}")  # ∂e/∂a = 2d · (∂d/∂a) = 2*8*(b+1) = 16*4 = 64
print(f"de/db = {b.grad}")  # ∂e/∂b = 2d · (∂d/∂b) = 2*8*a = 16*2 = 32

# 计算图可视化：
# Computational graph:
#   a ──┬── MUL → c ──┬── ADD → d ── POW → e
#       │              │
#   b ──┘          a ──┘


# ============================================================
# 3. 注意事项 / Important Notes
# ============================================================

# 3a. 梯度会累积！必须手动清零
# Gradients accumulate! Must zero them manually
x = torch.tensor(1.0, requires_grad=True)

y1 = x * 2
y1.backward()
print(f"第一次: x.grad = {x.grad}")  # 2.0

y2 = x * 3
y2.backward()
print(f"第二次（累积）: x.grad = {x.grad}")  # 2.0 + 3.0 = 5.0！

# 正确做法：清零
x.grad.zero_()
y3 = x * 3
y3.backward()
print(f"清零后: x.grad = {x.grad}")  # 3.0

# 3b. 使用 no_grad() 停止梯度追踪（推理时）
# Use no_grad() to stop gradient tracking (during inference)
with torch.no_grad():
    # 这里的计算不会被记录到计算图
    # Computations here won't be recorded in the graph
    result = x * 2
    print(f"result.requires_grad = {result.requires_grad}")  # False

# 3c. detach() 从计算图分离
# detach() separates from computation graph
z = x * 2
z_detached = z.detach()  # 创建不追踪梯度的副本
```

### 3.4 动态图 vs 静态图

```
PyTorch（动态图 / Define-by-Run）：
  ┌──────────────────────────────────┐
  │  每次调用 forward() 都构建新图    │
  │  - 支持 if/for/while 条件控制    │
  │  - 像写普通 Python 一样灵活      │
  │  - 适合研究和调试                │
  │  类比：Java 的动态代理            │
  └──────────────────────────────────┘

TensorFlow 1.x（静态图 / Define-then-Run）：
  ┌──────────────────────────────────┐
  │  先定义整个计算图，然后执行        │
  │  - 不支持动态控制流（1.x）        │
  │  - 编译优化可能更好               │
  │  - 适合大规模部署                 │
  │  类比：Java 的静态代理            │
  └──────────────────────────────────┘
```

```python
# ============================================================
# 动态图的强大之处：条件分支
# Power of dynamic graphs: conditional branching
# ============================================================
def dynamic_model(x: torch.Tensor) -> torch.Tensor:
    """根据输入动态选择计算路径 / Dynamically choose computation path.
    
    Args:
        x: 输入张量 / Input tensor.
    
    Returns:
        计算结果 / Computed result.
    """
    if x.sum() > 0:  # 运行时条件！静态图做不到
        return x ** 2
    else:
        return x ** 3

x1 = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = dynamic_model(x1)  # 走 x**2 分支
y1.sum().backward()
print(f"正输入梯度: {x1.grad}")  # [2.0, 4.0]

x2 = torch.tensor([-1.0, -2.0], requires_grad=True)
y2 = dynamic_model(x2)  # 走 x**3 分支
y2.sum().backward()
print(f"负输入梯度: {x2.grad}")  # [3.0, 12.0]
```

## 4. 详细推理（Deep Dive）

### 4.1 反向传播的时间复杂度

```
Time: O(N)，其中 N 是计算图中的节点数
Space: O(N)，需要存储所有中间结果（用于计算梯度）

这是反向传播的关键权衡：
  用空间换时间（存储前向结果以加速反向计算）

对于大模型（如 GPT-3，175B 参数）：
  - 前向传播需要的激活值存储是主要的显存瓶颈
  - 解决方案：梯度检查点（Gradient Checkpointing）
    只保存部分层的激活值，需要时重新计算
    时间增加 ~33%，但显存减少 ~60%
```

### 4.2 为什么要用计算图（而非数值微分）？

```
方法对比：

1. 数值微分（有限差分）：
   ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / 2ε
   
   缺点：
   - 每个参数需要两次前向传播 → O(P) 次前向（P=参数数量）
   - 175B 参数 → 需要 350B 次前向传播！
   - 浮点精度问题（ε 太大不准，太小会被截断）

2. 符号微分：
   直接求导数的解析表达式
   
   缺点：
   - 表达式可能爆炸性增长（"表达式膨胀"）
   - 无法处理控制流

3. 自动微分（Autograd）✅：
   在计算图上自动应用链式法则
   
   优点：
   - 只需一次前向 + 一次反向 → O(1) 次遍历
   - 精确（不是近似）
   - 支持动态控制流
```

### 4.3 梯度裁剪（Gradient Clipping）

```python
import torch
import torch.nn as nn

# ============================================================
# 梯度裁剪：防止梯度爆炸（大模型训练必备）
# Gradient clipping: Preventing gradient explosion (essential for LLM)
# ============================================================

model = nn.Linear(100, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()

# 方法 1: 按范数裁剪（推荐）
# Method 1: Clip by norm (recommended)
# 如果总梯度范数 > max_norm，按比例缩放所有梯度
# If total gradient norm > max_norm, scale all gradients proportionally
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方法 2: 按值裁剪
# Method 2: Clip by value
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

optimizer.step()

# 大模型训练中的典型设置：
# Typical settings in LLM training:
# max_norm = 1.0 (GPT-2, GPT-3)
# max_norm = 0.3 (LLaMA)
```

## 5. 例题（Worked Examples）

### 例题 1：手推简单函数的反向传播

**问题：** 计算 `L = (x*w + b - y)²` 对 `w` 和 `b` 的梯度。给定 x=2, w=0.5, b=0.1, y=3。

**解答：**

```python
import torch

# 手动计算 / Manual computation
x, w, b, y = 2.0, 0.5, 0.1, 3.0
z = x * w + b           # 2*0.5 + 0.1 = 1.1
L = (z - y) ** 2        # (1.1 - 3)^2 = 3.61

# ∂L/∂z = 2(z - y) = 2(1.1 - 3) = -3.8
# ∂z/∂w = x = 2
# ∂z/∂b = 1
# ∂L/∂w = ∂L/∂z · ∂z/∂w = -3.8 * 2 = -7.6
# ∂L/∂b = ∂L/∂z · ∂z/∂b = -3.8 * 1 = -3.8
print(f"手动: dL/dw = -7.6, dL/db = -3.8")

# PyTorch 自动验证 / PyTorch auto verification
x_t = torch.tensor(2.0)
w_t = torch.tensor(0.5, requires_grad=True)
b_t = torch.tensor(0.1, requires_grad=True)
y_t = torch.tensor(3.0)

L_t = (x_t * w_t + b_t - y_t) ** 2
L_t.backward()
print(f"PyTorch: dL/dw = {w_t.grad.item()}, dL/db = {b_t.grad.item()}")
# 输出: dL/dw = -7.6, dL/db = -3.8 ✅ 完全一致
```

### 例题 2：用 autograd 验证手推的两层网络梯度

```python
import torch
import torch.nn as nn

# ============================================================
# 两层网络：手动 vs 自动梯度对比
# Two-layer network: manual vs automatic gradient comparison
# ============================================================

# 定义参数（与 3.2 节的手动例子一致）
W1 = torch.tensor([[0.15, 0.20], [0.25, 0.30]], requires_grad=True)
b1 = torch.tensor([0.35, 0.35], requires_grad=True)
W2 = torch.tensor([[0.40, 0.45]], requires_grad=True)
b2 = torch.tensor([0.60], requires_grad=True)

x = torch.tensor([0.05, 0.10])
y_true = torch.tensor([0.01])

# 前向传播 / Forward pass
z1 = W1 @ x + b1
a1 = torch.sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = torch.sigmoid(z2)

# MSE 损失 / MSE loss
loss = 0.5 * (y_true - a2) ** 2

# 反向传播 / Backward pass
loss.backward()

print("自动微分结果 / Autograd results:")
print(f"  dL/dW1 = {W1.grad}")
print(f"  dL/dW2 = {W2.grad}")
print(f"  dL/db1 = {b1.grad}")
print(f"  dL/db2 = {b2.grad}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 对于函数 `f(x) = sin(x²) + e^(2x)`，使用 PyTorch 的 autograd 计算 `x = 1.0` 处的导数，并与手动计算结果对比。

> **提示：** 手动求导: `f'(x) = 2x·cos(x²) + 2e^(2x)`，所以 `f'(1) = 2cos(1) + 2e² ≈ 1.081 + 14.778 = 15.859`

**练习 2：** 解释以下代码为什么会报错，以及如何修复：
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
y.backward()  # Error!
```

> **答案：** `backward()` 只能对标量调用。`y` 是向量 `[2.0, 4.0]`，需要 `y.sum().backward()` 或传入 `gradient` 参数。

### 进阶题

**练习 3：** 实现梯度检查（Gradient Checking），用数值微分验证 autograd 的正确性：

```python
def gradient_check(model, x, y, epsilon=1e-5):
    """用数值微分验证 autograd 梯度 / Verify autograd with numerical differentiation.
    
    Args:
        model: 模型 / Model.
        x: 输入 / Input.
        y: 目标 / Target.
        epsilon: 扰动量 / Perturbation amount.
    """
    # TODO: 实现梯度检查
    # 1. 用 autograd 计算梯度
    # 2. 对每个参数用有限差分计算数值梯度
    # 3. 比较两者的相对误差
    pass
```

> **参考答案：**
> ```python
> def gradient_check(model, x, y, epsilon=1e-5):
>     loss_fn = nn.MSELoss()
>     
>     # Autograd 梯度 / Autograd gradient
>     pred = model(x)
>     loss = loss_fn(pred, y)
>     loss.backward()
>     
>     for name, param in model.named_parameters():
>         if param.grad is None:
>             continue
>         auto_grad = param.grad.clone()
>         num_grad = torch.zeros_like(param)
>         
>         # 数值微分 / Numerical differentiation
>         for i in range(param.numel()):
>             param_flat = param.data.view(-1)
>             old_val = param_flat[i].item()
>             
>             param_flat[i] = old_val + epsilon
>             loss_plus = loss_fn(model(x), y)
>             
>             param_flat[i] = old_val - epsilon
>             loss_minus = loss_fn(model(x), y)
>             
>             num_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)
>             param_flat[i] = old_val
>         
>         rel_error = torch.norm(auto_grad - num_grad) / (torch.norm(auto_grad) + torch.norm(num_grad) + 1e-8)
>         print(f"{name}: relative error = {rel_error.item():.2e}")
> ```

**练习 4：** 解释大模型训练中的"梯度累积"（Gradient Accumulation）技术：为什么需要？如何实现？梯度累积 4 步等价于什么？

> **提示：** 当 batch_size=8 但 GPU 显存只够 batch_size=2 时，可以累积 4 次梯度再更新参数，等价于 batch_size=8 的效果。
