# 反向传播算法 / Backpropagation

## 1. 背景（Background）
> 反向传播利用链式法则从输出层反向计算梯度。PyTorch 的 `autograd` 自动完成这个过程。

## 2-3. 知识点与内容
```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + 3*x[1]
y.backward()
print(x.grad)  # [4.0, 3.0] — dydx0=2x0=4, dydx1=3
```

## 4. 详细推理
- 链式法则：dL/dw = dL/dy × dy/dw
- PyTorch 动态图：每次前向传播构建新图，灵活支持条件分支
- `requires_grad=True` 标记需要梯度的参数

## 5-6. 例题/习题
**练习：** 手推两层网络的反向传播，再用 autograd 验证。
