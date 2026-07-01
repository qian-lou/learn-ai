# 02-pytorch — PyTorch 框架

> **所属阶段**：阶段四 · 深度学习基础
> **学习目标**：掌握 PyTorch 从张量、建模到训练、加速、持久化的全流程
> **预估时长**：7-9 天（本子模块）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [tensor-basics](./01-tensor-basics.md) | Tensor 基础与自动求导 | Tensor 创建/形状操作(view/permute/cat/stack)、广播、matmul、autograd、stride 与 contiguous |
| 02 | [nn-module](./02-nn-module.md) | nn.Module 模型构建 | 继承 Module 实现 forward、ModuleList/ModuleDict、参数管理与冻结、train/eval、初始化 |
| 03 | [training-loop](./03-training-loop.md) | 训练循环与 DataLoader | Dataset/DataLoader、损失函数、五步训练循环、验证/EarlyStopping、梯度累积 |
| 04 | [gpu-acceleration](./04-gpu-acceleration.md) | GPU 加速与混合精度 | 设备管理、显存监控、AMP autocast、GradScaler、FP16/BF16/TF32 对比 |
| 05 | [model-save-and-load](./05-model-save-and-load.md) | 模型保存与加载 | state_dict、Checkpoint、跨设备加载、weights_only、safetensors、分片保存 |

---

## 🔑 知识点详解

### 01 · Tensor 基础与自动求导
- **核心概念**：Tensor 是"能跑在 GPU 上 + 自动记录运算历史"的多维数组；大模型里所有数据（token、参数、梯度、注意力权重）都是 Tensor。
- **关键 API**：`torch.tensor/zeros/randn`；形状 `view/reshape/permute/transpose/unsqueeze/squeeze/cat/stack`；`@`/`matmul`；`requires_grad_()` + `.backward()` + `.grad`。
- **易错点**：① `view()` 要求内存连续，`permute/transpose` 后需 `.contiguous()` 才能 view（或直接用 `reshape`）；② `*` 是逐元素乘、`@` 才是矩阵乘；③ 有梯度的 Tensor 转 NumPy 要先 `.detach()`，GPU Tensor 要先 `.cpu()`。
- **Java 视角**：Tensor ≈ 增强版多维数组——`int[][]` 升级为可 GPU 计算、可自动微分的智能对象。
- **前置**：阶段一 NumPy（ndarray 与 Tensor 几乎一一对应）。

### 02 · nn.Module 模型构建
- **核心概念**：所有模型都继承 `nn.Module` 并实现 `forward()`；框架自动完成参数注册、设备迁移、序列化。大模型是 Module 组成的树。
- **关键 API**：`super().__init__()`、`nn.Linear/Embedding/LayerNorm/Dropout`、`nn.Sequential/ModuleList/ModuleDict`、`named_parameters()`、`nn.Parameter`、`model.train()/eval()`。
- **易错点**：① 子模块存进普通 `list` 不会被注册（`parameters()` 为空、`.to()` 也搬不动），必须用 `nn.ModuleList/ModuleDict`；② 调用模型用 `model(x)` 而非 `model.forward(x)`（后者跳过 hooks）；③ `model.eval()` 只切换 Dropout/BN 行为，**不停止梯度**，推理须再加 `torch.no_grad()`。
- **Java 视角**：`nn.Module` ≈ 抽象基类，`forward()` ≈ 重写的 `process()`，`state_dict()` ≈ `serialize()`。
- **前置**：01（模型层的输入输出都是 Tensor）。

### 03 · 训练循环与 DataLoader
- **核心概念**：训练 = 前向→算损失→反向→更新的循环；DataLoader 是后台并行的批量数据管道。
- **关键 API**：五步循环 `zero_grad() → output=model(x) → loss=criterion(output,y) → loss.backward() → optimizer.step()`；`Dataset(__len__/__getitem__)`、`DataLoader(batch_size, shuffle, num_workers, pin_memory)`；`nn.CrossEntropyLoss`（输入 logits，不要先 softmax）。
- **易错点**：① 忘记 `zero_grad()` 会累积上一步梯度；② `CrossEntropyLoss` 目标是类别索引不是 one-hot，且输入是未过 softmax 的 logits；③ 语言模型损失需把 `[B,S,V]`/`[B,S]` reshape 成 `[B*S,V]`/`[B*S]`。
- **Java 视角**：训练循环 ≈ 事件循环；DataLoader ≈ BlockingQueue + 线程池（后台异步加载、前台消费）。
- **前置**：02（需要一个 Module 模型）、01-neural-network-theory/04（优化器）。

### 04 · GPU 加速与混合精度
- **核心概念**：GPU 靠数千个小核并行做矩阵运算，比 CPU 快 50-100×；混合精度用半精度算、单精度存关键量，省一半显存、快 2-3×。
- **关键 API**：`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`、`tensor.to(device)`、`torch.amp.autocast('cuda', dtype=...)`、`GradScaler`、`pin_memory=True`+`non_blocking=True`。
- **易错点**：① 模型和输入必须在**同一设备**，否则报错；② FP16 范围小（最大 65504）易上溢/下溢，须配 `GradScaler`；BF16 范围同 FP32、**不需要** GradScaler；③ GPU 计时前后要 `torch.cuda.synchronize()`，否则测到的是异步下发时间而非真实耗时。
- **Java 视角**：GPU 加速 ≈ 把单机 Tomcat 升级为分布式集群——同样的代码、底层并行度天差地别。
- **前置**：03（在训练循环里插入设备迁移与 AMP）。

### 05 · 模型保存与加载
- **核心概念**：优先保存 `state_dict`（纯 {名字:Tensor} 字典），与模型类实现解耦；恢复训练还需存 optimizer/scheduler/epoch。
- **关键 API**：`torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path, map_location=..., weights_only=True))`；`safetensors.torch.save_file/load_file`；`model.save_pretrained(max_shard_size=...)`。
- **易错点**：① 不要 `torch.save(model)` 存整个对象——pickle 有安全风险且强依赖类定义；② `torch.load` 默认走 pickle 可执行恶意代码，加载不可信权重务必 `weights_only=True`；③ 跨设备加载忘记 `map_location` 会在无 GPU 机器上报错。
- **Java 视角**：`state_dict` 保存 ≈ 对象序列化，但粒度更细、可部分加载、可跨平台。
- **前置**：02（要有 state_dict 可保存）、03（Checkpoint 用于恢复训练）。

---

## 🎯 学习要点

- **PyTorch 的动态计算图是核心优势**：每次 forward 重建图，支持 if/for 等运行时控制流，调试像写普通 Python。
- **把五步训练循环练成肌肉记忆**：`zero_grad → forward → loss → backward → step`，从 MNIST 到 GPT 结构完全一致，后续所有项目都复用。
- **形状注释是硬习惯**：每个 Tensor 旁标 `# Shape: [B, S, D]`，这是读懂/写对大模型代码的前提，尤其多头注意力的 split/merge。
- **区分 `model.eval()` 与 `torch.no_grad()`**：前者改 Dropout/BN 行为，后者停梯度追踪省显存，推理时两者都要。
- **优先 BF16 而非 FP16**：在 A100/H100 上 BF16 免 GradScaler、更省心，能说清 FP16/BF16/TF32 的位宽与取舍。
- **默认用 safetensors + weights_only**：既安全又加载快，是 Hugging Face 生态的事实标准。

---

## 🔗 关联

- **上一模块**：[01-neural-network-theory](../01-neural-network-theory/README.md)（把理论落地为可训练代码）
- **下一模块**：[03-cnn](../03-cnn/README.md)（用 nn.Module 搭建卷积网络）
- **本阶段总览**：[阶段四 · 深度学习基础](../README.md)
- **关联 Day**：训练循环/张量思维是后续 [agent-course Day-16 embedding-basics](../../agent-course/Day-16-embedding-basics.md) 起 RAG 向量化实践的底层基础。
