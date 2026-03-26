# 模型量化 / Model Quantization

## 1. 背景（Background）

> **为什么要学这个？**
>
> 7B 模型原始大小约 **14GB (FP16)**，量化到 INT4 后变为 **~4GB**，可以在消费级 GPU 甚至 CPU 上运行。量化是**大模型民主化**的关键技术——不需要 A100，你的 MacBook 也能跑大模型。
>
> 对于 Java 工程师来说，量化就像 **数据压缩**——JPEG 对图片做有损压缩，量化对模型权重做有损压缩，在可接受的质量损失下大幅减小体积。

## 2. 知识点（Key Concepts）

| 精度 | 位数 | 7B 模型大小 | 显存需求 | 速度 |
|------|------|------------|---------|------|
| FP32 | 32-bit | 28 GB | ~32 GB | 基准 |
| FP16/BF16 | 16-bit | 14 GB | ~16 GB | 2x |
| INT8 | 8-bit | 7 GB | ~8 GB | 2-3x |
| INT4 (GPTQ) | 4-bit | 3.5 GB | ~6 GB | 3-4x |
| INT4 (GGUF) | 4-bit | 3.5 GB | CPU 可运行 | 取决于 CPU |

## 3. 内容（Content）

### 3.1 量化方法对比

```
主流量化方案：

1. bitsandbytes (NF4) — QLoRA 使用
   运行时量化，加载模型时自动量化
   优势: 简单，与 HuggingFace 无缝集成
   劣势: 推理稍慢（需要反量化）

2. GPTQ — 离线量化
   在校准数据上优化量化误差
   优势: 推理更快（权重已压缩）
   劣势: 需要校准数据和量化时间

3. AWQ (Activation-aware Weight Quantization)
   根据激活值分布选择性量化
   优势: 质量最好
   劣势: 量化过程较慢

4. GGUF — llama.cpp 使用
   专为 CPU 推理优化的格式
   命名: Q4_K_M（推荐）, Q5_K_M, Q8_0
   优势: 纯 CPU 运行，跨平台
```

### 3.2 量化实战

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ============================================================
# 方式 1: bitsandbytes 4-bit 量化
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算精度
    bnb_4bit_use_double_quant=True,        # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# ============================================================
# 方式 2: GPTQ 离线量化后直接加载
# ============================================================
# 加载已量化的模型
# model = AutoModelForCausalLM.from_pretrained(
#     "TheBloke/Llama-2-7B-GPTQ",
#     device_map="auto",
# )
```

### 3.3 GGUF + llama.cpp（CPU 推理）

```bash
# llama.cpp: 纯 CPU 也能跑大模型
# 安装
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp && make

# 下载 GGUF 模型（HuggingFace 上有大量）
# 推理
# ./main -m model.Q4_K_M.gguf -p "Hello" -n 100

# Python 绑定
# pip install llama-cpp-python
```

```python
# from llama_cpp import Llama
# llm = Llama(model_path="model.Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=0)
# output = llm("Hello!", max_tokens=100)
```

## 4. 详细推理（Deep Dive）

### 4.1 量化的数学原理

```
线性量化:
  x_int = round(x_float / scale) + zero_point
  x_dequant = (x_int - zero_point) * scale

NF4 (NormalFloat4):
  假设权重服从正态分布 N(0, σ)
  将正态分布均匀分成 16 个区间
  每个区间用该区间的中位数表示
  → 相比均匀 INT4，更好地保留了分布信息

量化粒度:
  Per-tensor: 整个张量共用 scale → 最粗，质量最差
  Per-channel: 每个通道独立 scale → 较好
  Per-group: 每 128 个元素独立 scale → 最好（GPTQ 使用）
```

## 5. 例题（Worked Examples）

```python
# 对比不同精度的模型大小
import os
# FP16: ~14 GB / INT8: ~7 GB / INT4: ~3.5 GB
# 模型质量：FP16 ≈ INT8 > GPTQ-INT4 > GGUF-Q4_K_M
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 bitsandbytes 加载 4-bit 量化模型，测量显存占用。

**练习 2：** 对比 FP16 和 INT4 模型在同一问题上的回答质量。

### 进阶题

**练习 3：** 用 GGUF 格式在纯 CPU 上运行 7B 模型，测量推理速度。

**练习 4：** 用 AutoGPTQ 对一个模型做离线量化，自定义校准数据集。
