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
#     "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # TheBloke 已停更，改用官方/活跃仓库
#     device_map="auto",
# )
```

### 3.3 GGUF + llama.cpp（CPU 推理）

```bash
# llama.cpp: 纯 CPU 也能跑大模型
# 安装（新版用 CMake，make 已弃用）
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp && cmake -B build && cmake --build build -j

# 下载 GGUF 模型（推荐 bartowski / unsloth 仓库）
# 推理（可执行已从 main 改名为 llama-cli）
# ./build/bin/llama-cli -m model.Q4_K_M.gguf -p "Hello" -n 100

# Python 绑定
# pip install llama-cpp-python
```

```python
# from llama_cpp import Llama
# llm = Llama(model_path="model.Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=0)
# output = llm("Hello!", max_tokens=100)
```

### 3.4 2024-2025 量化进展：FP8/FP4 · GGUF IQ-quant · torchao

> 2025 年低比特从 INT4 进一步下探到 **8-bit/4-bit 浮点**，并出现硬件原生支持，质量损失显著低于同位宽整数量化。
> Sub-4-bit moved to floating-point (FP8/FP4) with native hardware support — far less quality loss than same-width integer quant.

**浮点量化 FP8 / FP4（NVIDIA 主线）**
- **FP8**：两种编码 `E4M3`（1符4阶3尾，范围 ±448，权重/激活首选）与 `E5M2`（5阶2尾，范围 ±57344，动态范围大）。Hopper / Blackwell **原生支持**，W8A8 推理**近无损**。
  FP8: E4M3 (±448, weights/acts) and E5M2 (wider range); native on Hopper/Blackwell, near-lossless W8A8.
- **FP4 / NVFP4**：4-bit 微缩放（microscaling）浮点，**Blackwell** 原生，配合 FP8 KV cache 把超大模型压进单机，质量明显优于 INT4。
  NVFP4: 4-bit microscaling float, Blackwell-native, beats INT4 quality.

```bash
# 用 llm-compressor 产出 FP8 权重供 vLLM 部署（动态 FP8 无需校准数据）
# Produce FP8 weights for vLLM via llm-compressor (dynamic FP8 needs no calibration)
vllm serve Qwen/Qwen2.5-7B-Instruct --quantization fp8   # 在线动态量化 / on-the-fly
```

**GGUF 的 IQ 系列（极致压缩，CPU/端侧）**
- `IQ2_XXS` / `IQ3_XXS` / `IQ4_XS`：基于 **importance-matrix（imatrix）** 的非均匀量化，按权重重要性分配码本。
  imatrix-based non-uniform quant — bits follow weight importance.
- 同位宽下质量优于老的 `Qn_K`：如 `IQ4_XS` 体积小于 `Q4_K_M` 而困惑度更接近 FP16，适合显存/内存极度紧张的端侧。
  At equal bits, IQ beats legacy Qn_K — e.g. IQ4_XS is smaller than Q4_K_M yet closer to FP16.

**torchao（PyTorch 原生，一行量化）**
```python
# Time: O(N) over params  Space: 权重位宽 32→4，约 -75% 显存 / ~75% VRAM cut
from torchao.quantization import quantize_, int4_weight_only

# int4 weight-only：仅压权重，激活仍 bf16，走 tinygemm kernel / weight-only, bf16 acts
quantize_(model, int4_weight_only(group_size=128))   # group_size 越小越准、略增体积
# int8 weight-only 同理：quantize_(model, int8_weight_only())
```
- 高速 kernel：**Marlin**（W4A16/W8A16 GEMM，重叠反量化与计算，batch 256 仍约 2x 加速）、**Machete**（vLLM 为 Hopper 定制的 W4A16/W8A16 kernel）。
  Fast kernels: Marlin (W4A16/W8A16 GEMM), Machete (vLLM's Hopper W4A16 kernel).
- **HQQ**（Half-Quadratic Quantization）：**无需校准数据**、量化极快，质量接近需校准的 GPTQ/AWQ。
  HQQ: calibration-free, very fast, quality near GPTQ/AWQ.

> 仓库更新提示 / repo note：**TheBloke 已停更**，2026 年 GGUF 取自 **bartowski / unsloth**，FP8·NVFP4·GPTQ 取自**官方仓库或 RedHatAI（llm-compressor 产出）**。

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

*参考答案*：

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="cuda")
# 加载后读取已分配显存 / Read allocated VRAM after loading
print(f"VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
# 7B NF4 权重约 ~4GB；加上激活/KV cache，整机峰值约 5-6GB
# 7B NF4 weights ≈ ~4GB; with activations/KV cache peak ≈ 5-6GB
```

**练习 2：** 对比 FP16 和 INT4 模型在同一问题上的回答质量。

*参考答案*：

思路：同一 prompt、同一采样参数（建议 `temperature=0` 贪心解码以保证可复现），分别用 FP16（`torch_dtype=torch.float16`）和 4-bit 模型生成，逐条人工对比。

```python
# 关键：固定随机性，只让"精度"成为变量 / Fix randomness; precision is the only variable
out = model.generate(**inputs, do_sample=False, max_new_tokens=256)
```

观察结论：短问答两者通常几乎无差异；INT4 在长链路推理、数学、代码等任务上偶有退化。客观评测可跑 perplexity 或小型 benchmark（如 MMLU 子集），FP16 ≈ INT8 > AWQ/GPTQ-INT4 ≳ GGUF-Q4。

### 进阶题

**练习 3：** 用 GGUF 格式在纯 CPU 上运行 7B 模型，测量推理速度。

*参考答案*：

```bash
# 从 bartowski（TheBloke 已停更）下载 GGUF / Download GGUF from bartowski
# 例：Qwen2.5-7B-Instruct-Q4_K_M.gguf
# 可执行已从 main 改名为 llama-cli / binary renamed from main to llama-cli
./build/bin/llama-cli -m Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -p "Explain quantization in one sentence." -n 128 -t 8
# 运行结束后 llama.cpp 会打印 eval time 与 tokens/s（即 decode 吞吐）
# llama.cpp prints eval time and tokens/s (decode throughput) at the end
```

说明：纯 CPU 速度取决于内存带宽与线程数，桌面级 CPU 上 7B-Q4 一般在个位数到十几 tokens/s；IQ 系列（如 IQ4_XS）体积更小、质量接近。也可用 `pip install llama-cpp-python` 在 Python 中自行计时（记录生成 token 数 / 耗时）。

**练习 4：** 用 AutoGPTQ 对一个模型做离线量化，自定义校准数据集。

*参考答案*：

注意：AutoGPTQ 已基本停止维护，2026 年推荐用 **GPTQModel**（社区接棒）或 **llm-compressor / llmcompressor**（兼容 vLLM）做离线量化；AWQ 可用 `autoawq`。下面用 GPTQModel 演示自定义校准集（校准数据应贴近真实业务分布）。

```python
from gptqmodel import GPTQModel, QuantizeConfig

# 校准数据：几百条代表性文本即可 / a few hundred representative samples suffice
calib = ["你的领域语料样本 1", "sample 2", "..."]
cfg = QuantizeConfig(bits=4, group_size=128)  # per-group 量化，质量最佳
model = GPTQModel.load("Qwen/Qwen2.5-7B-Instruct", cfg)
model.quantize(calib)
model.save("./Qwen2.5-7B-GPTQ-Int4")
# group_size=128 是质量/体积的常用折中 / common quality-size trade-off
```
