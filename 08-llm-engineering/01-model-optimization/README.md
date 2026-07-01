# 01-model-optimization — 模型优化

> **所属阶段**：阶段八 · 大模型部署与工程化
> **学习目标**：掌握模型压缩（量化/蒸馏）与推理加速三大类技术，把"能跑通"的模型变成"跑得起、跑得快、成本可控"的服务
> **预估时长**：4-5 天（含动手量化 + 跑通 vLLM/llama.cpp）

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [quantization](./01-quantization.md) | 模型量化 | 线性量化数学原理、NF4/INT8/INT4、GPTQ/AWQ/HQQ 离线量化、FP8/NVFP4 浮点量化、GGUF IQ 系列与 torchao 一行量化 |
| 02 | [inference-acceleration](./02-inference-acceleration.md) | 推理加速 | KV Cache（含显存估算）、FlashAttention（IO 感知）、Continuous Batching、Speculative Decoding、PagedAttention |
| 03 | [knowledge-distillation](./03-knowledge-distillation.md) | 知识蒸馏 | soft label + Temperature、KL 蒸馏损失、数据蒸馏（Alpaca/Vicuna）、on-policy 蒸馏 |

---

## 🔑 知识点详解

### 01 · 模型量化（Quantization）

- **核心概念**：用更低的位宽表示权重（乃至激活），以可接受的精度损失换取显存/体积/速度——本质是"对模型权重做有损压缩"。
- **关键公式/API**：线性量化 `x_int = round(x_float / scale) + zero_point`，反量化 `x_dequant = (x_int - zero_point) * scale`。最该记的落地 API：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`（运行时 NF4）与 `torchao` 的 `quantize_(model, int4_weight_only(group_size=128))`（一行 weight-only 量化）。
- **易错点**：① NF4 假设权重服从正态分布，与均匀 INT4 不同，别混为一谈；② 量化粒度决定质量，`per-group`（如 128）优于 `per-channel` 优于 `per-tensor`，group_size 越小越准但略增体积；③ 2026 年仓库已变——`TheBloke` 停更、`AutoGPTQ` 基本停维护，GGUF 取自 `bartowski/unsloth`，离线量化改用 `GPTQModel` 或 `llm-compressor`。
- **Java 视角**：等同于 JPEG 有损压缩之于图片——牺牲人眼/模型难以察觉的细节，换来数量级的体积下降；量化方案选型就像选压缩算法（速度 vs 质量 vs 兼容性）。
- **前置**：无（阶段内起点），但需理解 FP16/BF16 与显存占用的关系。

### 02 · 推理加速（Inference Acceleration）

- **核心概念**：LLM 推理的瓶颈往往不在算力而在**内存访问与重复计算**；加速手段就是围绕"缓存、IO、批处理、并行"四个方向做工程优化。
- **关键公式/API**：KV Cache 显存 `= 2 × n_layers × n_kv_heads × d_head × seq_len × batch × dtype_size`（GQA 下务必用 `n_kv_heads` 而非 query head 数）。落地首选 `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`——PyTorch 2.x 自动选 FlashAttention 后端。
- **易错点**：① 算 KV Cache 时误用 query head 数会高估约 8 倍（LLaMA-70B GQA 实为 8 个 KV head，约 1.25GB 而非 10GB）；② SDPA 内部已做 `1/√d` 缩放，不要再手动除；③ `use_cache=False` 会让每步重算全部历史 K/V，复杂度从 O(N) 退化到 O(N²)。
- **Java 视角**：KV Cache ≈ Redis 缓存（避免重复计算），Continuous Batching ≈ 线程池的动态任务窃取（GPU 不空转），Speculative Decoding ≈ 用快路径乐观预测 + 慢路径校验。
- **前置**：01 量化（同为部署优化）；理解 Transformer 自回归生成与 Attention。

### 03 · 知识蒸馏（Knowledge Distillation）

- **核心概念**：用大模型（Teacher）的"软标签/生成行为"教小模型（Student），让小模型以更小的体积逼近大模型能力——迁移的是"暗知识"（类别间关系），不是硬答案。
- **关键公式/API**：蒸馏损失 `L = α · KL(soft_teacher ∥ soft_student) · T² + (1-α) · CE(hard_label, student)`，其中软标签 = `softmax(logits / T)`。代码核心：`F.kl_div(F.log_softmax(s/T), F.softmax(t/T), reduction='batchmean') * T**2`。
- **易错点**：① KL 项必须乘 `T²` 校正梯度量级，漏乘会让软标签几乎不起作用；② Temperature 并非越高越好，太高信息被过度平滑，分类任务常见最优在 T≈2~5；③ Teacher 前向要 `eval()` + `no_grad()`，否则白占显存还可能误更新。
- **Java 视角**：像"导师带新人"——新人不必重走导师全部踩坑路径，直接学经验总结；大模型时代的数据蒸馏（GPT-4 造数据训小模型）更像"用专家标注好的样本做培训"。
- **前置**：需理解交叉熵/softmax；数据蒸馏部分与阶段五微调（SFT）相通。

---

## 🎯 学习要点

- 量化是**降低部署成本最直接**的手段：先跑通 `bitsandbytes` 4-bit 加载 7B 模型，用 `torch.cuda.max_memory_allocated()` 亲手量出显存从 ~14GB 降到 ~5GB。
- 建立"精度—体积—质量"的直觉：FP16 ≈ INT8 > AWQ/GPTQ-INT4 ≳ GGUF-Q4；短问答几乎无差异，长链推理/数学/代码上 INT4 偶有退化——用固定采样（`do_sample=False`）做可复现对比。
- 会用**至少一种离线量化工具链**（GPTQModel 或 llm-compressor），并理解校准数据要贴近真实业务分布。
- 手算一次 KV Cache 显存，牢记 GQA 用 `n_kv_heads`；用 SDPA 替换手写 attention 并对比显存/速度。
- 理解 2026 主线：低比特已下探到 **FP8(E4M3/E5M2) / NVFP4** 浮点量化（Hopper/Blackwell 原生），质量显著优于同位宽整数量化。
- 蒸馏侧：跑通一个 `DistillationLoss` 分类蒸馏（Teacher→Student），并扫 T∈{2,5,10} 观察"平滑程度 vs 信息量"的权衡；理解现代主线是**数据蒸馏 + on-policy 蒸馏**。

---

## 🔗 关联

- **上一模块**：无（本模块为阶段八起点）
- **下一模块**：[02-model-serving](../02-model-serving/) — 把优化后的模型部署为高性能 API 服务（vLLM 的 PagedAttention 正是本模块推理加速思想的工业化落地）
- **阶段总览**：[阶段八 · 大模型部署与工程化](../README.md)
- **配套实战**：`agent-course/` [Day 51 成本优化](../../agent-course/Day-51-cost-optimization.md)、[Day 67 压测/缓存/延迟优化](../../agent-course/Day-67-load-cache-latency.md) — 从 Agent 视角看"量化/加速"在成本与延迟上的收益
