# vLLM/TGI 部署 / Model Serving with vLLM/TGI

## 1. 背景（Background）

> **为什么要学这个？**
>
> vLLM 和 TGI 是大模型推理的**工业级引擎**。vLLM 的 PagedAttention 可将吞吐量提升 **10-24 倍**。它们提供 OpenAI 兼容 API，让你无缝切换到自部署模型。
>
> 对于 Java 工程师来说，vLLM 就像 **Tomcat/Undertow**——高性能推理服务器。

## 2. 知识点（Key Concepts）

| 引擎 | 特点 | 适用场景 |
|------|------|---------|
| vLLM | PagedAttention，极高吞吐 | 生产首选 |
| TGI | HuggingFace，Docker 友好 | 快速部署 |
| Ollama | 极简本地部署 | 开发测试 |

## 3. 内容（Content）

### 3.1 vLLM 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["What is AI?", "Explain Python"], params)
for out in outputs:
    print(out.outputs[0].text)
```

### 3.2 vLLM 作为 API 服务

```bash
# 启动 OpenAI 兼容的 API 服务器（2026 推荐 vllm serve 子命令，见 3.5）
# 旧写法 python -m vllm.entrypoints.openai.api_server 仍可用但已不推荐
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

```python
# 用 OpenAI SDK 调用（兼容接口）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### 3.3 TGI Docker 部署

```bash
# HuggingFace TGI
docker run --gpus all -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --max-input-tokens 4096 \
    --max-total-tokens 8192
```

### 3.4 PagedAttention 原理

```
vLLM 的核心创新 — PagedAttention:

传统方式: 为每个请求预分配最大长度的 KV Cache
  → 大量显存浪费（实际生成长度 << max_length）

PagedAttention: 像操作系统的虚拟内存一样管理 KV Cache
  → 按需分配固定大小的"页"
  → 不同请求可以共享相同前缀的页（Prefix Caching）
  → 显存利用率从 ~30% → 90%+
  → 吞吐量提升 2-4x
```

### 3.5 vLLM V1 引擎与 2024-2025 特性

> 2025 年 vLLM 全面切换到 **V1 引擎**（默认）。V1 重写了调度器：用零拷贝 DMA 替代 GPU↔CPU 张量搬运，prefill/decode 统一按 token 预算调度，高并发下调度开销大幅下降。
> The V1 engine (now default) rewrote the scheduler — zero-copy DMA + a unified token-budget schedule for prefill/decode — cutting overhead at high concurrency.

```bash
# 现代启动方式：vllm serve 子命令（替代旧的 python -m vllm.entrypoints.openai.api_server）
# Modern entry point: the `vllm serve` CLI (replaces the old api_server module)
# --enable-prefix-caching：V1 默认开，命中相同前缀直接复用 KV / on by default in V1
# --tensor-parallel-size：张量并行卡数，须整除注意力头数 / must divide #attn heads
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

```python
# OpenAI 兼容客户端：与官方 SDK 完全同构 / OpenAI-compatible client, drop-in
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "用一句话解释 PagedAttention"}],
    # 结构化输出：约束模型必须吐合法 JSON / structured outputs: force valid JSON
    extra_body={"guided_json": {"type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"]}},
)
print(resp.choices[0].message.content)
```

V1 关键特性一览 / V1 feature map：
- **PagedAttention**：分页式 KV Cache，显存利用率 90%+（见 3.4）/ paged KV cache。
- **Prefix caching**：`--enable-prefix-caching`，V1 重写后近零开销，**默认开启** / near-zero overhead, default on。
- **Continuous batching**：逐 token 动态组批，无需等整批 / per-token dynamic batching。
- **Speculative decoding**：投机解码，V1 支持 ngram 与 EAGLE/MTP draft，可降延迟 1.5-3x / draft-then-verify。
- **Structured outputs**：`guided_json` / `guided_regex` / `guided_grammar`，默认 `auto` 后端（xgrammar）/ constrained decoding。
- **Tool calling**：`--enable-auto-tool-choice --tool-call-parser hermes` 暴露 OpenAI 兼容函数调用 / function calling。
- **TP / PP**：`--tensor-parallel-size`（层内切分）+ `--pipeline-parallel-size`（跨节点流水）/ tensor & pipeline parallel。

一句话横向对比 / one-liner landscape：吞吐通用首选 **vLLM**；**SGLang** 的 RadixAttention 在多轮/Agent 前缀复用更激进；**TensorRT-LLM** 在 NVIDIA 卡上做极致 kernel 编译、延迟最低但部署最重；**LMDeploy** 的 TurboMind 对国产/量化模型（含 W4A16）开箱体验佳。

## 4. 详细推理（Deep Dive）

### 4.1 选型对比

```
vLLM:  吞吐量最高，生态最好，支持各种模型
TGI:   HuggingFace 集成好，Docker 部署简单
Ollama: 一行命令跑模型，适合个人开发
SGLang: 结构化/约束生成场景最优
```

## 5. 例题（Worked Examples）

```python
# 批量推理性能测试
import time
prompts = [f"Question {i}: What is AI?" for i in range(100)]
start = time.time()
outputs = llm.generate(prompts, params)
elapsed = time.time() - start
print(f"100 请求耗时: {elapsed:.1f}s, 吞吐: {100/elapsed:.1f} req/s")
```

## 6. 习题（Exercises）

### 基础题
**练习 1：** 用 vLLM 部署一个 OpenAI 兼容的 API 服务。

*参考答案*：

2026 年首选 `vllm serve` 子命令（比旧的 `python -m vllm.entrypoints.openai.api_server` 更简洁）：

```bash
# 启动 OpenAI 兼容服务 / launch OpenAI-compatible server
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching          # 复用相同前缀的 KV，省算力 / reuse shared-prefix KV
```

```bash
# 验证：标准 /v1/chat/completions 接口 / verify with the standard endpoint
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hi"}]}'
```

底层自动启用 PagedAttention 与 continuous batching，无需手动配置。

**练习 2：** 用 Ollama 在本地运行 Qwen-7B。

*参考答案*：

```bash
# 拉取并交互运行（首次自动下载 GGUF 权重）/ pull & run (auto-downloads GGUF on first use)
ollama run qwen2.5:7b
# 后台服务模式同样暴露 OpenAI 兼容接口于 http://localhost:11434/v1
# Server mode also exposes an OpenAI-compatible API at :11434/v1
```

要点：Ollama 适合本地开发，开箱即用、自动量化（默认 Q4_K_M）；高并发生产仍应选 vLLM。可用 `ollama list` 查看已装模型。

### 进阶题
**练习 3：** 对比 vLLM 和 HuggingFace 原生推理的吞吐量。

*参考答案*：

同一模型、同一批 prompts，分别计时并算 req/s（或 tokens/s）：

```python
import time
prompts = [f"Question {i}: What is AI?" for i in range(200)]

# vLLM：内置 continuous batching，一次 generate 全量
t = time.time(); llm.generate(prompts, params)
print("vLLM req/s:", len(prompts) / (time.time() - t))

# HF 原生：逐条或小 batch model.generate，无动态组批
# HF native: per-sample/small-batch generate, no continuous batching
```

预期：vLLM 凭 PagedAttention + continuous batching，高并发下吞吐通常领先 HF 原生 **数倍到一个数量级**。公平起见两边精度/采样参数一致，并丢弃首次预热。

**练习 4：** 配置 vLLM 的 tensor parallelism 在多 GPU 上部署。

*参考答案*：

```bash
# 张量并行：把每层权重切到 N 张卡，N 一般取 GPU 数且需能整除注意力头数
# Tensor parallelism: shard each layer across N GPUs (N must divide #attn heads)
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
# 离线 API 同理：LLM(model=..., tensor_parallel_size=4)
```

要点：(1) `tensor-parallel-size` 必须能整除模型注意力头数，否则报错；(2) 多机再叠加 `--pipeline-parallel-size`；(3) TP 用于单卡放不下的大模型（如 72B），同节点优先保证高速 NVLink 互联以降低通信开销。
