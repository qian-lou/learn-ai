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
# 启动 OpenAI 兼容的 API 服务器
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
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
    --max-input-length 4096 \
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
**练习 2：** 用 Ollama 在本地运行 Qwen-7B。

### 进阶题
**练习 3：** 对比 vLLM 和 HuggingFace 原生推理的吞吐量。
**练习 4：** 配置 vLLM 的 tensor parallelism 在多 GPU 上部署。
