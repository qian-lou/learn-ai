# vLLM/TGI 部署 / Model Serving with vLLM/TGI

## 1. 背景（Background）
> vLLM 使用 PagedAttention 实现高吞吐推理，TGI 是 HuggingFace 的推理服务器。这是大模型上线的关键环节。

## 2-3. 知识点与内容
```python
# vLLM — 高吞吐推理引擎
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-7B-Instruct")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["你好世界"], params)

# vLLM 作为 API 服务
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct

# TGI (Text Generation Inference) — Docker 方式
# docker run -p 8080:80 ghcr.io/huggingface/text-generation-inference --model-id model
```

## 4-6. 推理/例题/习题
**练习：** 用 vLLM 部署一个兼容 OpenAI API 的推理服务。
