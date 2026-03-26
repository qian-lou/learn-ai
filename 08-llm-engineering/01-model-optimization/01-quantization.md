# 模型量化 / Model Quantization

## 1. 背景（Background）
> 量化将模型权重从 FP32/FP16 压缩到 INT8/INT4，大幅减少显存和推理延迟。让大模型能在消费级硬件上运行。

## 2-3. 知识点与内容
```python
# GPTQ 量化
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model-gptq", device_map="auto")

# bitsandbytes 4-bit 量化（QLoRA 基础）
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained("model", quantization_config=bnb_config)

# GGUF 格式（llama.cpp 使用，纯 CPU 推理）
# 常见规格：Q4_K_M（推荐）、Q5_K_M、Q8_0
```

## 4. 详细推理
- FP16→INT8: 模型体积减半，速度提升 2x
- FP16→INT4: 模型体积减 75%，可在 8GB 显存运行 7B 模型

## 5-6. 例题/习题
**练习：** 对比 FP16/INT8/INT4 模型的推理速度和输出质量。
