# HuggingFace Transformers 库 / HuggingFace Transformers

## 1. 背景（Background）

> **为什么要学这个？**
>
> HuggingFace 是大模型生态的 **"Maven Central"**——它提供了预训练模型、分词器、数据集和训练工具的统一接口。超过 50 万个模型、10 万个数据集托管在 HuggingFace Hub 上。掌握 HuggingFace，你就能快速使用几乎所有开源大模型。
>
> 对于 Java 工程师来说，HuggingFace 就是 AI 界的 **Spring Boot + Maven**——`Auto` 类系列自动推断模型类型并加载，就像 Spring 的自动配置。
>
> **在整个体系中的位置：** HuggingFace 是使用大模型的第一个实用工具，贯穿推理、微调、评估全流程。

## 2. 知识点（Key Concepts）

| 组件 | 功能 | Java 类比 |
|------|------|----------|
| `AutoTokenizer` | 自动加载分词器 | JSON Parser |
| `AutoModel` | 自动加载模型 | Spring Bean |
| `pipeline` | 一行代码搞定任务 | Spring Boot Starter |
| `Trainer` | 封装训练循环 | JUnit Runner |
| `Hub` | 模型仓库 | Maven Central |

## 3. 内容（Content）

### 3.1 Pipeline API（最快上手方式）

```python
from transformers import pipeline

# ============================================================
# Pipeline: 一行代码完成各种 NLP 任务
# Pipeline: One-liner for various NLP tasks
# ============================================================

# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_new_tokens=50, do_sample=True)
print(result[0]['generated_text'])

# 情感分析
classifier = pipeline("sentiment-analysis")
print(classifier("I love this product!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 问答
qa = pipeline("question-answering")
context = "HuggingFace was founded in 2016 in New York."
print(qa(question="Where was HuggingFace founded?", context=context))

# 翻译
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
print(translator("Hello, how are you?"))

# 零样本分类
zero_shot = pipeline("zero-shot-classification")
result = zero_shot("I want to buy a new laptop", candidate_labels=["shopping", "travel", "food"])
print(result)
```

### 3.2 手动加载模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================================
# 手动模式：更灵活的控制
# Manual mode: More flexible control
# ============================================================

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 编码
text = "Artificial intelligence"
inputs = tokenizer(text, return_tensors="pt")
print(f"input_ids: {inputs['input_ids']}")

# 生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# ============================================================
# 加载大模型的优化方式
# Optimized loading for large models
# ============================================================

# 半精度加载（显存减半）
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 4-bit 量化加载（QLoRA 需要）
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,
#     device_map="auto",
# )
```

### 3.3 Auto 类家族

```
Auto 类自动推断模型类型：

AutoModelForCausalLM       → GPT 等自回归模型
AutoModelForSeq2SeqLM      → T5 等编码器-解码器模型  
AutoModelForSequenceClassification → 分类任务
AutoModelForTokenClassification    → NER 任务
AutoModelForQuestionAnswering      → 问答任务
AutoModel                  → 基础模型（特征提取）

只需要改 from_pretrained 的模型名即可切换模型
→ GPT-2, LLaMA, Qwen, Mistral 用同一套代码
```

## 4. 详细推理（Deep Dive）

### 4.1 HuggingFace 生态全景

```
HuggingFace 核心库：
  transformers:  模型 + 分词器
  datasets:      数据集加载和处理
  peft:          参数高效微调（LoRA）
  trl:           强化学习训练（RLHF/DPO）
  accelerate:    分布式训练
  evaluate:      评估指标
  safetensors:   安全的模型存储格式
  gradio:        快速 UI 原型
```

## 5. 例题（Worked Examples）

### 例题：模型推理 + 流式输出

```python
from transformers import TextStreamer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 流式输出（像 ChatGPT 一样逐字打印）
streamer = TextStreamer(tokenizer)
inputs = tokenizer("Once upon a time", return_tensors="pt")
model.generate(**inputs, max_new_tokens=100, streamer=streamer)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 pipeline 完成文本生成、情感分析、翻译三个任务。

**练习 2：** 手动加载 GPT-2，对比 `temperature=0.1` 和 `temperature=1.5` 的生成效果。

### 进阶题

**练习 3：** 用 `device_map="auto"` 加载一个 7B 模型，体验自动设备分配。

**练习 4：** 用 `AutoModelForSequenceClassification` 对 BERT 做情感分类微调。
