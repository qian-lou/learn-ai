# HuggingFace Transformers 库 / HuggingFace Transformers

## 1. 背景（Background）
> HuggingFace 是大模型的"Maven Central"——提供预训练模型、分词器、数据集的统一接口。掌握它就能快速使用各种大模型。

## 2-3. 知识点与内容
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Pipeline API（最简方式）
generator = pipeline("text-generation", model="gpt2")
result = generator("AI is", max_length=50)

# 手动使用分词器和模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 4-6. 推理/例题/习题
**练习：** 用 HuggingFace 加载一个中文模型，完成文本生成任务。
