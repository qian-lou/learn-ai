# 指令微调实战 / Instruction Fine-tuning Practice

## 1. 背景（Background）
> 指令微调让基座模型学会遵循指令（对话、问答）。SFT 数据格式和质量至关重要。

## 2-3. 知识点与内容
```python
# SFT 数据格式（Alpaca 格式）
# {"instruction": "翻译成英文", "input": "你好", "output": "Hello"}

from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("json", data_files="sft_data.jsonl")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)
trainer.train()
```

## 4-6. 推理/例题/习题
**练习：** 准备指令微调数据集，用 LoRA + SFTTrainer 微调一个对话模型。
