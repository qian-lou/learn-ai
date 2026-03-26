# Trainer API / HuggingFace Trainer

## 1. 背景（Background）
> Trainer 封装了训练循环，只需定义模型、数据、参数即可开始训练，极大降低了代码量。

## 2-3. 知识点与内容
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

## 4-6. 推理/例题/习题
**练习：** 用 Trainer 微调 BERT 做文本分类。
