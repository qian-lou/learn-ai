# 实验管理（Weights & Biases）/ Experiment Tracking

## 1. 背景（Background）
> W&B/MLflow 跟踪实验参数、指标和模型版本。类似 Java 的日志系统，但专门为 ML 实验设计。

## 2-3. 知识点与内容
```python
import wandb

wandb.init(project="llm-finetune", config={"lr": 1e-4, "epochs": 3, "model": "qwen-7b"})

for epoch in range(num_epochs):
    train_loss = train(model, dataloader)
    val_loss = evaluate(model, val_dataloader)
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

wandb.finish()

# HuggingFace Trainer 集成
training_args = TrainingArguments(report_to="wandb", ...)
```

## 4-6. 推理/例题/习题
**练习：** 用 W&B 跟踪一次 LoRA 微调实验，对比不同 rank 的效果。
