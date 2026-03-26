# 实验管理 / Experiment Tracking

## 1. 背景（Background）

> **为什么要学这个？**
>
> ML 实验管理跟踪超参数、指标和模型版本。当你跑了 50 次 LoRA 微调实验，用不同的 rank、learning rate、数据集——如何找到最优配置？Weights & Biases (W&B) 和 MLflow 就是解决这个问题的。
>
> 对于 Java 工程师来说，这就像 **ELK 日志系统 + 配置管理**——记录每次运行的参数和结果，支持对比分析。

## 2. 知识点（Key Concepts）

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| W&B | 云端托管，可视化强 | 团队协作 |
| MLflow | 开源自部署 | 企业内部 |
| TensorBoard | PyTorch/TF 原生 | 个人实验 |

## 3. 内容（Content）

### 3.1 Weights & Biases

```python
import wandb

# ============================================================
# W&B 实验跟踪 / Experiment tracking
# ============================================================
wandb.init(
    project="llm-finetune",
    name="lora-r16-lr1e4",
    config={
        "model": "qwen-7b",
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "epochs": 3,
        "batch_size": 4,
    }
)

for epoch in range(3):
    train_loss = train(model, dataloader)
    val_loss = evaluate(model, val_dataloader)
    
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch,
        "learning_rate": scheduler.get_last_lr()[0],
    })

# 保存模型到 W&B
wandb.save("best_model.pt")
wandb.finish()
```

### 3.2 HuggingFace Trainer 集成

```python
from transformers import TrainingArguments

# 一行集成 W&B
training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",          # 自动记录所有指标
    run_name="bert-imdb-v1",
    logging_steps=10,
)
# Trainer 会自动记录 loss、lr、throughput 等
```

### 3.3 MLflow（自部署方案）

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-finetune")

with mlflow.start_run(run_name="lora-r16"):
    mlflow.log_params({"lr": 1e-4, "rank": 16})
    mlflow.log_metrics({"loss": 0.5, "accuracy": 0.92})
    mlflow.pytorch.log_model(model, "model")
```

## 4. 详细推理（Deep Dive）

```
实验管理最佳实践：
  1. 每次实验都记录完整配置（可复现）
  2. 用 sweep 自动搜索超参数
  3. 用 tag 标记实验类型
  4. 团队共享实验结果面板
```

## 5-6. 例题/习题

**练习 1：** 用 W&B 跟踪一次 LoRA 微调，对比 r=4, 8, 16 的效果。

**练习 2：** 用 W&B Sweep 自动搜索最佳学习率。

**练习 3：** 搭建本地 MLflow 服务，记录和对比多次实验。
