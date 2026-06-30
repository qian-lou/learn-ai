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

## 5. 例题（Worked Examples）

### 例题 1：使用 MLflow 追踪模型超参数与训练 Loss 曲线 / Tracking parameters and metrics with MLflow

大模型微调需要频繁尝试不同的学习率和批大小。本例演示如何使用 MLflow 自动化记录这些实验数据。

```python
import mlflow
import numpy as np

# 1. 设置实验名称 / Set MLflow experiment name
mlflow.set_experiment("LoRA-Finetune-Experiments")

# 2. 模拟训练与日志追踪 / Start run and log parameters
# Time: O(Steps), Space: O(1)
with mlflow.start_run(run_name="run_lr_2e-5_r_8"):
    # 记录超参数 / Log hyperparameters
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("lora_rank", 8)
    mlflow.log_param("model_name", "llama-3-8b")
    
    # 模拟 10 个 Epoch 损失下降 / Simulate training steps
    for epoch in range(1, 11):
        loss = 2.5 / (epoch ** 0.5) + np.random.randn() * 0.05
        # 记录每个步骤的损失 / Log training metrics
        mlflow.log_metric("loss", loss, step=epoch)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
    # 保存模型元数据标签 / Set tags
    mlflow.set_tag("framework", "peft")
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在模型实验管理中，什么是超参数（Hyperparameter）和评估指标（Metric）？MLflow 中记录两者的 API 有什么本质区别？
*参考答案*：
- **超参数（Hyperparameter）**：在训练开始前设定的参数（如学习率、网络层数、Batch Size），在训练过程中通常是不变的值。MLflow 使用 `mlflow.log_param()` 记录。
- **评估指标（Metric）**：在训练过程中或训练结束后随步骤变化的观测指标（如 Loss、Accuracy、BLEU 分数）。MLflow 使用 `mlflow.log_metric()` 记录，且支持传入 `step` 参数来绘制时序变化曲线。

### 进阶题
**练习 2**：在分布式多节点训练中，如何设置 MLflow 的远程追踪服务端（Tracking Server），以便让各个计算卡节点都能将日志指标统一打入集中的 PostgreSQL 数据库中？
*参考答案*：
需要在启动 MLflow 远程服务时指定后端数据库连接 URI 以及文件存储路径：
```bash
# mlflow server --backend-store-uri postgresql://user:pwd@db-host:5432/mlflow --default-artifact-root s3://my-mlflow-bucket/ --host 0.0.0.0 --port 5000
```
然后在 Python 代码中配置环境变量，使 SDK 指向该服务：
```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://db-host:5000"
```\n