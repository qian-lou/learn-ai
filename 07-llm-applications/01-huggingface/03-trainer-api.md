# Trainer API / HuggingFace Trainer

## 1. 背景（Background）

> **为什么要学这个？**
>
> HuggingFace Trainer 封装了完整的训练循环——梯度计算、优化器更新、学习率调度、分布式训练、混合精度、日志记录、模型保存——**一个配置搞定一切**。
>
> 对于 Java 工程师来说，Trainer 就像 **Spring Boot Test Runner**——你只需要定义模型和数据，训练框架负责其余所有事情。

## 2. 知识点（Key Concepts）

| 组件 | 功能 |
|------|------|
| `TrainingArguments` | 训练超参数配置 |
| `Trainer` | 训练循环引擎 |
| `compute_metrics` | 自定义评估指标 |
| `data_collator` | 批量数据整理 |
| WandB / TensorBoard | 训练可视化 |

## 3. 内容（Content）

### 3.1 BERT 文本分类完整流程

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 1. 加载数据和模型
# ============================================================
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ============================================================
# 2. 数据预处理
# ============================================================
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# 3. 评估指标
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# ============================================================
# 4. 训练配置
# ============================================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,                      # 混合精度
    logging_steps=100,
    report_to="tensorboard",
)

# ============================================================
# 5. 创建 Trainer 并训练
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
trainer.save_model("./best_model")
```

### 3.2 关键超参数

```
大模型微调常用超参数：

学习率: 1e-5 ~ 5e-5（BERT 微调）
        1e-4 ~ 3e-4（LoRA 微调）
Batch Size: 越大越好（受显存限制）
Epochs: 1-5（大模型 1-3 即可，防止过拟合）
Weight Decay: 0.01 ~ 0.1
Warmup: 前 5-10% 步用线性 warmup
FP16/BF16: 几乎总是开启（速度翻倍，显存减半）
```

## 4. 详细推理（Deep Dive）

### 4.1 Trainer 内部做了什么？

```
Trainer.train() 等价于：

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        with autocast(fp16):          # 混合精度
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()  # 梯度缩放
        
        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()           # 学习率调度
            optimizer.zero_grad()
        
        if step % logging_steps == 0:
            log_metrics(loss, lr)      # 日志
        
        if step % save_steps == 0:
            save_checkpoint()          # 保存
    
    if eval_strategy == "epoch":
        evaluate()                     # 评估
```

## 5. 例题（Worked Examples）

```python
# 用 Trainer 做 NER
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
# 同样的 Trainer API，只需换模型和数据
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 Trainer 微调 BERT 做 IMDB 情感分类，达到 90%+ 准确率。

**练习 2：** 添加 Early Stopping callback，防止过拟合。

### 进阶题

**练习 3：** 自定义 Trainer（继承 Trainer 类），重写 `compute_loss` 方法实现 focal loss。

**练习 4：** 用 Trainer + LoRA 微调一个 7B 模型做分类任务。
