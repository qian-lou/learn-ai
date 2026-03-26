# LoRA/QLoRA 微调 / LoRA and QLoRA Fine-tuning

## 1. 背景（Background）
> 全参数微调 7B 模型需要 > 28GB 显存。LoRA 通过低秩分解，只训练 0.1% 参数即可达到接近全量微调的效果。QLoRA 进一步用 4-bit 量化降低显存需求。

## 2-3. 知识点与内容
```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 核心思想: W' = W + BA
# W: [d, k] 原始权重（冻结）
# B: [d, r] 低秩矩阵（训练）
# A: [r, k] 低秩矩阵（训练）
# r << min(d, k)，如 r=16

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # 秩
    lora_alpha=32,     # 缩放因子
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 只对 Q/V 投影层加 LoRA
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 约 0.1% 参数可训练
```

## 4. 详细推理
- 7B 模型全参微调：~28GB 显存 → LoRA: ~8GB → QLoRA(4bit): ~4GB
- LoRA 合并后推理无额外开销：`W_merged = W + BA`

## 5-6. 例题/习题
**练习：** 用 QLoRA 在消费级 GPU 上微调 Qwen-7B。
