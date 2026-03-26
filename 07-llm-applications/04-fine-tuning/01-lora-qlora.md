# LoRA/QLoRA 微调 / LoRA and QLoRA Fine-tuning

## 1. 背景（Background）

> **为什么要学这个？**
>
> 全参数微调 7B 模型需要 **~28GB 显存 + 56GB 优化器状态**——普通开发者根本做不到。LoRA（Low-Rank Adaptation）通过**低秩分解**，只训练 **0.1%-1%** 的参数就能达到接近全量微调的效果。QLoRA 进一步用 4-bit 量化，让你用**单张消费级 GPU（如 RTX 3090 24GB）** 微调 7B 甚至 13B 模型。
>
> 对于 Java 工程师来说，LoRA 就像 **AOP（面向切面编程）**——不修改原始类（原始权重冻结），通过添加切面逻辑（低秩矩阵）来改变行为。

## 2. 知识点（Key Concepts）

| 方法 | 可训练参数 | 7B 显存需求 | 效果 |
|------|-----------|-------------|------|
| 全参微调 | 100% | ~112 GB | 最好 |
| LoRA (r=16) | ~0.1% | ~16 GB | 接近全参 |
| QLoRA (4bit+LoRA) | ~0.1% | ~6 GB | 接近 LoRA |

## 3. 内容（Content）

### 3.1 LoRA 核心原理

```
LoRA 数学原理：

原始前向传播: h = W·x     (W ∈ R^{d×k})
LoRA 前向传播: h = W·x + BA·x

W: 原始权重 (冻结，不训练)       — 如 [4096, 4096]
B: 降维矩阵 (训练)  — [4096, r]   r=16 远小于 4096
A: 升维矩阵 (训练)  — [r, 4096]

参数量对比:
  原始: 4096 × 4096 = 16.7M
  LoRA: 4096 × 16 + 16 × 4096 = 131K (减少 99.2%)

初始化:
  A: 随机高斯
  B: 全零 → 初始时 BA = 0，模型行为不变

合并推理 (无额外开销):
  W_merged = W + BA → 推理时与原始模型速度一样！
```

### 3.2 LoRA 实战

```python
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. 加载基座模型
# ============================================================
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# 2. 配置 LoRA
# ============================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                           # 秩（越大拟合能力越强）
    lora_alpha=32,                  # 缩放因子 (alpha/r 控制更新幅度)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    # 对 Attention 的 Q/K/V/O 投影层加 LoRA
)

# ============================================================
# 3. 应用 LoRA
# ============================================================
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 (0.06%) / all params: 6,742,609,920

# ============================================================
# 4. 训练（使用 Trainer 或 SFTTrainer）
# ============================================================
# trainer = SFTTrainer(model=model, ...)
# trainer.train()

# ============================================================
# 5. 保存和加载 LoRA 权重
# ============================================================
model.save_pretrained("./lora_adapter")
# 只保存 LoRA 权重（~几十 MB），而非整个模型（~13 GB）

# 加载
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 合并为完整模型（推理部署用）
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

### 3.3 QLoRA（4-bit 量化 + LoRA）

```python
from transformers import BitsAndBytesConfig

# ============================================================
# QLoRA: 4-bit 量化 + LoRA
# 单卡 24GB 即可微调 7B 模型
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 量化
    bnb_4bit_compute_dtype="bfloat16",  # 计算精度
    bnb_4bit_use_double_quant=True,     # 双重量化（进一步压缩）
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 然后正常应用 LoRA
model = get_peft_model(model, lora_config)
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么低秩有效？

```
研究发现：大模型权重矩阵的"有效秩"远小于其维度

  4096×4096 的权重矩阵，实际有效秩可能只有 16-64
  → 权重变化主要发生在低维子空间
  → LoRA 用 r=16 就能捕获大部分变化

r 值选择：
  r=4:  极致压缩，简单任务
  r=8:  常用，性价比最高
  r=16: 推荐，效果稳定
  r=64: 接近全参微调，复杂任务
```

### 4.2 target_modules 选择

```
哪些模块加 LoRA？

最小:  q_proj, v_proj            — 够用
推荐:  q_proj, k_proj, v_proj, o_proj — 更好
全面:  还包括 gate_proj, up_proj, down_proj — 最好但更贵
```

## 5. 例题（Worked Examples）

```python
# 快速验证 LoRA 效果
from peft import LoraConfig, get_peft_model
import torch.nn as nn

# 小模型示例
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
config = LoraConfig(r=8, target_modules=["linear"])
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 LoRA 微调 GPT-2 做文本生成，观察微调前后效果差异。

**练习 2：** 对比 r=4, r=16, r=64 的效果和训练速度。

### 进阶题

**练习 3：** 用 QLoRA 在单卡 GPU 上微调 Qwen-7B。

**练习 4：** 实现多个 LoRA adapter 切换：同一基座模型加载不同的 LoRA 完成不同任务。
