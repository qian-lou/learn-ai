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
model_name = "Qwen/Qwen2.5-7B"  # 开源免 gating 的现代基座（替代上代 Llama-2）
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
# trainable params: ~5M (~0.07%) / all params: ~7.6B（具体数值随模型与 target_modules 变化）

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
import torch
from transformers import BitsAndBytesConfig

# ============================================================
# QLoRA: 4-bit 量化 + LoRA
# 单卡 24GB 即可微调 7B 模型
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度（用 torch dtype，非字符串）
    bnb_4bit_use_double_quant=True,         # 双重量化（进一步压缩）
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

### 4.3 2024-2025 进展：unsloth · DoRA · all-linear

2024-2025 LoRA 生态有三个值得纳入默认配方的进展：**unsloth**（更快更省显存的训练引擎）、**DoRA/rsLoRA**（更强的 LoRA 变体）、以及 PEFT 的 **`all-linear`**（自动选层）。

```
unsloth：手写 Triton kernel 重写前向/反向 + 优化的 4-bit 路径
  • QLoRA 提速 2-5x，显存再降 ~60%（长上下文优势更明显）
  • 单卡能塞更大模型 / 更长序列；API 与 transformers+peft 基本兼容
  • 适合消费级 GPU（如 RTX 4090）跑 7B-14B QLoRA
```

```python
# unsloth 最小示例：加载即已 4-bit 量化，再套 LoRA / load 4-bit then add LoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",  # 官方预量化权重 / pre-quantized
    max_seq_length=2048,
    load_in_4bit=True,                          # QLoRA 路径 / 4-bit base
)
# unsloth 版 get_peft_model：内部用自家 kernel 加速
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules="all-linear",   # 见下，自动覆盖全部线性层 / auto-pick all linears
    use_dora=False,                # 也支持 DoRA
    use_gradient_checkpointing="unsloth",  # 自家梯度检查点，再省显存 / extra VRAM save
)
# 之后照常交给 trl 的 SFTTrainer 训练 / hand off to SFTTrainer as usual
```

```python
# DoRA / rsLoRA：标准 PEFT 里一个开关即可启用 / one flag each in vanilla PEFT
from peft import LoraConfig, TaskType

cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32,
    target_modules="all-linear",   # 当前推荐：自动匹配所有 nn.Linear（除输出头）
                                    # recommended: auto-cover every linear layer
    use_dora=True,                 # DoRA：权重分解 LoRA（幅度+方向分开学），低秩下更接近全参
                                   # DoRA: decompose weight into magnitude+direction
    use_rslora=True,               # rsLoRA：秩稳定缩放，用 alpha/√r 替代 alpha/r
                                   # rank-stabilized scaling, more stable at large r
)
# 说明 / notes:
#   • DoRA 精度更好但有额外计算开销；可与 QLoRA 叠加，推理前同样能 merge。
#   • all-linear 省去手写 target_modules，且对新架构更稳健（自动适配 q/k/v/o + MLP）。
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

*参考答案*：GPT-2 的注意力是合并的 `c_attn`，target_modules 要写它而非 q/v_proj。

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# GPT-2 用 c_attn（合并的 QKV 投影）/ GPT-2 uses fused c_attn, not q/v_proj
lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                  target_modules=["c_attn"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()  # 仅训练极少参数 / only a tiny fraction trainable
# 之后用 SFTTrainer/Trainer 在你的语料上训练，再对比生成结果
# Then train with SFTTrainer/Trainer and compare generations before/after
```

**练习 2：** 对比 r=4, r=16, r=64 的效果和训练速度。

*参考答案*：r 控制低秩矩阵的秩——越大可训练参数越多、拟合更强但更慢、易过拟合。

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

for r in (4, 16, 64):
    base = AutoModelForCausalLM.from_pretrained("gpt2")
    # lora_alpha 通常设为 2*r 以保持更新幅度稳定 / keep alpha≈2*r
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=2 * r,
                     target_modules=["c_attn"])
    m = get_peft_model(base, cfg)
    print(f"r={r}:", end=" "); m.print_trainable_parameters()
# 结论：r=8~16 性价比最高；r=64 收益递减且更慢 / r=8~16 best trade-off
```

### 进阶题

**练习 3：** 用 QLoRA 在单卡 GPU 上微调 Qwen-7B。

*参考答案*：4-bit 量化加载 + `prepare_model_for_kbit_training` + LoRA，单卡 24GB 即可。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

name = "Qwen/Qwen2.5-7B"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)  # 启用梯度检查点等 / enable grad checkpointing
tok = AutoTokenizer.from_pretrained(name)

lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora)

ds = load_dataset("json", data_files="sft.jsonl", split="train")
trainer = SFTTrainer(
    model=model, train_dataset=ds,
    args=SFTConfig(output_dir="./qlora_out", per_device_train_batch_size=4,
                   gradient_accumulation_steps=4, bf16=True, max_seq_length=1024),
    processing_class=tok)  # TRL 0.12+ 用 processing_class / use processing_class
trainer.train()
```

**练习 4：** 实现多个 LoRA adapter 切换：同一基座模型加载不同的 LoRA 完成不同任务。

*参考答案*：基座只加载一次，用 `load_adapter` 挂多个 adapter，`set_adapter` 运行时切换，显存高效。

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
# 首个 adapter 命名为 "translate" / first adapter named "translate"
model = PeftModel.from_pretrained(base, "./lora_translate", adapter_name="translate")
# 再挂载第二个，复用同一基座（省显存）/ attach a second, sharing the base weights
model.load_adapter("./lora_summary", adapter_name="summary")

model.set_adapter("translate")   # 切到翻译任务 / switch to translation
# ... 推理 ...
model.set_adapter("summary")     # 切到摘要任务 / switch to summarization
```
