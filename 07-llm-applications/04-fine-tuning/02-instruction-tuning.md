# 指令微调实战 / Instruction Fine-tuning Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 指令微调（Instruction Tuning / SFT）是将基座模型（如 LLaMA）变成对话模型（如 ChatGPT）的关键步骤。基座模型只会"续写"，不会"对话"——指令微调教会它理解和遵循指令。
>
> 对于 Java 工程师来说，指令微调就像是**给框架写 Controller**——基座模型是框架（Spring），微调数据是路由配置（@RequestMapping），告诉模型如何响应不同类型的请求。

## 2. 知识点（Key Concepts）

| 数据格式 | 代表 | 特点 |
|----------|------|------|
| Alpaca | Stanford | instruction/input/output 三字段 |
| ShareGPT | 社区 | 多轮对话 conversations |
| OpenAI | OpenAI | messages 数组 (system/user/assistant) |

## 3. 内容（Content）

### 3.1 SFT 数据格式

```python
# ============================================================
# Alpaca 格式（最常用）
# ============================================================
alpaca_data = {
    "instruction": "将以下文本翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is nice today.",
}

# 转换为训练文本
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


# ============================================================
# ShareGPT / 多轮对话格式
# ============================================================
sharegpt_data = {
    "conversations": [
        {"from": "human", "value": "你好，请介绍一下机器学习"},
        {"from": "gpt", "value": "机器学习是人工智能的一个分支..."},
        {"from": "human", "value": "它和深度学习有什么区别？"},
        {"from": "gpt", "value": "深度学习是机器学习的一个子集..."},
    ]
}


# ============================================================
# OpenAI Messages 格式
# ============================================================
openai_data = {
    "messages": [
        {"role": "system", "content": "你是一个有用的AI助手"},
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种..."},
    ]
}
```

### 3.2 完整 SFT 流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================
# 1. 加载模型 + LoRA
# ============================================================
model_name = "Qwen/Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

# ============================================================
# 2. 准备数据
# ============================================================
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

def format_instruction(example):
    """将 Alpaca 格式转换为训练文本."""
    if example.get("input"):
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_instruction)

# ============================================================
# 3. 训练
# ============================================================
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    max_seq_length=2048,
    logging_steps=10,
    save_steps=200,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### 3.3 数据质量要诀

```
SFT 数据质量 > 数量:

  1000 条高质量数据 > 100000 条低质量数据

高质量数据标准:
  1. 指令清晰明确
  2. 回答准确完整
  3. 格式一致
  4. 覆盖多种任务类型
  5. 语言自然流畅

数据来源:
  - 人工编写（最贵但最好）
  - GPT-4 生成（Self-Instruct / Evol-Instruct）
  - 开源数据集（Alpaca, ShareGPT, UltraChat）
```

## 4. 详细推理（Deep Dive）

### 4.1 SFT 的损失函数

```
SFT 只对 Response 部分计算损失:

  Instruction tokens: [不计算损失]
  Response tokens:    [计算 CrossEntropyLoss]

为什么？
  我们不希望模型"学会提问"，只希望它"学会回答"
  → 对 instruction 部分的 labels 设为 -100（忽略）
```

## 5. 例题（Worked Examples）

```python
# 用 GPT-4 生成 SFT 数据
def generate_sft_data(task_description, n_samples=10):
    prompt = f"""生成 {n_samples} 条指令微调数据，任务类型：{task_description}
    格式：JSONL，每行一个 JSON 对象：
    {{"instruction": "...", "input": "...", "output": "..."}}"""
    # response = gpt4.invoke(prompt)
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 准备 100 条高质量指令微调数据，覆盖翻译、摘要、问答三种任务。

**练习 2：** 用 LoRA + SFTTrainer 微调 GPT-2，让它学会遵循指令。

### 进阶题

**练习 3：** 实现 Self-Instruct：用一个已有模型生成训练数据，微调另一个模型。

**练习 4：** 对比不同数据量（100/1000/10000 条）对 SFT 效果的影响。
