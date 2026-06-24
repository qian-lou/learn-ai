# GPT 系列 / GPT Series (GPT-1/2/3/4)

## 1. 背景（Background）

> **为什么要学这个？**
>
> GPT 系列是大模型时代的**定义者**。从 GPT-1 的 117M 参数到 GPT-4 的巨型规模（官方未公开参数量），OpenAI 证明了一个惊人的结论：**只要规模足够大，简单的自回归语言模型就能涌现出惊人的智能**。
>
> GPT 对技术栈的影响：Prompt Engineering、Few-shot Learning、RLHF、Chain-of-Thought 等核心技术都因 GPT 系列而生。
>
> **在整个体系中的位置：** GPT 是 Decoder-only 架构的代表，也是当前大模型的主流路线（LLaMA、Qwen、Claude 都是 Decoder-only）。

## 2. 知识点（Key Concepts）

| 模型 | 年份 | 参数量 | 核心贡献 |
|------|------|--------|---------|
| GPT-1 | 2018 | 117M | 预训练 + 微调（与 BERT 同期） |
| GPT-2 | 2019 | 1.5B | Zero-shot 能力，拒绝发布 |
| GPT-3 | 2020 | 175B | Few-shot / In-Context Learning |
| InstructGPT | 2022 | 1.3B | RLHF 人类对齐 |
| GPT-4 | 2023 | 官方未公开（传闻 ~1.7T MoE） | 多模态，强推理 |

## 3. 内容（Content）

### 3.1 GPT 的核心思想

```
GPT 的一切都基于一个极简任务：预测下一个 token

  P(text) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂) × ...

训练目标: 最大化 Σ log P(tᵢ | t₁, t₂, ..., tᵢ₋₁)

这个任务看似简单，但规模够大后：
  - 学会了语法 ✅
  - 学会了事实知识 ✅
  - 学会了推理（涌现能力）✅
  - 学会了编程 ✅
  - 学会了数学 ✅
```

### 3.2 GPT 系列技术演进

```
GPT-1 → GPT-2: "规模的力量"
  从 117M → 1.5B
  发现: 不需要微调！只靠提示（prompt）就能做任务
  → Zero-shot Learning 的发现

GPT-2 → GPT-3: "In-Context Learning"
  从 1.5B → 175B
  发现: 在 prompt 中给几个例子，模型就能学会新任务
  → Few-shot Learning
  
  示例:
    Translate English to French:
    sea otter => loutre de mer
    cheese => fromage
    plush giraffe =>
    
  GPT-3 输出: girafe en peluche ✅

GPT-3 → InstructGPT/ChatGPT: "对齐"
  发现: 原始 GPT-3 不懂拒绝有害请求，也不擅长对话
  解决: SFT + RLHF → InstructGPT → ChatGPT
  
GPT-3.5 → GPT-4: "多模态 + 推理"
  支持图像输入
  推理能力大幅增强（律师考试 top 10%）
  可能使用 MoE（Mixture of Experts）架构
```

### 3.3 使用 GPT-2 生成文本

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# GPT-2 文本生成
# GPT-2 text generation
# ============================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# 基础生成
prompt = "Artificial intelligence will"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)


# ============================================================
# In-Context Learning（少样本学习）
# In-Context Learning (Few-shot)
# ============================================================
few_shot_prompt = """Classify the sentiment:
Text: "I love this movie!" -> Positive
Text: "Terrible experience." -> Negative
Text: "The food was amazing!" ->"""

inputs = tokenizer(few_shot_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.4 GPT 架构细节

```
GPT-3 架构参数:
  层数: 96
  d_model: 12288
  n_heads: 96
  d_head: 128
  d_ff: 49152 (4 × d_model)
  vocab_size: 50257
  max_len: 2048
  参数量: 175B

训练资源:
  数据: 300B tokens (Common Crawl + Books + Wikipedia)
  硬件: 10000 V100 GPU
  训练时间: ~34 天
  成本: ~$4.6M（2020 年价格）
```

## 4. 详细推理（Deep Dive）

### 4.1 涌现能力（Emergent Abilities）

```
涌现: 小模型完全没有，大模型突然获得的能力

  算术:   GPT-3 (175B) 能做 2 位数加法
          GPT-2 (1.5B) 完全不行
  
  翻译:   GPT-3 的 few-shot 翻译接近专门的翻译模型
  
  推理:   Chain-of-Thought (CoT) 只在 100B+ 模型上有效
          "Let's think step by step" 魔法提示

  编程:   GPT-3 → Codex → GitHub Copilot

这些能力不是设计出来的，是规模"涌现"出来的
→ Scaling Laws 的核心验证
```

### 4.2 GPT vs 开源大模型

```
闭源: GPT-4, Claude, Gemini
  优势: 效果最强, API 易用
  劣势: 价格贵, 数据安全, 不可定制

开源: LLaMA, Qwen, Mistral, DeepSeek
  优势: 可微调, 可私有部署, 成本可控
  劣势: 效果稍弱（但差距在缩小）
  
实际选择:
  原型验证 → GPT-4 API
  生产部署 → 开源模型微调
  Java 工程师 → 用 API 做应用开发，理解原理做技术选型
```

## 5. 例题（Worked Examples）

### 例题：对比不同大小 GPT-2 的生成质量

```python
models = ["gpt2", "gpt2-medium", "gpt2-large"]  # 124M, 355M, 774M
for model_name in models:
    tok = GPT2Tokenizer.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name)
    inp = tok("The future of AI is", return_tensors="pt")
    out = mdl.generate(**inp, max_new_tokens=30, do_sample=True, temperature=0.7)
    print(f"{model_name}: {tok.decode(out[0], skip_special_tokens=True)}\n")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 GPT-2 做 few-shot 情感分类，对比 0-shot、1-shot、5-shot 的效果。

*参考答案*：在 prompt 中放入 0/1/5 个"文本→标签"示例，再让模型续写最后一条的标签。

```python
def build_prompt(shots, query):
    demos = "".join(f'Text: "{t}" -> {l}\n' for t, l in shots)
    return f"Classify sentiment:\n{demos}Text: \"{query}\" ->"
# shots=[] 即 0-shot；取 1 条为 1-shot；取 5 条为 5-shot
out = model.generate(**tokenizer(build_prompt(shots, q), return_tensors="pt"),
                     max_new_tokens=3, do_sample=False)  # 贪心，便于复现
```
预期：随示例增多，GPT-2 越能模仿"输出单个标签词"的格式，准确率通常 0-shot < 1-shot < 5-shot。但 GPT-2（124M–1.5B）本身较弱，提升有限且对示例顺序/措辞敏感；In-Context Learning 的威力要到 GPT-3（175B）规模才充分显现。评测时建议固定标签集并只取生成首词归一化匹配。

**练习 2：** 用不同 temperature（0.1, 0.5, 1.0, 1.5）生成文本，分析多样性和质量的权衡。

*参考答案*：temperature `T` 缩放 logits：`softmax(logits / T)`。`T→0` 趋近贪心（确定、保守），`T` 越大分布越平、采样越随机。

```python
for T in [0.1, 0.5, 1.0, 1.5]:
    out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=T)
    print(T, tokenizer.decode(out[0], skip_special_tokens=True))
```
权衡：`T=0.1` 文本连贯但单调、易重复；`T=0.5–0.7` 多样性与连贯性平衡，最常用；`T=1.0` 更有创意但偶有跑题；`T=1.5` 高度随机，常出现语法错乱、胡言乱语。实践中常配合 `top_p`（核采样）/`top_k` 截断长尾，再调 `T`。

### 进阶题

**练习 3：** 实现 Chain-of-Thought 提示，观察 GPT-2 是否有推理能力（预期：没有，需要更大模型）。

*参考答案*：CoT 即在 prompt 里加"Let's think step by step"或给出带推理步骤的示例，引导模型先写中间推理再给答案。

```python
prompt = ("Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many now?\n"
          "A: Let's think step by step.")
out = model.generate(**tokenizer(prompt, return_tensors="pt"),
                     max_new_tokens=60, do_sample=False)
```
观察结论与 Wei et al. 2022（CoT）一致：CoT 是一种**涌现能力，仅在 ~100B+ 参数模型上才稳定生效**。GPT-2（≤1.5B）即使加了 CoT 提示，也往往写出看似步骤、实则计算错误的文本，正确率不会系统性提升——这正反证了"推理"依赖规模。要复现有效 CoT 需换更大模型（如 GPT-3.5/4、LLaMA-3-70B）。

**练习 4：** 用 GPT-2 的 logits 分析模型对下一个 token 的预测概率分布，可视化 top-10 候选词。

*参考答案*：取最后一个位置的 logits，softmax 后用 `topk` 取前 10。

```python
import torch
inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits          # Shape: [B, N, vocab_size]
probs = torch.softmax(logits[0, -1], dim=-1) # 最后一个 token 的下一词分布
topp, topi = probs.topk(10)
for p, i in zip(topp, topi):
    print(f"{tokenizer.decode(i):>12s}  {p.item():.3f}")
```
预期：" Paris" 概率最高，其后是" the"" a"等高频续接词。可进一步对比施加 temperature 前后分布的"尖锐/平坦"变化，直观理解采样温度对候选词概率的影响。柱状图用 `matplotlib.bar(范围, topp)` 绘制即可。
