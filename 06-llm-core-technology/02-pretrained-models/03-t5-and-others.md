# T5 及其他模型 / T5 and Other Models

## 1. 背景（Background）

> **为什么要学这个？**
>
> T5（Text-to-Text Transfer Transformer，2019，Google）提出了一个优雅的统一范式：**所有 NLP 任务都是"文本到文本"**。分类、翻译、摘要、问答——统统转换为输入一段文本、输出一段文本。这个思想深刻影响了后续的 FLAN、ChatGPT 等模型。
>
> 对于 Java 工程师来说，"文本到文本"就像把所有业务都收敛到同一个 `Function<String, String>` 接口——不再为分类/翻译/摘要各写一套 API，统一入口靠 prompt 前缀路由不同任务。
>
> 同时，开源大模型生态正在爆发：LLaMA（Meta）、Qwen（阿里）、Mistral、DeepSeek 等模型提供了 GPT-4 的替代方案。理解这些模型的异同对技术选型至关重要。
>
> **在整个体系中的位置：** T5 代表 Encoder-Decoder 路线，LLaMA/Qwen 代表开源 Decoder-only 路线。

## 2. 知识点（Key Concepts）

| 模型 | 架构 | 参数量 | 特点 |
|------|------|--------|------|
| T5 | Encoder-Decoder | 220M-11B | 统一 text-to-text 格式 |
| FLAN-T5 | Encoder-Decoder | 80M-11B | 指令微调版 T5 |
| LLaMA-2 | Decoder-only | 7B-70B | Meta 开源标杆 |
| LLaMA-3 | Decoder-only | 8B-70B | 性能大幅提升 |
| Qwen-2 | Decoder-only | 0.5B-72B | 中文优化，阿里开源 |
| Mistral | Decoder-only | 7B | Sliding Window Attention |
| DeepSeek-V2 | Decoder-only (MoE) | 236B (21B active) | MLA + MoE 效率优化 |

## 3. 内容（Content）

### 3.1 T5 的统一范式

```
T5 将所有任务统一为 text-to-text:

分类:
  输入: "sentiment: This movie is great!"
  输出: "positive"

翻译:
  输入: "translate English to German: That is good."  
  输出: "Das ist gut."

摘要:
  输入: "summarize: [长文本...]"
  输出: "[摘要文本]"

问答:
  输入: "question: What is AI? context: AI is..."
  输出: "Artificial Intelligence"

优势: 一个模型解决所有任务，统一训练和推理接口
```

### 3.2 使用 T5 和 FLAN-T5

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ============================================================
# FLAN-T5: 经过指令微调的 T5
# FLAN-T5: Instruction-tuned T5
# ============================================================
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# 翻译
inputs = tokenizer("Translate to French: How are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 摘要
text = "summarize: Transformers have revolutionized NLP. They use attention mechanisms..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 开源大模型生态

```
2024-2025 开源大模型景观:

Meta LLaMA 系列:
  LLaMA-1 (2023): 7B-65B, 开源先驱
  LLaMA-2 (2023): 7B-70B, 70B 引入 GQA（7B/13B 仍为 MHA）, 上下文 2K→4K（可经 RoPE 外推至 32K+）
  LLaMA-3/3.1 (2024): 8B-405B, 15T tokens 训练, 初版 3.0 为 8K，3.1 起支持 128K 上下文

阿里 Qwen 系列:
  Qwen-2 (2024): 0.5B-72B, 中英双语优化
  Qwen-2.5 (2024): 数学/代码能力增强

Mistral/Mixtral:
  Mistral-7B: Sliding Window Attention, 32K 上下文
  Mixtral-8x7B: MoE 架构, 仅 13B active 参数

DeepSeek:
  DeepSeek-V2: MLA + MoE, 极致效率
  DeepSeek-Coder: 专注代码生成
```

### 3.4 模型选型指南

```
场景                推荐模型              理由
──────────────────────────────────────────────────
中文 NLP            Qwen-2.5             中文优化最好
代码生成            DeepSeek-Coder       代码专精
通用对话            LLaMA-3              社区生态最好
资源受限(单卡)      Qwen-2.5-7B          效果/成本平衡
翻译/摘要           FLAN-T5              Encoder-Decoder 适合
文本分类(传统)      BERT/RoBERTa         推理快、效果好
```

## 4. 详细推理（Deep Dive）

### 4.1 为什么 Decoder-only 成为主流？

```
Encoder-Decoder (T5):
  ✅ 翻译/摘要等任务效果好
  ❌ 参数利用率低（Encoder + Decoder 各一半参数）
  ❌ 不擅长自由对话

Decoder-only (GPT/LLaMA):
  ✅ 参数利用率高（所有参数都用于生成）
  ✅ 统一的自回归范式
  ✅ 规模够大后理解+生成都好
  ✅ 更适合对话/指令跟随
  
结论: 在大模型时代，Decoder-only + 指令微调 是最优路线
```

## 5. 例题（Worked Examples）

### 例题：对比不同模型在同一任务上的表现

```python
from transformers import pipeline

# 对比不同模型的零样本文本分类
classifiers = {
    "FLAN-T5": pipeline("text2text-generation", model="google/flan-t5-base"),
}

text = "Classify sentiment: This restaurant has amazing food!"
for name, clf in classifiers.items():
    result = clf(text, max_length=10)
    print(f"{name}: {result[0]['generated_text']}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 FLAN-T5 完成翻译、摘要、分类三种任务，体验统一 text-to-text 范式。

*参考答案*：三种任务复用**同一个**模型与 `generate` 接口，只改输入 prompt 的指令前缀，体现 text-to-text 的统一性。

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
def run(prompt, max_len=50):
    return tok.decode(model.generate(**tok(prompt, return_tensors="pt"),
                                     max_length=max_len)[0], skip_special_tokens=True)
print(run("Translate to French: How are you?"))                 # 翻译
print(run("summarize: Transformers use attention to ... [长文]")) # 摘要
print(run('Is this review positive or negative? "I love it!"'))  # 分类
```
要点：FLAN-T5 经过指令微调，能直接听懂自然语言指令；三任务的差异仅在输入文本，输出都是一段文本——这正是 T5 "万物皆 text-to-text" 的核心思想。

**练习 2：** 在 HuggingFace 上对比 LLaMA-3-8B 和 Qwen-2.5-7B 的中文理解能力。

*参考答案*：两者都用 `AutoModelForCausalLM` 加载，套各自的 chat template 后喂入相同中文问题对比输出。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
def load(name): return (AutoTokenizer.from_pretrained(name),
                        AutoModelForCausalLM.from_pretrained(name, device_map="auto"))
# name 分别取 "meta-llama/Meta-Llama-3-8B-Instruct" / "Qwen/Qwen2.5-7B-Instruct"
```
预期：**Qwen-2.5-7B 因在大规模中英双语语料上训练、中文 tokenizer 更高效，中文理解/成语/古文/数学应用题通常优于同量级 LLaMA-3-8B**；LLaMA-3-8B 英文与通用推理强、社区生态最好，但中文相对偏弱。建议用 C-Eval、CMMLU 等中文 benchmark 量化对比，而非仅看单条样例。注意 LLaMA-3 需在 HF 申请访问权限。

### 进阶题

**练习 3：** 下载 Qwen-2.5-7B，用 `transformers` 进行本地推理，测量推理速度（tokens/sec）。

*参考答案*：用 `time` 包住 `generate`，以"新生成 token 数 / 耗时"计算吞吐。

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                             torch_dtype=torch.bfloat16, device_map="auto")
inputs = tok("用一句话介绍大模型。", return_tensors="pt").to(model.device)
_ = model.generate(**inputs, max_new_tokens=8)               # 预热 / warm-up
t0 = time.time(); out = model.generate(**inputs, max_new_tokens=256); dt = time.time() - t0
n_new = out.shape[1] - inputs["input_ids"].shape[1]
print(f"{n_new / dt:.1f} tokens/sec")
```
要点：务必先**预热**（首次含编译/缓存开销）；速度取决于 GPU、精度（BF16/FP16 比 FP32 快很多）、batch 与是否启用 KV Cache（`generate` 默认开启）。7B 模型单张 A100/4090 上单序列解码约几十 tokens/sec。显存不够可加载 4-bit 量化（`load_in_4bit=True`）。

**练习 4：** 研究 MoE 架构的原理：为什么 Mixtral-8x7B 有 47B 参数但只需要 13B 的计算量？

*参考答案*：MoE（Mixture of Experts）把 Transformer 每层的单个 FFN 换成 **8 个并行专家 FFN + 一个路由器（router）**。每个 token 经路由器只选 **Top-2** 专家计算并加权求和，其余 6 个专家**不参与计算**。

```
总参数 ≈ 8 个专家的 FFN + 共享部分(注意力/嵌入/router) ≈ 46.7B（"47B"）
每 token 激活 ≈ 2 个专家 + 共享部分      ≈ 12.9B（"13B active"）
```
关键区别：**参数量算"全部专家之和"，而每次前向的计算量（FLOPs）只算被选中的 2 个专家**。注意力等非专家部分对所有 token 共享，故并非简单 `2/8`。这样既用海量参数扩大模型容量（记忆/能力上限高），又把单 token 推理成本控制在约 13B 稠密模型的水平——以显存换算力。代价：8 个专家都要常驻显存，且路由需做负载均衡防止"专家坍塌"。
