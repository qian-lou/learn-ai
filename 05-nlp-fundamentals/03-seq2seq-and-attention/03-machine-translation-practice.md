# 机器翻译实战 / Machine Translation Practice

## 1. 背景（Background）

> **为什么要学这个？**
>
> 机器翻译（Machine Translation）是 **Transformer 诞生的背景**——"Attention Is All You Need" 就是在机器翻译任务上验证的。通过实战完整的翻译项目，串联 Tokenization → Embedding → Encoder-Decoder → Attention → 生成 的全部知识。
>
> 对于 Java 工程师来说，Encoder-Decoder ≈ 一次"编码请求→解码响应"的管道：Encoder 把源句压成内部表示（类似把请求序列化成中间态），Decoder 再逐 token 解码出目标句。
>
> **在整个体系中的位置：** 机器翻译是 Seq2Seq 的经典应用。从 RNN+Attention 到 Transformer，翻译见证了 NLP 最重要的架构变革。

## 2. 知识点（Key Concepts）

| 组件 | 传统方案 | 现代方案 |
|------|---------|---------|
| 分词 | 词级分词 | BPE/SentencePiece |
| 编码器 | BiLSTM | Transformer Encoder |
| 解码器 | LSTM + Attention | Transformer Decoder |
| 评估指标 | BLEU | BLEU + COMET |
| 预训练模型 | - | mBART, NLLB, opus-mt |

## 3. 内容（Content）

### 3.1 使用预训练翻译模型

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# 方式 1: pipeline（最简单）
# Method 1: Pipeline (simplest)
# ============================================================
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
result = translator("Hello, how are you today?")
print(f"翻译结果: {result[0]['translation_text']}")


# ============================================================
# 方式 2: 手动加载模型（更灵活）
# Method 2: Manual loading (more flexible)
# ============================================================
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Machine learning is transforming every industry."
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,           # Beam Search
    length_penalty=1.0,
    early_stopping=True,
)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"翻译: {translation}")
```

### 3.2 Beam Search 解码

```
解码策略对比：

Greedy（贪心）：
  每步选概率最大的 token
  快速但可能错过更好的全局翻译

Beam Search（束搜索）：
  保留 top-K 个候选序列（beam_width=K）
  每步扩展所有候选，保留总概率最高的 K 个
  
  beam_width=1 → 退化为 Greedy
  beam_width=4 → 常用设置
  beam_width=10 → 质量更高但更慢

示例 (beam_width=2):
  Step 1: "I" → ["I love" (0.6), "I like" (0.3)]
  Step 2: 
    "I love" → ["I love you" (0.36), "I love it" (0.18)]
    "I like" → ["I like it" (0.21), "I like you" (0.06)]
  保留 top-2: ["I love you" (0.36), "I like it" (0.21)]
```

### 3.3 翻译评估指标 BLEU

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# ============================================================
# BLEU Score（翻译质量评估标准指标）
# BLEU Score (standard translation quality metric)
# ============================================================

reference = [["机器", "学习", "正在", "改变", "每个", "行业"]]
candidate = ["机器", "学习", "正在", "改变", "所有", "行业"]

# 句子级 BLEU
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score:.4f}")

# BLEU 原理:
# 计算 n-gram 精确率的几何平均值 + 短句惩罚
# BLEU-1: 1-gram 精确率（词匹配）
# BLEU-2: 2-gram 精确率（词对匹配）
# BLEU-4: 4-gram 精确率（常用指标）
#
# BLEU ∈ [0, 1]（论文常按 0–100 百分制报告，×100 即可），越高越好
# 人工翻译: ~0.4-0.6（即 40–60，因为翻译有多种正确方式）
# 现代 NMT: ~0.3-0.5（即 30–50）
```

### 3.4 批量翻译

```python
# ============================================================
# 批量翻译 / Batch translation
# ============================================================
texts = [
    "The weather is nice today.",
    "I want to learn artificial intelligence.",
    "Deep learning has revolutionized NLP.",
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_length=128, num_beams=4)

translations = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
for src, tgt in zip(texts, translations):
    print(f"EN: {src}")
    print(f"ZH: {tgt}")
    print()
```

## 4. 详细推理（Deep Dive）

### 4.1 从 RNN 翻译到 Transformer 翻译

```
RNN Seq2Seq + Attention (2015-2017):
  - 顺序处理，训练慢
  - 长序列效果差
  - BLEU ~25-30

Transformer (2017+):
  - 完全并行，训练快 3-5x
  - 长序列效果好
  - BLEU ~35-45

大模型翻译 (GPT-4, 2023+):
  - 零样本翻译，无需专门训练
  - 理解上下文和习语
  - 某些语言对上接近人工水平
```

### 4.2 翻译中的常见问题

```
1. 长度控制:
   生成太短或太长 → length_penalty 参数
   
2. 重复问题:
   "the the the..." → repetition_penalty / no_repeat_ngram_size
   
3. 多译问题:
   一个句子有多种正确翻译 → BLEU 天花板较低
   
4. 低资源语言:
   训练数据少的语言翻译质量差 → 多语言预训练（mBART/NLLB）
```

## 5. 例题（Worked Examples）

### 例题：对比不同模型的翻译质量

```python
models = [
    "Helsinki-NLP/opus-mt-en-zh",
    # "facebook/nllb-200-distilled-600M",
]

text = "Artificial intelligence will transform education fundamentally."

for model_name in models:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inp = tok(text, return_tensors="pt")
    out = mdl.generate(**inp, max_length=128)
    result = tok.decode(out[0], skip_special_tokens=True)
    print(f"{model_name.split('/')[-1]}: {result}")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 `Helsinki-NLP/opus-mt-en-zh` 翻译 10 个英文句子，人工评判翻译质量。

*参考答案*：

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

sentences = [
    "The weather is nice today.",
    "Machine learning is transforming every industry.",
    "I would like a cup of coffee, please.",
    "She has been studying English for three years.",
    "The meeting was postponed due to the storm.",
    "Could you tell me how to get to the station?",
    "This book changed the way I think about life.",
    "They are building a new bridge across the river.",
    "Artificial intelligence raises important ethical questions.",
    "He apologized for being late to the interview.",
]

for en in sentences:
    zh = translator(en)[0]["translation_text"]
    print(f"EN: {en}\nZH: {zh}\n")
```

人工评判建议从三个维度打分（如 1–5 分）：
- **忠实度 (Adequacy)**：原文意思是否完整、无遗漏/无臆造。
- **流畅度 (Fluency)**：译文是否符合中文表达习惯、读起来自然。
- **术语/习语**：专有名词、固定搭配、习语是否处理得当。

预期：opus-mt 在通用、结构清晰的句子上质量不错；容易出问题的是**习语、长难句、一词多义和文化专有表达**（这类正是 BLEU 也难充分反映、需要人工判断的地方）。

**练习 2：** 对比 `num_beams=1`（Greedy）和 `num_beams=4`（Beam Search）的翻译质量差异。

*参考答案*：

同一模型、同一句子，只改 `num_beams`，对比输出与耗时。

```python
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

name = "Helsinki-NLP/opus-mt-en-zh"
tok = AutoTokenizer.from_pretrained(name)
mdl = AutoModelForSeq2SeqLM.from_pretrained(name)

text = "Although it was raining heavily, they decided to continue the journey."
inp = tok(text, return_tensors="pt")

for nb in (1, 4):                       # 1 = Greedy, 4 = Beam Search
    t0 = time.time()
    out = mdl.generate(**inp, num_beams=nb, max_length=128)
    print(f"beams={nb} ({time.time()-t0:.2f}s): "
          f"{tok.decode(out[0], skip_special_tokens=True)}")
```

质量差异（结论）：
- **Greedy (num_beams=1)** 每步只取当前概率最大的 token，速度最快，但**只看局部最优**，可能错过整体概率更高的译文，长句/有歧义时偶尔出现别扭或不通顺。
- **Beam Search (num_beams=4)** 同时保留 4 条候选路径，每步扩展再剪枝，更接近**全局最优**，译文通常**更流畅、更准确**，BLEU 也略高；代价是计算量约为 beam 宽度倍、更慢、更耗显存。
- 经验：num_beams=4~5 是质量/速度的常用折中；继续增大收益递减，且过大 beam 有时反而偏好"安全但平淡"的短译文（需配合 `length_penalty`）。差异在简单短句上可能看不出来，在长难句上更明显。

### 进阶题

**练习 3：** 用 BLEU 评估翻译质量：收集 50 个英中平行句对，对比不同模型的 BLEU 分数。

*参考答案*：

中文 BLEU 要先**分词**（用 jieba），让 reference 和 candidate 都是词列表；多模型对比时建议用 `sacrebleu`（语料级、标准化，结果可复现）。

```python
import jieba
from transformers import pipeline
import sacrebleu

# 50 条平行句对 / parallel pairs
src_list = [...]            # 英文源句
ref_list = [...]            # 人工中文参考译文

def cut(s):                 # 中文分词后用空格连接 / segment Chinese
    return " ".join(jieba.cut(s))

models = ["Helsinki-NLP/opus-mt-en-zh"]   # 可再加 NLLB 等
for name in models:
    translator = pipeline("translation", model=name)
    hyps = [cut(translator(s)[0]["translation_text"]) for s in src_list]
    refs = [[cut(r) for r in ref_list]]   # sacrebleu: refs 为 [[ref1, ref2, ...]]
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print(f"{name}: BLEU = {bleu.score:.2f}")
```

要点：(1) **中文必须分词**，否则 n-gram 统计无意义；(2) 用**语料级 BLEU**（corpus_bleu）而非句子级平均，更稳定且是论文标准做法；(3) 多参考译文能提高 BLEU 上限（翻译本就一对多）。结论参考本文 4.1 节：opus-mt 这类专用 NMT 在 en-zh 上 BLEU 大致落在 ~30+ 量级，更强的多语模型（NLLB/mBART）在覆盖面和低资源对上通常更高。注意 BLEU 只是近似指标，应结合人工评估或 COMET 一起看。

**练习 4：** 尝试用 GPT 做翻译（通过 prompt），对比专门的翻译模型和通用 LLM 的翻译效果。

> **提示：** Prompt 如 `"Translate the following English text to Chinese:\n\n{text}\n\nTranslation:"`

*参考答案*：

通用 LLM 靠 prompt 做**零样本翻译**，无需专门训练。

```python
# 伪代码：调用任意 Chat LLM（OpenAI / 本地模型均可）
def gpt_translate(text: str, client) -> str:
    prompt = (f"Translate the following English text to Chinese. "
              f"Only output the translation, no explanation.\n\n{text}\n\nTranslation:")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                 # 翻译任务用 0，结果确定、稳定
    )
    return resp.choices[0].message.content.strip()

# 对同一组句子，分别用 gpt_translate 和 opus-mt（练习1）翻译，并排对比
```

对比结论：

| 维度 | 专用 NMT（opus-mt 等） | 通用 LLM（GPT 等）|
|------|----------------------|-------------------|
| 部署/成本 | 小、可离线、免费、快 | 大、多需 API、较贵、较慢 |
| 常规句子 | 质量稳定 | 质量相当或更好 |
| 习语/上下文/语气 | 偏直译，易僵硬 | **更强**：懂习语、随上下文调整、可控风格 |
| 可控性 | 弱（固定模型）| **强**：可用 prompt 指定语气/术语/格式 |
| 一致性/稳定性 | 高（确定性强）| 可能波动，偶有"过度发挥"或加解释 |
| 低资源语言 | 取决于训练对 | 取决于预训练覆盖，常更灵活 |

总结：常规、批量、低成本场景用专用 NMT 更划算且稳定；需要**理解上下文、处理习语/文化表达、可控风格或术语**时通用 LLM 往往更好。实践技巧：给 LLM 加约束（"只输出译文、不要解释"）、temperature=0、必要时提供术语表/few-shot 示例，可显著提升一致性。
