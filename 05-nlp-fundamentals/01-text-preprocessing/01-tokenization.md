# 分词 / Tokenization

## 1. 背景（Background）

> **为什么要学这个？**
>
> 分词（Tokenization）是 NLP 的**第一步**——将原始文本转换为模型可以处理的数字序列。对于 Java 工程师来说，分词就像是**编译器的词法分析（Lexical Analysis）**阶段——将源代码字符串分割成有意义的 Token。
>
> 大模型使用**子词分词**（Subword Tokenization），如 BPE（GPT）、WordPiece（BERT）、SentencePiece（T5/LLaMA）。理解分词器的工作原理，直接影响你对模型输入、上下文窗口、token 计费的理解。
>
> **在整个体系中的位置：** 分词器决定了模型的"语言"——它决定了词表大小、输入粒度和 OOV（未登录词）处理方式。

## 2. 知识点（Key Concepts）

| 分词方法 | 代表模型 | 粒度 | 词表大小 | OOV 处理 |
|----------|---------|------|----------|----------|
| 字符级 | 早期 RNN | 单字符 | ~100 | 无 OOV |
| 词级 | 传统 NLP | 完整词 | ~100K+ | 有 OOV ❌ |
| BPE | GPT/LLaMA | 子词 | 32K-100K | 拆为子词 ✅ |
| WordPiece | BERT | 子词 | 30K | 拆为子词 ✅ |
| SentencePiece | T5/LLaMA | 子词 | 32K | 拆为子词 ✅ |

## 3. 内容（Content）

### 3.1 分词方法演进

```
分词的演进：

词级分词：
  "I love machine learning" → ["I", "love", "machine", "learning"]
  问题：词表巨大 + 无法处理新词（OOV）

字符级分词：
  "love" → ["l", "o", "v", "e"]
  问题：序列太长，失去词级语义

子词分词（BPE，现代标准）：
  "unbelievably" → ["un", "believ", "ably"]
  优势：词表适中 + 无 OOV + 保留语义
  
  高频词保持完整：  "the" → ["the"]
  低频词拆为子词：  "tokenization" → ["token", "ization"]
  未知词也能处理：  "ChatGPT" → ["Chat", "G", "PT"]
```

### 3.2 BPE 算法详解

```python
import re

# ============================================================
# BPE (Byte Pair Encoding) 算法原理
# BPE Algorithm Principle
# ============================================================

# BPE 训练过程（简化版）：
# 1. 将所有词拆成字符 + 特殊结尾符
# 2. 统计相邻字符对的频率
# 3. 合并最高频的字符对为新 token
# 4. 重复步骤 2-3，直到达到目标词表大小

# 简化实现 / Simplified implementation
def learn_bpe(corpus: list[str], num_merges: int = 10):
    """学习 BPE 合并规则 / Learn BPE merge rules.
    
    Args:
        corpus: 词列表 / List of words.
        num_merges: 合并次数 / Number of merges.
    """
    # 将词拆成字符
    vocab = {}
    for word in corpus:
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] = vocab.get(chars, 0) + 1
    
    for i in range(num_merges):
        # 统计所有相邻对的频率
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        
        if not pairs:
            break
        
        # 找最高频的对
        best_pair = max(pairs, key=pairs.get)
        print(f"Merge #{i+1}: {best_pair} (freq={pairs[best_pair]})")
        
        # 执行合并：用词边界正则避免跨 token 误合并（Sennrich 原实现同款）
        new_vocab = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        # (?<!\S)/(?!\S) 确保只在完整 token 边界处替换
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        for word in vocab:
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab

# 示例
corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
learn_bpe(corpus, num_merges=5)
```

### 3.3 使用 Hugging Face Tokenizer

```python
from transformers import AutoTokenizer

# ============================================================
# 加载预训练分词器 / Load pretrained tokenizer
# ============================================================

# BERT 分词器（WordPiece）
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
# GPT-2 分词器（BPE）
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, this is a tokenization example!"

# tokenize: 文本 → token 字符串列表
tokens = bert_tok.tokenize(text)
print(f"Tokens: {tokens}")
# ['hello', ',', 'this', 'is', 'a', 'token', '##ization', 'example', '!']

# encode: 文本 → token ID 列表（含特殊 token）
ids = bert_tok.encode(text)
print(f"IDs: {ids}")

# decode: token IDs → 文本
decoded = bert_tok.decode(ids)
print(f"Decoded: {decoded}")

# ============================================================
# 批量编码（模型输入标准格式）
# Batch encoding (standard model input format)
# ============================================================
texts = ["Hello world", "This is a longer sentence"]
encoded = bert_tok(
    texts,
    padding=True,          # 短序列补齐
    truncation=True,       # 长序列截断
    max_length=32,
    return_tensors="pt",   # 返回 PyTorch 张量
)
print(f"input_ids: {encoded['input_ids'].shape}")        # [2, 32]
print(f"attention_mask: {encoded['attention_mask'].shape}")  # [2, 32]
```


### 3.4 特殊 Token

不同分词器根据预训练目标会定义一些带特定语义占位符的特殊 Token：

- **BERT（WordPiece）**:
  - `[CLS]`: 表示序列的开始，通常用于获取整句的分类表征向量。
  - `[SEP]`: 句子分隔符，将两个不同的句子隔开。
  - `[PAD]`: 填充占位符，用来将批次中短于 max_length 的文本补齐。
  - `[MASK]`: 掩码占位符，用于 Masked LM 预训练时的遮盖预测。
- **GPT / LLaMA (BPE)**:
  - `<|endoftext|>`: 生成结束占位符，在微调中也常被直接配置为 `pad_token`。

---

## 4. 详细推理（Deep Dive）

### 4.1 BPE 字节对编码工作原理
BPE 是一种无监督的子词切分算法：
1. **词表初始化**：将所有词切分为字母，并将基础的 ASCII 码或 UTF-8 字节字符作为初始词表（大小约 256）。
2. **频率统计**：在海量语料上统计相邻字符对（Bigram）的共现频次。
3. **合并规则提取**：将频率最高的一组字符对合并（如 `e` 和 `s` 合并为新 Token `es`），并将此合并路径（Merge Rules）记录进分词规则表。
4. **自底向上组合**：循环执行上两步，直至词表扩充到预设目标（如 LLaMA 的 32000 个 Token）。
5. **处理 OOV 零失败率**：由于词表保留了基础字节字符，任何未登录的新词最终都可被切解退化为纯单字节或单个字符的组合形式，彻底避免了 OOV 的产生。

---

## 5. 例题（Worked Examples）

### 例题 1：利用 BPE 合并规则手动对未知句子执行切分 / Performing manually BPE tokenization

本例题演示如何根据已有的合并规则词表，自底向上对一个输入的测试文本执行子词拆解。

```python
from typing import List, Dict

# 定义已习得的合并规则：值为学习顺序 rank，越小越先合并（BPE 推理按学习顺序依次应用） / Learned merge rules, value = learning-order rank (smaller merges first)
merge_rules: Dict[tuple, int] = {
    ('l', 'o'): 1,
    ('lo', 'w'): 2,
    ('e', 'r'): 3,
    ('n', 'e'): 4,
    ('ne', 'w'): 5,
    # 先合出子词 'est'，"newest" 才能完整合并 / Build 'est' first so "newest" fully merges
    ('e', 's'): 6,
    ('es', 't'): 7,
    ('new', 'est'): 8,
}

def bpe_tokenize_word(word: str, rules: Dict[tuple, int]) -> List[str]:
    """使用 BPE 合并规则对单个词进行切分 / Slice single word using BPE.
    
    Time: O(L^2) - L 为单词字符长度 / L is word character length.
    Space: O(L)
    """
    # 拆解为字符序列，添加结尾符 / Start with chars
    tokens = list(word) + ['</w>']
    
    while len(tokens) > 1:
        # 统计当前词中所有的相邻元组 / Get all current adjacent bigrams
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        
        # 寻找匹配规则中 rank 最小（最先学到）的元组合并 / Find pair with smallest rank
        best_pair = None
        best_rank = float('inf')
        for pair in pairs:
            if pair in rules:
                rank = rules[pair]
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    
        if best_pair is None:
            break  # 没有任何可以合并的规则，退出 / No more merges.
            
        # 执行替换合并 / Replace bigram with merged token
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        
    return tokens

# 验证切分词汇 / Verify word tokenize
print(bpe_tokenize_word("lower", merge_rules))  # ['low', 'er', '</w>']
print(bpe_tokenize_word("newest", merge_rules)) # ['newest', '</w>']
```

---

## 6. 习题（Exercises）

### 基础题
**练习 1**：在使用 Hugging Face 的 `AutoTokenizer` 时，如果我们的批量输入句子长度各不相同，我们需要传入什么配置参数来保证模型能并行处理？
*参考答案*：
必须开启 `padding=True` 参数来进行短句补齐，使批次输出为对齐的矩形张量；同时要通过 `truncation=True` 防止极个别长句子撑爆上下文显存。
```python
# encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

### 进阶题
**练习 2**：有些分词器（如 LLaMA 使用的 SentencePiece）会把空格作为普通的控制字符 ` ` 包含在内，而有些分词器（如 BERT）在分词前会通过 RegEx 直接剔除掉所有连续空格。请分析这两种空格处理机制对跨语言文本重建（特别是保留代码缩进）有什么不同的工程影响？
*参考答案*：
- **SentencePiece（LLaMA/T5 风格）**：由于将空格视为普通字符（如用 `▁`（U+2581）代表空格），分词是完全无损的可逆过程（Round-trip Lossless）。我们可以完美从 token IDs 解码还原包括连续空格、制表符在内的原始格式。这在**代码生成模型（Code LLM）**上至关重要，能保证代码缩进不丢失。
- **WordPiece/RegEx清洗（BERT 风格）**：在预处理阶段会将多余空格合并或裁剪。这意味着分词后再做 decode 还原文本时，无法完美重建原始字符级排版（如丢失缩进和格式），这对于长文本重建、格式高度敏感的下游代码编译等任务会带来工程缺陷。
