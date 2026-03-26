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
        
        # 执行合并
        new_vocab = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
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

```
不同模型的特殊 Token：

BERT:     [CLS] 序列开始  [SEP] 分隔  [PAD] 填充  [MASK] 掩码
GPT-2:    
