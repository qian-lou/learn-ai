# 文本清洗流水线 / Text Cleaning Pipeline

## 1. 背景（Background）

> **为什么要学这个？**
>
> "Garbage in, garbage out" — 真实世界的文本数据充满噪音：HTML 标签、特殊字符、重复内容、乱码、广告。大模型训练的数据质量**直接决定**模型质量。GPT-4 和 LLaMA 之间的差距，很大程度上来自训练数据的清洗质量。
>
> 对于 Java 工程师来说，文本清洗就像是**ETL（Extract-Transform-Load）流水线**中的 Transform 阶段——对原始数据进行标准化、去噪、过滤。
>
> **在整个体系中的位置：** 数据清洗是所有 NLP 任务的前置步骤。在大模型时代，数据工程的重要性甚至超过了模型架构。

## 2. 知识点（Key Concepts）

| 清洗步骤 | 目的 | 适用场景 |
|----------|------|----------|
| HTML 去标签 | 去除网页标记 | Web 爬虫数据 |
| 正则化 | 统一格式（大小写、空白） | 全场景 |
| 去停用词 | 去除无意义高频词 | 传统 NLP（大模型不需要）|
| 词形还原 | 统一词形（running→run）| 传统 NLP |
| 去重 | 删除重复文档 | 大模型训练数据 |
| 质量过滤 | 过滤低质量文本 | 大模型训练数据 |
| 敏感内容过滤 | 去除有害内容 | 大模型训练数据 |

## 3. 内容（Content）

### 3.1 基础文本清洗

```python
import re
import unicodedata

# ============================================================
# 基础文本清洗函数
# Basic text cleaning functions
# ============================================================

def clean_text(text: str) -> str:
    """通用文本清洗 / General text cleaning.
    
    Args:
        text: 原始文本 / Raw text.
    
    Returns:
        清洗后的文本 / Cleaned text.
    """
    # 1. 去除 HTML 标签 / Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Unicode 标准化 / Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 3. 转小写 / Lowercase
    text = text.lower()
    
    # 4. 去除 URL / Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # 5. 去除邮箱 / Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # 6. 合并多余空白 / Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_for_llm(text: str) -> str:
    """大模型训练数据清洗（保留更多信息）.
    
    大模型清洗与传统 NLP 不同：
    - 不去停用词（模型需要完整语法）
    - 不做词形还原（模型自己学习）
    - 保留标点符号（有语义作用）
    """
    # 1. Unicode 标准化
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 去除控制字符（保留换行和制表符）
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
    
    # 3. 合并连续空行（最多保留 2 个换行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. 去除行首尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


# 测试 / Test
raw = "<p>Hello World! Visit https://example.com  \n\n\n  for more info.</p>"
print(f"基础清洗: {clean_text(raw)}")
print(f"LLM 清洗: {clean_for_llm(raw)}")
```

### 3.2 大模型训练数据清洗流水线

```python
import hashlib
from typing import Optional

# ============================================================
# 大模型训练数据清洗流水线（参考 RedPajama/The Pile）
# LLM training data pipeline (inspired by RedPajama/The Pile)
# ============================================================

class TextCleaningPipeline:
    """文本清洗流水线 / Text cleaning pipeline."""
    
    def __init__(self, min_length: int = 50, min_word_count: int = 10):
        self.min_length = min_length
        self.min_word_count = min_word_count
        self.seen_hashes = set()  # 用于去重
    
    def clean(self, text: str) -> Optional[str]:
        """执行完整清洗流水线 / Run full cleaning pipeline.
        
        Returns:
            清洗后的文本，不合格返回 None / Cleaned text or None.
        """
        # Step 1: 基础清洗
        text = clean_for_llm(text)
        
        # Step 2: 长度过滤
        if len(text) < self.min_length:
            return None
        
        # Step 3: 词数过滤
        word_count = len(text.split())
        if word_count < self.min_word_count:
            return None
        
        # Step 4: 去重（基于内容哈希）
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return None
        self.seen_hashes.add(text_hash)
        
        # Step 5: 质量启发式过滤
        if not self._quality_check(text):
            return None
        
        return text
    
    def _quality_check(self, text: str) -> bool:
        """启发式质量检查 / Heuristic quality check."""
        words = text.split()
        # 检查平均词长（过短可能是乱码）
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 20:
            return False
        # 检查大写比例（过高可能是标题/广告）
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if upper_ratio > 0.5:
            return False
        return True

# 使用 / Usage
pipeline = TextCleaningPipeline()
texts = ["Good text here.", "x", "<h1>SPAM AD</h1>", "A " * 100]
for t in texts:
    result = pipeline.clean(t)
    status = "✅ 保留" if result else "❌ 过滤"
    print(f"{status}: {t[:50]}...")
```

### 3.3 中文文本清洗

```python
import re

def clean_chinese(text: str) -> str:
    """中文文本清洗 / Chinese text cleaning."""
    # 全角转半角 / Full-width to half-width
    text = unicodedata.normalize('NFKC', text)
    # 去除中文标点以外的特殊字符
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef -\u007ea-zA-Z0-9]', '', text)
    # 合并空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

## 4. 详细推理（Deep Dive）

### 4.1 大模型数据清洗的关键指标

```
数据清洗质量维度（参考 LLaMA 论文）：

1. 去重 (Deduplication): ~30% 数据是重复的
   - 精确去重：MinHash + LSH
   - 近似去重：SimHash
   
2. 语言检测：过滤非目标语言文档
   - 工具：fastText langdetect, lingua

3. 质量过滤：
   - 困惑度过滤（用小模型打分）
   - 启发式规则（词长、标点比例、重复行）
   
4. 敏感内容过滤：
   - 有害内容分类器
   - 个人信息检测（PII）

数据量级：
  Common Crawl 原始数据: ~250TB
  清洗后可用数据:       ~5TB（仅 2%）
```

## 5. 例题（Worked Examples）

### 例题：构建完整的文本预处理 Pipeline

```python
from dataclasses import dataclass

@dataclass
class CleaningStats:
    total: int = 0
    passed: int = 0
    too_short: int = 0
    duplicate: int = 0
    low_quality: int = 0

# 统计清洗过程
stats = CleaningStats()
pipeline = TextCleaningPipeline()

for text in raw_corpus:
    stats.total += 1
    result = pipeline.clean(text)
    if result:
        stats.passed += 1

print(f"总计: {stats.total}, 通过: {stats.passed} ({stats.passed/stats.total:.1%})")
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 编写一个正则表达式，提取文本中的所有 URL、邮箱和电话号码。

*参考答案*：

```python
import re
from typing import Dict, List

URL_RE = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
# 电话：可选国家码 + 3-4 段数字，允许空格/横线/括号分隔
PHONE_RE = re.compile(r'(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{4}')

def extract_entities(text: str) -> Dict[str, List[str]]:
    """提取 URL / 邮箱 / 电话 / Extract urls, emails, phones.

    Returns:
        各类匹配结果的字典 / dict of matched lists.
    """
    return {
        "urls": URL_RE.findall(text),
        "emails": EMAIL_RE.findall(text),
        "phones": PHONE_RE.findall(text),
    }

sample = "Contact us at info@test.com or visit https://example.com, tel: +1 415-555-1234"
print(extract_entities(sample))
```

要点：先抽邮箱再抽 URL/电话可减少误匹配；电话号码格式各国差异极大，正则只能覆盖常见模式，工业级场景建议用 `phonenumbers` 库做校验。`[A-Za-z]{2,}` 限制邮箱顶级域至少两位，避免把句号当成域名后缀。

**练习 2：** 对比清洗前后数据对 TF-IDF 分类模型效果的影响。

*参考答案*：

实验设计：同一份带噪文本（含 HTML、URL、大小写混乱、多余空白），分别用"原始"和"`clean_text` 清洗后"两个版本训练同一个 `TfidfVectorizer + LogisticRegression`，对比测试集 F1。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def run(train_texts, test_texts, y_train, y_test):
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))])
    pipe.fit(train_texts, y_train)
    return f1_score(y_test, pipe.predict(test_texts), average='macro')

raw_f1 = run(raw_train, raw_test, y_train, y_test)
clean_f1 = run([clean_text(t) for t in raw_train],
               [clean_text(t) for t in raw_test], y_train, y_test)
print(f"raw={raw_f1:.4f}  clean={clean_f1:.4f}")
```

预期结论与原因：
- 清洗通常带来**小幅但稳定的提升**。主要收益来自**统一词形**：`<p>Good</p>`、`good`、`Good ` 经清洗后归并为同一个 token，否则它们在 TF-IDF 里是不同特征，词表被噪声稀释、有效信号变弱。
- 去 HTML/URL 还能**缩小词表、降维**，减少过拟合到无意义 token 的风险。
- 注意：清洗不是越狠越好。对 TF-IDF 这类传统模型，去停用词、统一大小写一般有益；但**过度清洗**（如删掉所有标点、否定词）可能抹掉判别信号，反而掉点——应以验证集 F1 为准做消融。

### 进阶题

**练习 3：** 实现基于 MinHash 的近似去重算法。

*参考答案*：

MinHash 用多个哈希函数估计两个文档 shingle 集合的 Jaccard 相似度：两个集合的 MinHash 签名在某一位相等的概率，恰好等于它们的 Jaccard 相似度。

```python
import re
from typing import List, Set

def shingles(text: str, k: int = 5) -> Set[str]:
    """生成 k-shingle（k 个连续词）/ k-word shingles."""
    words = re.findall(r'\w+', text.lower())
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

def minhash_signature(shs: Set[str], num_perm: int = 128) -> List[int]:
    """计算 MinHash 签名 / Compute MinHash signature.

    用 num_perm 个不同种子的哈希，取每个哈希下的最小值。
    Time: O(|shs| * num_perm)  Space: O(num_perm)
    """
    sig = []
    for seed in range(num_perm):
        # 对每个 shingle 加盐哈希，取最小 / min over salted hashes
        sig.append(min(hash((seed, s)) & 0xFFFFFFFF for s in shs))
    return sig

def estimated_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    # 签名相等位的比例 ≈ Jaccard 相似度 / fraction of equal positions
    equal = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return equal / len(sig_a)

# 近似去重：相似度超过阈值则判为重复 / dedup by threshold
def is_duplicate(text_a, text_b, threshold=0.8):
    sa, sb = minhash_signature(shingles(text_a)), minhash_signature(shingles(text_b))
    return estimated_jaccard(sa, sb) >= threshold
```

要点：num_perm 越大估计越准但越慢（128 是常用值）；真实大规模去重还需配合 **LSH（局部敏感哈希）分桶**，把"两两比较 O(N²)"降到近似 O(N)，否则文档量大时无法承受。生产中可直接用 `datasketch` 库的 `MinHash + MinHashLSH`。

**练习 4：** 从 Wikipedia dump 中提取纯文本，构建一个清洗 pipeline，统计最终数据量。

*参考答案*：

Wikipedia dump 是 wiki 标记的 XML，不能直接用，需先抽正文再接本文的清洗流水线。

```python
# 1. 抽取纯文本：用现成工具解析 wiki markup
#    pip install wikiextractor
#    python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 \
#        --json -o extracted/
#    输出每行一个 JSON：{"id":..., "title":..., "text": "纯文本"}

import json, glob
from dataclasses import dataclass

@dataclass
class Stats:
    total: int = 0
    passed: int = 0
    chars: int = 0

pipeline = TextCleaningPipeline()   # 复用本文 3.2 节的流水线
stats = Stats()

for path in glob.glob("extracted/**/wiki_*", recursive=True):
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            stats.total += 1
            cleaned = pipeline.clean(doc["text"])   # 清洗 + 长度/去重/质量过滤
            if cleaned:
                stats.passed += 1
                stats.chars += len(cleaned)

print(f"文章数: {stats.total}, 保留: {stats.passed} "
      f"({stats.passed/stats.total:.1%}), 字符数: {stats.chars:,}")
```

要点：(1) 用 `wikiextractor` 等工具去除 `[[链接]]`、模板、表格等 wiki 标记，得到纯文本；(2) 再接本文的 `TextCleaningPipeline`（长度过滤、MD5 去重、质量启发式）；(3) 流式逐行处理，避免把整个 dump 读进内存。英文维基约数百万篇，清洗后保留率与阈值有关，通常能保留大部分长正文，过滤掉重定向页、极短消歧义页等。
