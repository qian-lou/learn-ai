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
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefu0020-\u007ea-zA-Z0-9]', '', text)
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

**练习 2：** 对比清洗前后数据对 TF-IDF 分类模型效果的影响。

### 进阶题

**练习 3：** 实现基于 MinHash 的近似去重算法。

**练习 4：** 从 Wikipedia dump 中提取纯文本，构建一个清洗 pipeline，统计最终数据量。
