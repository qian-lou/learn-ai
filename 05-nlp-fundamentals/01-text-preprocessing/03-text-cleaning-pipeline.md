# 文本清洗流水线 / Text Cleaning Pipeline

## 1. 背景（Background）
> 真实文本充满噪音，大模型训练的数据质量直接决定模型质量。"Garbage in, garbage out."

## 2-3. 知识点与内容
```python
import re

def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)    # 去 HTML
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)    # 去标点
    text = re.sub(r'\s+', ' ', text)       # 合并空白
    return text

# 大模型数据清洗：去重、语言检测、质量过滤、敏感内容过滤
```

## 4-6. 推理/例题/习题
**练习：** 构建完整的文本清洗 Pipeline，处理原始 Web 语料。
