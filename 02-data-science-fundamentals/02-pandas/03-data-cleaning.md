# 数据清洗（缺失值/重复值）
# Data Cleaning

## 1. 背景（Background）

> 真实世界的数据总是"脏"的——缺失值、重复值、异常值。大模型训练前的数据清洗质量直接决定模型性能。"Garbage in, garbage out"。

## 2-3. 知识点与内容

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "text": ["hello", None, "world", "hello", ""],
    "label": [1, 2, np.nan, 1, 3],
})

# 缺失值处理 / Missing value handling
df.isnull().sum()            # 各列缺失值计数
df.dropna()                   # 删除含缺失值的行
df.fillna({"label": 0, "text": "unknown"})  # 填充缺失值
df["label"].interpolate()     # 插值填充

# 重复值处理 / Duplicate handling
df.duplicated().sum()         # 重复行计数
df.drop_duplicates()          # 删除重复行
df.drop_duplicates(subset=["text"])  # 按指定列去重

# 文本清洗（NLP 常用）/ Text cleaning
df["text"] = df["text"].str.lower().str.strip()
df = df[df["text"].str.len() > 0]  # 移除空文本
```

## 4-6. 推理/例题/习题

**练习：** 清洗一个包含空值、重复行和异常值的数据集，记录每步操作的数据条数变化。
