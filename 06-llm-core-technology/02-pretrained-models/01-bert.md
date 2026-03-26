# BERT 详解 / BERT in Depth

## 1. 背景（Background）
> BERT (Bidirectional Encoder Representations from Transformers) 开创了"预训练+微调"范式，是 NLP 领域的里程碑。

## 2-3. 知识点与内容
```python
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# BERT 预训练任务：MLM(掩码语言模型) + NSP(下一句预测)
# MLM: "I [MASK] NLP" → 预测 [MASK] = "love"

model = BertModel.from_pretrained("bert-base-uncased")
# bert-base: 12层, 768维, 12头, 110M参数
# bert-large: 24层, 1024维, 16头, 340M参数

# 微调做分类
classifier = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

## 4-6. 推理/例题/习题
**练习：** 用 BERT 微调做情感分类任务。
