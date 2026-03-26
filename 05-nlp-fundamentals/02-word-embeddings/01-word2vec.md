# Word2Vec（CBOW/Skip-gram）

## 1. 背景（Background）
> Word2Vec 将词映射为稠密向量，语义相似的词距离更近。"king - man + woman ≈ queen"。它是 Transformer Embedding 层的前身。

## 2-3. 知识点与内容
```python
import gensim.downloader
model = gensim.downloader.load('word2vec-google-news-300')
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
# CBOW: 上下文预测中心词 / Skip-gram: 中心词预测上下文
```

## 4-6. 推理/例题/习题
**练习：** 训练 Word2Vec 模型，用 t-SNE 可视化词向量。
