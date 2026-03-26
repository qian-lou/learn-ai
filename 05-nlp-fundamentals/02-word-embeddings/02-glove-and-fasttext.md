# GloVe 与 FastText

## 1. 背景（Background）
> GloVe 利用全局共现矩阵，FastText 处理子词信息解决 OOV（未登录词）问题。

## 2-3. 知识点与内容
- **GloVe**: 基于全局词-词共现矩阵因式分解，兼顾局部和全局信息
- **FastText**: 将词分解为字符 n-gram，可为从未见过的词生成向量
- 对比：Word2Vec 纯预测式，GloVe 统计+预测，FastText 子词级别

## 4-6. 推理/例题/习题
**练习：** 对比三种词嵌入在词相似度和类比任务上的表现。
