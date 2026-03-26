# 知识蒸馏 / Knowledge Distillation

## 1. 背景（Background）
> 知识蒸馏用大模型（Teacher）的输出来训练小模型（Student），实现模型压缩。

## 2-3. 知识点与内容
```
蒸馏流程：
1. Teacher 模型用 temperature=T 的 softmax 产生软标签
2. Student 模型学习软标签的分布（而非 hard label）
3. 损失 = α × CE(soft_label, student_output) + (1-α) × CE(hard_label, student_output)

大模型蒸馏实践：
- GPT-4 → GPT-3.5 级别的小模型
- LLaMA-70B → LLaMA-7B
- 数据蒸馏：用大模型生成训练数据（如 Alpaca 数据集）
```

## 4-6. 推理/例题/习题
**练习：** 用 BERT-base 蒸馏到 DistilBERT 大小的模型。
