# 01-numpy — NumPy 数值计算

> **所属阶段**：阶段二 · 数据科学基础
> **学习目标**：掌握 NumPy 高性能数值计算，理解向量化编程思想

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [ndarray-basics](./01-ndarray-basics.md) | ndarray 基础与创建 | 数组创建、dtype、shape/reshape 操作 |
| 02 | [indexing-and-slicing](./02-indexing-and-slicing.md) | 索引与切片 | 花式索引、布尔索引、多维切片 |
| 03 | [broadcasting](./03-broadcasting.md) | 广播机制 | 广播规则、shape 兼容性、常见模式 |
| 04 | [linear-algebra-ops](./04-linear-algebra-ops.md) | 线性代数运算 | 矩阵乘法、特征值分解、SVD |
| 05 | [performance-optimization](./05-performance-optimization.md) | 性能优化 | 向量化 vs 循环、内存布局、ufunc |

---

## 🎯 学习要点

- **永远不要对 Tensor/ndarray 写 for 循环**，必须使用向量化操作
- 理解广播机制是掌握 PyTorch 的前置基础
- 线性代数运算是机器学习数学的核心工具
