# 01-transformer — Transformer 架构

> **所属阶段**：阶段六 · 大模型核心技术
> **学习目标**：彻底理解 Transformer 架构的每个组件，能从零写出可运行的 GPT
> **预估时长**：5-7 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [self-attention](./01-self-attention.md) | 自注意力机制 | 缩放点积 `softmax(QKᵀ/√d_k)V`、因果掩码、O(N²) 复杂度瓶颈与注意力矩阵可视化 |
| 02 | [multi-head-attention](./02-multi-head-attention.md) | 多头注意力 | 一次大投影后 reshape 并行多头、头数/头维选择、MHA→MQA→GQA 的 KV 压缩演进 |
| 03 | [positional-encoding](./03-positional-encoding.md) | 位置编码 | 正弦编码→可学习编码→RoPE→ALiBi，绝对 vs 相对、长度外推 |
| 04 | [transformer-architecture](./04-transformer-architecture.md) | Transformer 完整架构 | Encoder/Decoder 三变体、FFN(SwiGLU)、残差、Pre-Norm vs Post-Norm、参数量估算 |
| 05 | [transformer-from-scratch](./05-transformer-from-scratch.md) | 从零实现 Transformer | 纯 PyTorch 逐组件搭 GPT、权重共享、KV Cache、与真实 LLaMA 的差异 |

---

## 🔑 知识点详解

### 01 · 自注意力机制
- **核心概念**：每个 token 用自己的 Query 去和所有 token 的 Key 算相关性，再对 Value 加权求和——一次全局信息聚合。
- **关键公式**：`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`；除以 `√d_k` 防止点积过大使 softmax 饱和、梯度消失。
- **易错点**：① 因果掩码在 softmax **之前**把上三角填 `-inf`，不能在 softmax 之后置 0（否则每行和不为 1）；② 复杂度是 O(N²·D)，序列翻倍显存涨 4 倍，长文本会爆显存。
- **Java 视角**：像一个全连接的消息传递系统——每个节点向所有节点发查询、按相关性加权聚合，等价于对邻接矩阵做一次带权广播。
- **前置**：矩阵乘法、softmax（阶段四）。

### 02 · 多头注意力
- **核心概念**：把 `d_model` 等分成 `h` 个头，每个头在低维子空间独立做注意力，让不同头分工关注语法/语义/指代等不同关系。
- **关键 API/公式**：`MultiHead = Concat(head₁,…,head_h)·W_O`；工程上用一次 `Linear(d, 3d)` 出 QKV 再 `view(B,N,h,d_head).transpose(1,2)`，**不是**跑 h 次小矩阵乘。
- **易错点**：① `d_model` 必须能被 `n_heads` 整除，否则 reshape 失败；② MHA 四个投影矩阵总参数是 `4·d_model²`，与头数无关——加头数不增参数，只是重新切分维度。
- **Java 视角**：一次 fork-join——把维度切成 h 份交给 h 个 worker 各算各的，最后 concat 合并，类似 `parallelStream().map(...)` 再汇总。
- **前置**：01 自注意力。

### 03 · 位置编码
- **核心概念**：自注意力对输入顺序是「置换等变」的（打乱 token 顺序结果只跟着换位），必须显式注入位置信号才能区分「I love you / you love I」。
- **关键公式**：正弦编码 `PE(pos,2i)=sin(pos/10000^(2i/d))`、`PE(pos,2i+1)=cos(...)`；RoPE 把位置编码为旋转角度，query/key 内积只保留相对距离 `(m−n)θ`。
- **易错点**：① 可学习位置编码 `nn.Embedding(max_len,d)` **无法外推**——输入超过 `max_len` 直接越界报错；② 位置编码是与词嵌入**相加**而非拼接。
- **Java 视角**：位置编码像数组下标——没有索引，序列就退化成无序的 Set，丢掉全部语序信息。
- **前置**：01 自注意力（理解为何需要位置信息）。

### 04 · Transformer 完整架构
- **核心概念**：Transformer = N 层「Attention 子层 + FFN 子层」，每个子层都包在「残差 + LayerNorm」里；三种用法——Encoder-only(BERT)、Decoder-only(GPT)、Encoder-Decoder(T5)。
- **关键公式**：FFN `= W₂·GELU(W₁x)`（中间维通常 `4·d_model`）；残差 `x_out = x + F(x)`，其导数 `1 + ∂F/∂x` 保证梯度有「1」保底、可堆到 96+ 层。
- **易错点**：① 现代大模型几乎全用 **Pre-Norm**（`x + Attn(LN(x))`），比原始 Post-Norm 稳定、常可免 warmup；② Attention 是线性加权，FFN 才提供非线性——两者缺一不可。
- **Java 视角**：像 Spring 框架——由标准化组件（Attention/FFN/Norm/残差）按固定契约组装；残差连接则像贯穿全链路的「梯度高速公路」。
- **前置**：01-03。

### 05 · 从零实现 Transformer
- **核心概念**：把前四节组件拼成可训练、可生成的 GPT——嵌入 → N×Block(Pre-Norm) → 末层 LayerNorm → LM Head，用交叉熵训 next-token。
- **关键 API**：`token_emb.weight = head.weight`（权重共享/weight tying，省 `vocab×d` 参数）；KV Cache 把每步解码从 O(N²) 降到 O(N)、总生成从 O(N³) 降到 O(N²)。
- **易错点**：① 因果掩码 buffer 用 `register_buffer` 注册，随模型迁移设备但不参与训练；② 生成时对 logits 做 `logits/temperature` 再 softmax 采样，温度越低越确定。
- **Java 视角**：堆叠 N 个 Block 就像 Servlet 的 Filter Chain——每层对隐状态变换后传给下一层，`nn.ModuleList` 就是把 Filter 注册进链条。
- **前置**：01-04，PyTorch 训练循环（阶段四）。

---

## 🎯 学习要点

- **必读原论文**：《Attention Is All You Need》（2017），先读懂 3.2 节的缩放点积注意力公式。
- **手推一遍形状**：拿纸笔跟着 `[B,N,D]→Q/K/V→[B,H,N,d_head]→scores[B,H,N,N]→out[B,N,D]` 走一遍，形状是理解多头的关键。
- **亲手实现是硬门槛**：不用 `nn.TransformerEncoder`，从零写出 05 的 GPT 并在 tinyshakespeare 上训到 loss≈1.5、能生成莎剧式文本——这是 AI 面试高频考点。
- **抓住 O(N²) 主线**：自注意力的平方复杂度是长文本瓶颈，串起 FlashAttention、稀疏注意力、GQA、KV Cache 等一切优化的动机。
- **对齐现代模型**：把玩具实现的 LayerNorm→RMSNorm、GELU-FFN→SwiGLU、可学习位置编码→RoPE、MHA→GQA，就无限逼近真实 LLaMA 解码层。
- **区分绝对/相对位置编码**：能说清为什么 RoPE/ALiBi 能外推到训练时没见过的更长上下文，而 BERT 的可学习编码不能。

---

## 🔗 关联

- **上一模块**：[阶段五 · NLP 基础](../../05-nlp-fundamentals/) — 词向量、RNN/LSTM、seq2seq 与注意力雏形。
- **下一模块**：[02-pretrained-models](../02-pretrained-models/) — 用本模块的架构搭出 BERT/GPT/T5。
- **本阶段总览**：[阶段六 README](../README.md)
- **实战延伸**：[agent-course · Day 02 模型参数](../../agent-course/Day-02-model-params.md)（temperature/top_p 与本模块 05 的采样一脉相承）、[Day 40 性能](../../agent-course/Day-40-performance.md)（KV Cache、上下文长度对延迟的影响）。
