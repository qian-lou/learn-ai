# 知识蒸馏 / Knowledge Distillation

## 1. 背景（Background）

> **为什么要学这个？**
>
> 知识蒸馏用**大模型（Teacher）**教**小模型（Student）**，让小模型获得接近大模型的能力。DistilBERT 比 BERT 小 40% 但保留 97% 的性能。在大模型时代，用 GPT-4 生成数据训练小模型也是一种蒸馏。
>
> 对于 Java 工程师来说，蒸馏就像**导师带新人**——新人不需要经历导师所有的学习过程，直接学习导师的经验总结。

## 2. 知识点（Key Concepts）

| 蒸馏类型 | 方法 | 代表 |
|----------|------|------|
| 输出蒸馏 | 学习 Teacher 的 soft label | DistilBERT |
| 特征蒸馏 | 学习 Teacher 的中间表示 | TinyBERT |
| 数据蒸馏 | 用 Teacher 生成训练数据 | Alpaca, Vicuna |

## 3. 内容（Content）

### 3.1 经典蒸馏原理

```
传统训练:
  Student 学习 hard label: [0, 1, 0, 0]（one-hot）
  → 只学到"正确答案"，丢失了类别间关系

蒸馏训练:
  Teacher 输出 soft label: [0.05, 0.85, 0.05, 0.05]
  → Student 学到"正确答案是 1，但 0 和 2 有微小可能"
  → 保留了类别间的相似度信息

损失函数:
  L = α × CE(soft_teacher, soft_student) + (1-α) × CE(hard_label, student)
  
  Temperature T: 控制软标签的"软度"
  softmax(logits / T)  T=1: 正常  T=5: 更软  T=20: 几乎均匀
```

### 3.2 蒸馏实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数 / Knowledge distillation loss."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失：KL 散度
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean',
        ) * (self.T ** 2)
        
        # 硬标签损失：交叉熵
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### 3.3 大模型时代的蒸馏

```
数据蒸馏（现代主流）:

不是直接蒸馏模型参数，而是用大模型生成训练数据

方式 1: Self-Instruct
  用 GPT-4 生成 instruction-response 数据
  → 训练小模型 → Alpaca (52K 条 GPT-3.5 数据)

方式 2: Evol-Instruct
  逐步增加指令复杂度
  → WizardLM

方式 3: 对话蒸馏
  收集 ChatGPT 对话 → 训练小模型
  → Vicuna (70K ShareGPT 对话)

效果:
  LLaMA-7B + Alpaca 数据 ≈ GPT-3.5 的 80% 效果
  成本: 训练 GPT-3.5 级别模型 < $100
```

> **2024-2025 现代蒸馏主线 / Modern distillation (2024-2025)：** 不再局限于 BERT→DistilBERT 的老套路，主流是「强模型（GPT-4 / Claude）→ 小模型」的**数据蒸馏**，并叠加 **on-policy distillation**——让 Student 先生成，再用 Teacher 对 Student 自己的输出打分/纠正（如 MiniLLM 的 reverse-KL、DistillKit / 各家 distillation 工具链），比纯离线静态数据更能贴合 Student 分布、缓解 exposure bias。

## 4. 详细推理（Deep Dive）

### 4.1 Temperature 的作用

```
T=1 (正常):  [0.01, 0.97, 0.01, 0.01]  → 几乎 one-hot
T=5 (温和):  [0.10, 0.70, 0.10, 0.10]  → 较软
T=20 (很软): [0.20, 0.40, 0.20, 0.20]  → 接近均匀

高温让概率分布更平滑：
  → Teacher 的"隐式知识"更明显
  → Student 能学到更多类别间关系
  → 但 T 太高信息被过度平滑
  → 推荐 T=2~10
```

## 5. 例题（Worked Examples）

```python
# 用 HuggingFace 做蒸馏
# Teacher: bert-base, Student: 小 BERT
teacher = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# 用 DistillationLoss 训练 student
```

## 6. 习题（Exercises）

### 基础题

**练习 1：** 实现 DistillationLoss，用 BERT 蒸馏到小模型做分类。

*参考答案*：

复用 3.2 的 `DistillationLoss`，训练循环关键：Teacher 推理时 `eval()` + `no_grad()`，只更新 Student。

```python
teacher.eval()
for batch in loader:
    with torch.no_grad():
        t_logits = teacher(**batch).logits          # 软标签来源 / soft targets
    s_logits = student(**batch).logits
    loss = distill_loss(s_logits, t_logits, batch["labels"])
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

要点：(1) Teacher 不回传梯度，省显存；(2) 软标签损失已乘 `T²` 校正梯度量级；(3) Student（如 `distilbert-base`）通常能保留 Teacher 约 95%+ 的准确率，体积小约 40%。

**练习 2：** 对比不同 Temperature（2, 5, 10）对蒸馏效果的影响。

*参考答案*：

固定其它超参，仅扫 `T ∈ {2, 5, 10}`，记录 Student 在验证集的准确率。

```python
for T in (2, 5, 10):
    distill_loss = DistillationLoss(temperature=float(T), alpha=0.7)
    acc = train_and_eval(student_fresh(), distill_loss)  # 每次重置 student
    print(f"T={T}: acc={acc:.4f}")
```

预期：T 越大软标签越平滑、暗知识（类间关系）越显著，但过大时信息被过度抹平。分类任务常见最优在 T≈2~5；本质是"平滑程度 vs 信息量"的权衡，没有放之四海皆准的值，需按任务实测。

### 进阶题

**练习 3：** 用 GPT-4 API 生成 1000 条指令数据，微调 GPT-2（数据蒸馏）。

*参考答案*：

两步：先用强 Teacher（GPT-4 类）批量造 `instruction → response` 数据，再 SFT 小模型。

```python
# 1) 生成数据 / generate data (Self-Instruct 风格，可给种子任务提多样性)
from openai import OpenAI
client = OpenAI()
data = []
for _ in range(1000):
    r = client.chat.completions.create(model="gpt-4o",
        messages=[{"role": "user", "content": "生成一条指令及其高质量回答，JSON 格式"}])
    data.append(r.choices[0].message.content)  # 落盘为 jsonl

# 2) SFT GPT-2：把样本拼成 "指令\n回答" 文本，按因果语言建模微调
#    Fine-tune GPT-2 with causal-LM loss on "instruction\nresponse"
```

要点：(1) 这是**数据蒸馏**，迁移的是 Teacher 的"行为/输出分布"而非参数；(2) 必须去重、过滤低质/越权样本，质量比数量更关键；(3) 合规提示：用闭源模型输出训练模型可能受其服务条款限制，生产前需确认许可。

**练习 4：** 对比数据蒸馏和模型蒸馏在小模型上的效果差异。

*参考答案*：

| 维度 | 模型蒸馏（logits/特征） | 数据蒸馏（生成数据） |
|------|------------------------|----------------------|
| 需要 Teacher | 训练时在线前向，开销大 | 仅离线造数据一次 |
| 是否需同词表/可访问权重 | 通常需要 | 不需要，黑盒 API 即可 |
| 监督信号 | 软标签信息丰富 | 仅硬文本（信息略少） |
| 适用 | 同架构压缩、分类 | 指令/对话能力迁移（LLM 主流） |

结论：经典分类压缩场景模型蒸馏样本效率更高；大模型指令对齐时数据蒸馏（Alpaca/Vicuna 路线）更实用、可跨架构。实验上控制 Student 与训练量一致再比准确率/胜率，才公平。二者也常结合：先数据蒸馏扩样本，再加 logits 蒸馏。
