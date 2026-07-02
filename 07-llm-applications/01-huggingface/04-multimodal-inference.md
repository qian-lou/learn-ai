# 多模态推理与视觉 RAG / Multimodal Inference & Visual RAG

## 1. 背景（Background）

> **为什么要学这个？**
>
> 2026 年主流大模型（GPT-4o、Claude、Qwen2.5-VL、Gemini）已**原生多模态**——能同时读图和文字。真实业务里大量信息躺在**图片、扫描件、PDF 里的图表**中：发票、合同截图、财报图、产品照片。本仓库的每日计划自己也承认「纯文本 RAG 常不够用」，但前面几章的推理与 RAG 全是纯文本。这一章补上视觉这条腿。
>
> 对 Java 工程师来说，视觉语言模型（VLM）就像**重载了入参的方法**：原来的 `chat(String text)` 现在多了一个 `chat(String text, Image img)` 重载——同一个模型，多态地接收「文本 + 图像」。
>
> **在整个体系中的位置：** 承接 01 的 `transformers` 推理与 03-rag 的检索链路，把两者从「纯文本」扩展到「图文混合」。

## 2. 知识点（Key Concepts）

| 概念 | 说明 | Java 类比 |
|------|------|----------|
| VLM（视觉语言模型） | 图像编码器 + LLM，能对图文联合推理 | 重载了图像入参的 `chat()` |
| 图像 token 化 | 图片切成 patch，投影成 LLM 能吃的 token | 把图片"序列化"成模型的输入 DTO |
| 视觉 RAG | 对图片/扫描页做多模态嵌入 + 检索 | 检索键从 `String` 换成"图文向量" |
| 结构化抽取 | 从图片直接抽出强类型字段 | 图片 → Pydantic/POJO |

## 3. 内容（Content）

### 3.1 用 transformers 加载开源 VLM 做图文推理

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ============================================================
# 开源 VLM 图文推理：Qwen2.5-VL（2026 主流开源视觉模型）
# Open-source VLM: load once, ask questions about an image
# 需 GPU + 已下载权重 / needs GPU + downloaded weights
# ============================================================

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# dtype/device_map 自动分配，7B 建议 bfloat16 + 单卡或多卡
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

# 多模态消息：content 是 image + text 的列表（对比纯文本只有 text）
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "file:///data/invoice.jpg"},  # 也支持 http/base64
        {"type": "text", "text": "这张发票的总金额和开票日期是多少？"},
    ],
}]

# 1) 套用聊天模板拿到文本 prompt；2) 抽取图像张量
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)

# 生成回答 / generate
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=256)
# 只保留新生成部分（去掉 prompt 回显）
trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
print(answer)  # 例：总金额 ¥1,280.00，开票日期 2026-06-30
```

### 3.2 走 API 的视觉输入：OpenAI 与 Claude

无 GPU 时，用闭源模型的视觉 API 最省事。图片有两种传法：**公网 URL** 或 **base64 内联**。

```python
import base64
from openai import OpenAI

# ============================================================
# OpenAI 视觉：content 数组里混入 image_url（可为 http 或 data URI）
# 需 OPENAI_API_KEY / needs API key
# ============================================================
client = OpenAI()

def to_data_uri(path: str) -> str:
    """把本地图片读成 base64 data URI。时间 O(n) 空间 O(n)，n=文件字节数。"""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/jpeg;base64,{b64}"

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "提取发票金额与日期，用 JSON 回答"},
            {"type": "image_url",
             "image_url": {"url": to_data_uri("invoice.jpg"), "detail": "high"}},
        ],
    }],
)
print(resp.choices[0].message.content)
```

```python
import base64
from anthropic import Anthropic

# ============================================================
# Claude 视觉：图片用 source(base64) 传，media_type 要写对
# 需 ANTHROPIC_API_KEY
# ============================================================
client = Anthropic()
img_b64 = base64.b64encode(open("invoice.jpg", "rb").read()).decode()

msg = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "提取发票金额与日期，用 JSON 回答"},
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        ],
    }],
)
print(msg.content[0].text)
```

### 3.3 视觉 RAG：让含图表的 PDF 可被检索

纯文本 RAG 会漏掉扫描件、图表、公式截图——因为它们**没有可抽取的文字**。视觉 RAG 的思路：把每一页**渲染成图片**，用多模态模型编码成向量入库，检索时命中"页图"，再交给 VLM 回答。

```python
from pdf2image import convert_from_path          # 需系统装 poppler
from transformers import CLIPModel, CLIPProcessor
import torch, numpy as np

# ============================================================
# 视觉 RAG 索引侧：PDF 每页 → 图 → CLIP 图像向量 → 入库
# CLIP 把图和文投影到同一空间，故可用"文本 query 检索图片"
# ============================================================
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_images(images: list) -> np.ndarray:
    """一批 PIL 图片 → L2 归一化的 CLIP 向量。时间 O(B) 空间 O(B·d)，d=512。"""
    inputs = clip_proc(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = clip.get_image_features(**inputs)      # [B, 512]
    feats = feats / feats.norm(dim=-1, keepdim=True)   # 归一化，点积即余弦
    return feats.cpu().numpy()

pages = convert_from_path("report.pdf", dpi=150)        # 每页一张 PIL 图
page_vecs = embed_images(pages)                         # [页数, 512]，存入向量库

# ============================================================
# 检索侧：文本 query → CLIP 文本向量 → 与页图向量做余弦 → 取 top-k 页
# ============================================================
def search(query: str, k: int = 3) -> list[int]:
    """返回最相关的 k 个页码。时间 O(N·d)，N=页数。"""
    t = clip_proc(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        q = clip.get_text_features(**t)
    q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy()  # [1, 512]
    sims = page_vecs @ q[0]                                # 余弦相似度
    return np.argsort(-sims)[:k].tolist()

hits = search("2025 年第四季度营收增长图", k=3)
# 再把命中的页图交给 3.1/3.2 的 VLM 回答具体问题（图文两路合流）
```

> **和纯文本 RAG 的分工**：文字型 PDF 仍走 03-rag 的文本切分（更省、更精）；只有**扫描件、图表密集页**才走视觉 RAG。生产上常两路并行、结果合并。

### 3.4 从图片结构化抽取字段（图片 → 强类型对象）

```python
from pydantic import BaseModel
from openai import OpenAI

# ============================================================
# 让 VLM 直接吐强类型 JSON：图片 → Pydantic 对象（对标"图片 → POJO"）
# ============================================================
class Invoice(BaseModel):
    seller: str
    total_amount: float
    invoice_date: str      # YYYY-MM-DD

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "抽取发票的卖方、总金额、开票日期"},
            {"type": "image_url", "image_url": {"url": to_data_uri("invoice.jpg")}},
        ],
    }],
    response_format=Invoice,          # 原生 strict schema，返回即强类型
)
inv: Invoice = completion.choices[0].message.parsed
print(inv.total_amount, inv.invoice_date)   # 拿到的是校验过的对象，非裸字符串
```

## 4. 详细推理（Deep Dive）

### 4.1 VLM 为什么能"看懂"图？

```
一张图进入 VLM 的三步（以 ViT 类视觉编码器为例）：

  1) 切 patch：把 H×W 图片切成 (H/p)×(W/p) 个小块（p 常为 14 或 16）
                 —— 类似把一段文本切成 token
  2) patch → 向量：每个 patch 过一个线性投影 + 视觉 Transformer，得图像特征
  3) 投影对齐：再过一个投影层，把图像特征映射到 LLM 的 token 嵌入空间
                 —— 于是图像"变成了一串 LLM 能吃的 token"

之后图像 token 和文本 token 拼在一起，走同一个 LLM 自注意力，
所以模型能在"图的某块"和"文字的某词"之间建立注意力关联——这就是"看懂"。
```

### 4.2 图像 token 数 = 成本，和分辨率直接挂钩

图像不是"一张算一个 token"。一张图消耗的 token 数 ≈ `(H/p) × (W/p)`（p 为 patch 边长）。

- 一张 `1024×1024`、`p=14` 的图 ≈ `73×73 ≈ 5300` 个视觉 token——**比很多整段文字还贵**。
- 所以 API 常提供 `detail: "low"/"high"`：`low` 把图降采样到很少的 token（省钱、够看大意），`high` 保留细节（读小字、表格时才用）。
- 生产建议：**先缩放/裁剪到"够读清关键信息"的最小分辨率**，别把 4000×3000 的原图直接丢进去。这和纯文本里控制 prompt 长度是同一件事——token 就是钱和延迟。

## 5. 例题（Worked Examples）

### 例题：发票批量抽取（图片 → 结构化台账）

```python
from pydantic import BaseModel
from openai import OpenAI

class Invoice(BaseModel):
    seller: str
    total_amount: float
    invoice_date: str

client = OpenAI()

def extract_invoice(path: str) -> Invoice:
    """单张发票图 → 强类型对象。时间 O(1) 次 API 调用（成本随图分辨率上升）。"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "抽取卖方、总金额、开票日期"},
            {"type": "image_url", "image_url": {"url": to_data_uri(path)}},
        ]}],
        response_format=Invoice,
    )
    return completion.choices[0].message.parsed

# 批量：把一个文件夹的发票图抽成台账 / batch to a ledger
rows = [extract_invoice(p) for p in ["a.jpg", "b.jpg"]]  # 生产上并发+限流
for r in rows:
    print(r.seller, r.total_amount, r.invoice_date)
```

**要点**：用 `response_format=Invoice` 拿到的是**校验过的对象**，金额类型错误会直接触发校验，比正则从文本里抠字段稳得多——这正是 01/12 结构化输出思路在图像上的延伸。

## 6. 习题（Exercises）

### 基础题

**练习 1：** 用 3.2 的 OpenAI 视觉 API，让模型描述一张图片里有哪些物体，并统计数量。

*参考答案*：把 `messages` 里的 text 改成「列出图中所有物体及其数量，用 JSON」，其余不变；需要精确计数时把 `detail` 设为 `"high"`。

### 进阶题

**练习 2：** 用 3.3 的视觉 RAG，对一份「图表密集的季度财报 PDF」建索引，检索"营收同比增长最高的季度"，把命中页交给 VLM 回答。

*参考答案*：`convert_from_path` 渲染每页 → `embed_images` 入库 → `search(query, k=3)` 拿页码 → 把对应 `pages[i]` 按 3.1/3.2 传给 VLM 追问。注意 `dpi` 太低会糊掉小字、太高会撑爆 token，`150` 是常用折中。

**练习 3：** VLM 也会「看错」——它可能把模糊扫描件里的 `8` 读成 `3`。设计一个校验策略降低这种幻觉。

*参考答案*：三招组合——① 关键字段（金额）要求模型**同时输出边界框/原文片段**便于人工抽查；② 对同一张图用 `temperature=0` 跑两次或换一个模型交叉核对，不一致则标记人工复核；③ 金额这类可校验字段加业务规则（如"税额 ≈ 金额 × 税率"）做二次验证。核心思想和纯文本 RAG 的"引用溯源 + 不确定就说不知道"一致。
