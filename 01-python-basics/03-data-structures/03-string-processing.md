# 字符串处理
# String Processing

## 1. 背景（Background）

> Python 的字符串处理能力远超 Java，内置了大量便捷方法。NLP 和大模型开发中，文本处理是最基本的操作——分词、清洗、格式化都依赖字符串操作。

## 2. 知识点（Key Concepts）

| 操作 | Java | Python |
|------|------|--------|
| 格式化 | `String.format()` | `f"Hello {name}"` |
| 分割 | `str.split(",")` | `str.split(",")` |
| 连接 | `String.join(",", list)` | `",".join(list)` |
| 多行 | `"""` (Java 15+) | `"""多行"""` |
| 正则 | `java.util.regex` | `re` 模块 |

## 3. 内容（Content）

### 3.1 常用方法

```python
s = "  Hello, World!  "

s.strip()          # trim()  →  "Hello, World!"
s.lower()          # toLowerCase()
s.upper()          # toUpperCase()
s.replace(",", ";") # replace
s.split(",")       # split  →  ["  Hello", " World!  "]
s.startswith("  H") # startsWith
s.find("World")    # indexOf  →  9
"World" in s       # contains  →  True

# f-string 格式化（Python 3.6+）
name, score = "Alice", 95.5
print(f"{name}: {score:.1f}")   # Alice: 95.5
print(f"{name:>10}")            # 右对齐，宽度 10
print(f"{1000000:,}")           # 千位分隔
```

### 3.2 正则表达式

```python
import re

text = "Call 123-456-7890 or 987-654-3210"
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)
# ['123-456-7890', '987-654-3210']

# 替换 / Replace
cleaned = re.sub(r'[^\w\s]', '', "Hello, World!")
# 'Hello World'
```

## 4. 详细推理（Deep Dive）

- Python 字符串是不可变的（同 Java），频繁拼接用 `"".join(list)` 而不是 `+=`
- 正则表达式用 `r""` 原始字符串避免转义

## 5. 例题（Worked Examples）

```python
# NLP 文本清洗管道 / NLP text cleaning pipeline
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # 去标点
    text = re.sub(r'\s+', ' ', text)      # 合并空白
    return text
```

## 6. 习题（Exercises）

**练习 1：** 实现一个函数将驼峰命名转为蛇形命名（`camelCase` → `camel_case`）。

**练习 2：** 用正则表达式从日志文本中提取所有 IP 地址。
