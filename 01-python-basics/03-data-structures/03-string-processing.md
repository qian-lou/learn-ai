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

# 性能要点：热路径上反复用同一模式，先 compile 复用 / precompile hot patterns
# 对应 Java：Pattern.compile(...) 一次，循环里复用 Matcher
PHONE_RE = re.compile(r'\d{3}-\d{3}-\d{4}')   # 编译一次 / compile once
for line in text.splitlines():
    PHONE_RE.findall(line)                     # 复用，免去重复解析 / reuse
```

### 3.3 不可变性 + join：为什么不能用 `+=` 拼接

```python
# 字符串不可变：每次 += 都生成新对象，旧的丢弃 / each += creates a NEW str
# 循环里 += 是 O(N^2) 灾难 / quadratic when looping
parts = ["tok"] * 10000

# ✗ 反模式：O(N^2)，N 次拷贝累积 / anti-pattern, copies pile up
bad = ""
for p in parts:
    bad += p            # 每次都把已有内容整体复制一遍 / full copy each time

# ✓ 正解：join 先算总长、一次性分配、一次填充 = O(N)
# str.join pre-sizes the buffer and fills once
good = "".join(parts)   # 对应 Java：用 StringBuilder 而非 String 连加
# 对应 Java：String.join(",", list) / StringBuilder.append(...)

# join 只接受字符串可迭代对象，含非 str 元素要先转换 / elements must be str
nums = [1, 2, 3]
csv = ",".join(str(n) for n in nums)   # "1,2,3"（生成器即可，惰性省内存）
```

### 3.4 字节 vs 字符：编码是文本处理的隐形地雷

```python
# str 是 Unicode 码点序列；bytes 是原始字节 —— I/O 边界必须显式编解码
# str = sequence of Unicode code points; bytes = raw octets
text = "你好"                  # 2 个字符 / 2 chars
data = text.encode("utf-8")    # b'\xe4\xbd\xa0\xe5\xa5\xbd'，6 字节 / 6 bytes
print(len(text), len(data))    # 2 6 —— 字符数 ≠ 字节数！/ chars != bytes

back = data.decode("utf-8")    # 字节还原为 str / decode back to str
# data.decode("ascii")         # ✗ UnicodeDecodeError：非 ASCII 字节

# 大模型场景：tokenizer 多基于字节级 BPE，对 bytes 操作，须显式处理编码
# LLM tokenizers (byte-level BPE) operate on bytes — handle encoding explicitly
# 对应 Java：String(内部 UTF-16) ↔ byte[] 需 getBytes(StandardCharsets.UTF_8)
```

### 3.5 高频方法补充（清洗/对齐/判定 / cleaning, padding, predicates）

```python
s = "model-v2.1.bin"

s.removeprefix("model-")    # "v2.1.bin"（3.9+，比 lstrip 安全）/ safer than strip
s.removesuffix(".bin")      # "model-v2.1"
"  x  ".strip()             # 两端去空白 / trim both ends
"0042".lstrip("0")          # "42" 去前导零 / strip leading chars

# 判定方法返回 bool，常用于数据校验 / predicate methods for validation
"3.14".replace(".", "").isdigit()   # True（去掉小数点后全是数字）
"alpha".isalpha()                    # True
"  ".isspace()                       # True

# 对齐/填充：日志、表格输出 / alignment for logs & tables
"7".zfill(3)                # "007" 补零 / zero-pad
"id".ljust(8, ".")          # "id......" 左对齐填充 / left-justify
# 注意 str.split() 无参时按"任意空白"切并丢弃空串，与 split(" ") 不同！
"a  b\tc".split()           # ['a', 'b', 'c']（折叠连续空白）/ collapses runs
"a  b".split(" ")           # ['a', '', 'b']（保留空串，易错）/ keeps empties
```

## 4. 详细推理（Deep Dive）

### 4.1 不可变性 + join 为何把 O(N²) 降回 O(N)

CPython 的 `str` 对象**只读**：底层是定长缓冲（PEP 393 紧凑布局，见 4.3），创建后内容、长度、哈希全部冻结。于是 `a += b` 无法原地追加，只能**新建一个长度为 `len(a)+len(b)` 的对象，把两段都拷过去**。在循环里，第 k 次拼接要复制已累积的约 k 个字符，总成本 `1+2+...+N ≈ N²/2`，即 **O(N²)**。

`"".join(seq)` 则分两步：**先遍历一遍算出总长度，一次性分配最终缓冲，再顺序填充**，总拷贝量恰好 N，即 **O(N)**。这与 Java 完全同构——`String` 不可变，所以循环拼接要用 `StringBuilder`（内部可变 `char[]` + 扩容），`String.join` 同样预先定长。

> 注意一个例外：CPython 对 `s += x` 这种**就地形式**做了特判优化（当 `s` 引用计数为 1 时尝试 `realloc` 原地扩展），所以小循环里未必真退化到 O(N²)。但这是实现细节、不可移植、对非就地写法（`t = s + x`）无效——**工程上一律用 `join` 才稳**。

### 4.2 字符串驻留（interning）：== 为何有时快如 `is`

CPython 会把一部分字符串放进**全局驻留池（intern pool）**，让"内容相同"的字符串**共享同一个对象**：

- **自动驻留**：编译期出现的、形似标识符的字面量（仅含字母/数字/下划线，如变量名、`"utf-8"`、`dict` 的字面量键）会被自动 intern；空串和单字符 Latin-1 字符串也被缓存。
- **手动驻留**：`sys.intern(s)` 可强制入池。

驻留的收益是**比较加速**：`str.__eq__` 会先比 `id`（指针），若两者是同一驻留对象，`a == b` **一次指针比较即返回**，无需逐字符；这对**字典键、海量重复短串**（如解析 JSON/CSV 的列名、token 字符串）特别值钱——既省内存又省比较时间。这正是 Java `String.intern()` + 字符串常量池的等价机制；Java 里 `"a" == "a"` 为 true 也是因为字面量被自动 intern。**但二者共同的坑**：不能依赖 `is`/`==`（引用相等）判断内容相等——运行期拼出来的串通常不在池中，必须用值相等比较。

```python
import sys
a = "lr_schedule"; b = "lr_schedule"     # 字面量，自动驻留 / auto-interned
print(a is b)                             # 通常 True（同一对象）
c = "lr_" + "schedule"                    # 编译期常量折叠，可能也 True
d = "".join(["lr_", "schedule"])          # 运行期生成，通常 False
print(a is d, a == d)                      # False True —— 比内容要用 ==！
e = sys.intern(d)                          # 手动入池后可共享 / force intern
```

### 4.3 PEP 393：一个 str 到底占多少内存

3.3 起，CPython 用**紧凑 Unicode（PEP 393）**：根据字符串里**最大码点**自动选 1/2/4 字节每字符的内部表示（Latin-1 / UCS-2 / UCS-4），纯 ASCII 文本每字符仅 1 字节。这让英文文本省内存、索引仍是 O(1)（定宽存储，`s[i]` 直接算偏移）。代价是只要混入一个高码点字符（如 emoji），整串就升级到更宽的表示。理解这点有助于解释 4.4 的现象：`len(str)` 数的是**码点数**，与 `encode()` 后的**字节数**无关。

### 4.4 小结

- `str` 不可变 ⇒ 可哈希、可做 `dict`/`set` 键、线程安全；拼接走 `join`/`io.StringIO`，别用循环 `+=`。
- `str`（Unicode 码点）与 `bytes`（原始字节）是两类对象，I/O 边界必须显式 `encode`/`decode`，且 `len` 计的是码点不是字节。
- 驻留与常量池是"用共享换比较/内存"的同一思想（Python `sys.intern` ↔ Java `String.intern`），但**判等永远用 `==`，不用 `is`**。

## 5. 例题（Worked Examples）

### 例题 1：NLP 文本清洗管道 / NLP text cleaning pipeline

```python
import re

# 把多步清洗串成一条管道：小写化→去标点→折叠空白
# Chain cleaning steps: lowercase → strip punctuation → collapse whitespace
# Time: O(N)  Space: O(N)（每步生成新串，str 不可变 / new str per step）
_PUNCT_RE = re.compile(r"[^\w\s]")       # 预编译，函数可被高频调用 / precompiled
_SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """规范化一段原始文本，便于后续分词。

    Normalize raw text for downstream tokenization.

    Args:
        text: 原始文本（可能含大小写/标点/多余空白）/ raw input text.

    Returns:
        清洗后的小写、单空格分隔文本 / cleaned lowercase text.
    """
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)       # 去标点 / drop punctuation
    text = _SPACE_RE.sub(" ", text)      # 多个空白合并为一个 / collapse runs
    return text

print(clean_text("  Hello,   WORLD!!  "))  # "hello world"
```

### 例题 2：海量短文本去重（用驻留思想省内存 + 用 join 省时间）

```python
import sys
from typing import Iterable, Iterator, List

# 场景：清洗训练语料时，要按行去重（去掉完全重复的样本），并把保留下来的
# token 重新拼回一行。数据量大时，重复短串极多，正是 §4.2 驻留发力的场景。
# Scenario: dedup corpus lines, then re-join tokens; intern hot short tokens.
# Time: O(总字符数)  Space: O(去重后字符数) / linear in input size
def dedup_and_normalize(lines: Iterable[str]) -> Iterator[str]:
    """逐行去重并把每行 token 用单空格重新连接（流式，不一次性载入）。

    Stream-dedup lines and re-join their tokens with single spaces.

    Args:
        lines: 任意可迭代的文本行（可为文件句柄，惰性读取）/ iterable of lines.

    Yields:
        去重后、token 单空格连接的规范行 / unique, space-joined lines.
    """
    seen: set[str] = set()               # set 判重平均 O(1)，远胜 list 的 O(N)
    for line in lines:
        # split() 无参：按任意空白切并丢弃空串（§3.5）/ split on any whitespace
        tokens: List[str] = line.split()
        if not tokens:                   # 跳过空行 / skip blank lines
            continue
        # 驻留高频短 token：内容相同的串共享对象，省内存 + 加速去重比较
        # intern hot tokens so equal strings share one object (§4.2)
        normalized = " ".join(sys.intern(t) for t in tokens)
        if normalized in seen:           # 已出现则跳过 / dedup
            continue
        seen.add(normalized)
        yield normalized                 # join 一次成串 O(N)，非 += / single join


sample = ["the  cat", "the cat", "a dog", "the   cat\n"]
print(list(dedup_and_normalize(sample)))  # ['the cat', 'a dog']
```

要点：(1) **去重用 `set`** 而非 `list in`，把每行判重从 O(N) 降到平均 O(1)（呼应 dict/set 章）；(2) **`sys.intern`** 让"the""cat"等高频 token 在内存里只存一份，海量语料下显著省内存、并让后续比较走指针快路（§4.2）；(3) **重组用 `" ".join(...)`** 一次分配、O(N) 完成，杜绝循环 `+=` 的 O(N²)（§4.1）；(4) 函数是**生成器**，可直接对接文件句柄做流式清洗，不必把整份语料读进内存。

## 6. 习题（Exercises）

**练习 1：** 实现一个函数将驼峰命名转为蛇形命名（`camelCase` → `camel_case`）。

*参考答案*：
```python
# Time: O(N) Space: O(N)
import re

def camel_to_snake(name: str) -> str:
    # 在大写字母前插入下划线再转小写 / insert "_" before uppercase, then lower
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

print(camel_to_snake("camelCase"))   # camel_case
print(camel_to_snake("HTTPServer"))  # h_t_t_p_server（连续大写为已知边界情况 / known edge case）
```

**练习 2：** 用正则表达式从日志文本中提取所有 IP 地址。

*参考答案*：
```python
# Time: O(N) Space: O(K)，K=匹配数 / number of matches
import re

log = "from 192.168.1.1 to 10.0.0.255, denied 8.8.8.8"
# \b 词边界 + 四段 1~3 位数字 / four 1-3 digit groups with word boundaries
ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", log)
print(ips)  # ['192.168.1.1', '10.0.0.255', '8.8.8.8']
# 注：此式不校验 0~255 范围，仅做提取 / does not validate octet range, extraction only
```
