# 字典与集合（HashMap 对比）
# Dict and Set (vs HashMap)

## 1. 背景（Background）

> Python 的 `dict` 对应 Java 的 `HashMap`，`set` 对应 `HashSet`。Python 3.7+ 的 `dict` 保持插入顺序（类似 `LinkedHashMap`）。字典是 Python 中最重要的数据结构之一，大模型的配置、超参数、数据集 metadata 都用字典表示。

## 2. 知识点（Key Concepts）

| 特性 | Java HashMap | Python dict |
|------|-------------|-------------|
| 创建 | `new HashMap<>()` | `{"key": "value"}` |
| 有序性 | 无序（LinkedHashMap 有序） | **3.7+ 保持插入顺序** |
| 空值安全 | 需要 `getOrDefault` | `dict.get(key, default)` |
| 合并 | `putAll()` | `{**d1, **d2}` 或 `d1 \| d2` |

## 3. 内容（Content）

### 3.1 字典基础

```python
# 创建 / Creation
scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

# CRUD 操作 / CRUD operations
scores["David"] = 88           # put("David", 88)
score = scores.get("Eve", 0)   # getOrDefault("Eve", 0)
del scores["Bob"]              # remove("Bob")
has_alice = "Alice" in scores  # containsKey("Alice")

# 遍历 / Iteration
for key in scores:                    # keySet()
    print(key)
for key, value in scores.items():     # entrySet()
    print(f"{key}: {value}")
```

### 3.2 常用技巧

```python
# 字典推导式 / Dict comprehension
squares = {x: x**2 for x in range(10)}

# 合并字典（Python 3.9+）/ Merge dicts
config = {"host": "localhost"} | {"port": 8080, "debug": True}

# defaultdict（类似 Java computeIfAbsent）
from collections import defaultdict
word_count = defaultdict(int)  # 默认值为 0
for word in "hello world hello".split():
    word_count[word] += 1  # 不需要判断 key 是否存在
```

### 3.3 集合

```python
# 创建 / Creation
s1 = {1, 2, 3, 4}
s2 = {3, 4, 5, 6}

# 集合运算 / Set operations
s1 & s2   # 交集 {3, 4}  （Java: retainAll）
s1 | s2   # 并集 {1,2,3,4,5,6}
s1 - s2   # 差集 {1, 2}
s1 ^ s2   # 对称差 {1, 2, 5, 6}
```

### 3.4 视图对象：keys/values/items 是"动态窗口"而非快照

```python
# 易错点：.keys()/.values()/.items() 返回的是"视图"，会随 dict 实时变化
# Pitfall: views are live windows over the dict, NOT snapshots/copies
d = {"a": 1, "b": 2}
ks = d.keys()           # dict_keys 视图 / a live view, O(1) to create
print(list(ks))         # ['a', 'b']
d["c"] = 3
print(list(ks))         # ['a', 'b', 'c'] —— 视图自动反映新增！

# 后果：边遍历边改 dict 会抛 RuntimeError（与 Java 的 ConcurrentModificationException 同理）
# for k in d:  d.pop(k)            # ✗ RuntimeError: dictionary changed size
for k in list(d):  d.pop(k)        # ✓ 先物化成 list 再遍历 / snapshot first

# keys 视图本身支持集合运算（values 不支持，因 value 可能不可哈希）
# dict_keys behaves like a set; dict_values does not
common_keys = {"a": 1, "x": 9}.keys() & {"a": 7, "y": 8}.keys()  # {'a'}
```

### 3.5 取值与计数的正确姿势（避免 KeyError / 减少二次查找）

```python
from collections import Counter

cfg = {"lr": 3e-4}

# get 只读默认；setdefault 读 + 不存在则写入并返回 / get vs setdefault
batch = cfg.get("batch", 32)            # 不修改 cfg / read-only default
cfg.setdefault("batch", 32)             # 若缺失则写入 32 / insert-if-absent
# 对应 Java：get→getOrDefault，setdefault→computeIfAbsent

# 性能陷阱：先 in 再 [] 是两次哈希查找；用 get 一次搞定
# Anti-pattern: `if k in d: x = d[k]` hashes twice; prefer a single get
if "lr" in cfg:                         # 第一次哈希 / hash #1
    lr = cfg["lr"]                      # 第二次哈希 / hash #2
lr = cfg.get("lr")                      # 一次哈希更优 / single lookup

# Counter：计数场景的瑞士军刀（dict 子类）/ Counter is a dict subclass
freq = Counter(["a", "b", "a", "c", "a"])
print(freq["z"])                        # 0，缺失键不抛错返 0 / missing -> 0
print(freq.most_common(2))              # [('a', 3), ('b', 1)] 按频次降序
```

### 3.6 集合的可哈希约束与去重不保序

```python
# 元素必须可哈希；list 不行，frozenset 可以 / elements must be hashable
# bad = {[1, 2]}          # TypeError: unhashable type: 'list'
ok = {frozenset([1, 2]), (3, 4)}        # frozenset/tuple 可作元素

# set 去重不保留原始顺序（哈希决定布局）；要保序去重用 dict.fromkeys
# set() does NOT preserve order; for order-preserving dedup use dict.fromkeys
raw = [3, 1, 2, 1, 3]
deduped = list(dict.fromkeys(raw))      # [3, 1, 2] 保持首次出现顺序 / stable
# 对应 Java：HashSet 无序去重；LinkedHashSet 才保序，与上式效果一致
```

## 4. 详细推理（Deep Dive）

### 4.1 开放寻址 vs Java 链地址：两种哈希冲突策略

哈希表的核心难题是**冲突**（不同键算出同一桶位）。两大主流解法：

| 维度 | CPython `dict`/`set`（开放寻址 Open Addressing） | Java `HashMap`（链地址 Separate Chaining） |
|------|------------------------------|--------------------------|
| 冲突处理 | 在同一连续数组内**探测下一个空槽** | 桶内挂**链表/红黑树**，冲突元素串起来 |
| 内存布局 | 单块连续数组，缓存友好 | 数组 + 大量小链表节点，指针跳转多 |
| 退化阈值 | 装载因子超过 **2/3** 即扩容 | 链长 ≥ 8 且容量 ≥ 64 时转**红黑树** |
| 最坏复杂度 | 退化到 O(N)（探测序列变长） | 退化到 O(log N)（树化兜底） |

CPython 的探测**不是简单线性 +1**，而是用伪随机扰动序列（`perturb` 由完整哈希值驱动）：`j = (5*j + 1 + perturb) % size`，每步右移 `perturb`。这样能打散聚集（clustering），让探测更均匀。因为是连续数组、无链表节点，**CPython 的哈希表通常比 Java 更省内存、更利于 CPU 缓存**；代价是删除要用"墓碑（dummy）"标记而非真删，且高装载下性能更敏感——所以它在 2/3 就扩容（Java 默认 0.75）。

### 4.2 Compact Dict：3.6+ 保序的工程奥秘

3.7+ "保持插入顺序"是**语言规范**，但实现上靠的是 3.6 引入的 **compact dict** 设计——它把一个哈希表拆成两层：

```text
indices:  [ -, 1, -, 0, -, 2, ... ]   # 稀疏数组，存"在 entries 中的下标"
entries:  [ (hash, key, val),         # 紧凑数组，严格按插入顺序追加
            (hash, key, val), ... ]
```

- 真正的键值对**按插入顺序**紧凑追加到 `entries`；
- 稀疏的 `indices` 数组只存小整数下标（指向 `entries`），承担哈希定位。

好处有二：(1) 遍历 `entries` 天然就是插入序，**保序零额外成本**；(2) 稀疏数组只存 int 下标而非完整 entry，**比旧实现省约 20–25% 内存**。这正是 Python 能让普通 `dict` 同时具备"O(1) 查找 + 有序"的原因——而 Java 必须额外用 `LinkedHashMap`（在 `HashMap` 基础上再挂一条双向链表）才能保序，内存开销更大。

### 4.3 set 与 dict 同源；hashable 契约

`set` 复用了几乎相同的开放寻址表，只是 entry 不存 value。所以**元素/键必须可哈希**：需要稳定的 `__hash__` 和自洽的 `__eq__`（`a == b` ⇒ `hash(a) == hash(b)`）。可变对象（`list`/`dict`/`set`）故意不实现 `__hash__`，因为内容一变哈希就失效、会让元素在表中"丢失"。这与 Java 完全一致：作 `HashMap` 键的对象，其 `hashCode`/`equals` 不应随状态改变，否则 `get` 找不回原值。

### 4.4 小结：复杂度与选型

- `dict`/`set` 的 get/put/`in` **平均 O(1)**，最坏 O(N)（恶意/退化哈希）；`list` 的 `in` 是 O(N)——**判重、去重、查表一律优先用 set/dict**。
- `Counter`（计数）、`defaultdict`（自动建值）、`OrderedDict`（显式有序 + `move_to_end`）都是 `dict` 子类，针对具体场景在原表上做了薄封装，零学习成本、零性能损失。

## 5. 例题（Worked Examples）

### 例题 1：统计字符频率 / Character frequency count

```python
# Time: O(N)  Space: O(K), K=字符种类数
from collections import Counter
freq = Counter("abracadabra")
print(freq.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]
```

### 例题 2：从语料构建词表（NLP/大模型 tokenizer 的第一步）

```python
from collections import Counter
from typing import Dict, List

# 场景：训练分词器/Embedding 前，要把高频词映射成连续整数 ID，并保留
# 特殊符号占位。低频词用 <unk> 兜底以控制词表规模。
# Scenario: build a token→id vocabulary, keeping specials, capping by frequency.
# Time: O(T + V log V)  Space: O(V)（T=总词数, V=去重词数 / total vs unique）
def build_vocab(
    corpus: List[str], max_size: int = 50_000, min_freq: int = 2
) -> Dict[str, int]:
    """从分好词的语料构建 token→id 映射表。

    Build a token-to-id vocabulary from a tokenized corpus.

    Args:
        corpus: 已分词的 token 列表 / a flat list of tokens.
        max_size: 词表上限（含特殊符号）/ max vocab size incl. specials.
        min_freq: 入表最低词频，过滤长尾 / drop tokens below this frequency.

    Returns:
        token→连续整数 id 的字典，保证特殊符号占据 0/1 / token→contiguous id.
    """
    # 特殊符号先占低位 id，保证跨次运行稳定 / reserve low ids, deterministic
    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    counter = Counter(corpus)              # 一次 O(T) 扫描计数 / single pass

    # most_common 已按频次降序；同频时 Counter 保持首次出现序（3.7+ 保序）
    # most_common is freq-desc; ties keep insertion order thanks to ordered dict
    for token, freq in counter.most_common():
        if freq < min_freq:                # 长尾词交给 <unk> / rare → <unk>
            break                          # 已降序，后面只会更低，可提前结束
        if len(vocab) >= max_size:         # 触顶即止 / cap reached
            break
        vocab.setdefault(token, len(vocab))  # 新词分配下一个连续 id / next id
    return vocab


tokens = "the cat sat on the mat the cat ran".split()
vocab = build_vocab(tokens, max_size=10, min_freq=2)
print(vocab)  # {'<pad>': 0, '<unk>': 1, 'the': 2, 'cat': 3}
```

要点：(1) 用 `Counter` 一遍 O(T) 完成计数，避免手写 `if k in d` 的双重哈希（§3.5/§4.1）；(2) `most_common()` 利用了 §4.2 的**保序**特性——同频词的相对顺序可复现，保证训练/推理词表一致；(3) `setdefault(token, len(vocab))` 用"当前表大小"天然生成连续 ID。对应 Java 需 `LinkedHashMap` + `entrySet().stream().sorted(...)` 才能复刻"保序 + 按频排序"，代码量数倍于此。

## 6. 习题（Exercises）

**练习 1：** 用字典实现一个简单的缓存（LRU 思路）。

*参考答案*：
```python
# Time: get/put 均 O(1)，借助 OrderedDict 保序 / O(1) via ordered dict
from collections import OrderedDict
from typing import Any

class LRUCache:
    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self._data:
            return None
        self._data.move_to_end(key)        # 命中后置为最新 / mark as recently used
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._cap:
            self._data.popitem(last=False)  # 淘汰最久未用 / evict least-recently-used
```

**练习 2：** 给定两个列表，用集合运算找出共同元素和独有元素。

*参考答案*：
```python
# Time: O(M+N) Space: O(M+N)
a, b = [1, 2, 3, 4], [3, 4, 5, 6]
sa, sb = set(a), set(b)
common = sa & sb            # 交集：共同元素 / intersection {3, 4}
only_a = sa - sb            # 差集：仅 a 独有 / {1, 2}
only_b = sb - sa            # 差集：仅 b 独有 / {5, 6}
print(common, only_a, only_b)
```

**练习 3：** 用 `defaultdict(list)` 实现分组操作（类似 Java 的 `Collectors.groupingBy`）。

*参考答案*：
```python
# Time: O(N) Space: O(N)，按奇偶分组示例 / group by parity
from collections import defaultdict

nums = [1, 2, 3, 4, 5, 6]
groups: dict[str, list[int]] = defaultdict(list)  # 缺失键自动建空 list / auto empty list
for n in nums:
    key = "even" if n % 2 == 0 else "odd"
    groups[key].append(n)   # 无需先判断 key 是否存在 / no containsKey check
print(dict(groups))  # {'odd': [1, 3, 5], 'even': [2, 4, 6]}
```
