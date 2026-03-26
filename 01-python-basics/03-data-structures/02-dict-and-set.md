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

## 4. 详细推理（Deep Dive）

- Python `dict` 使用哈希表实现，查找/插入/删除均为 O(1) 平均
- Python 3.6+ `dict` 在 CPython 中保序，3.7+ 作为语言规范保证
- `Counter` 是 `dict` 的子类，专用于计数场景

## 5. 例题（Worked Examples）

```python
# 统计字符频率 / Character frequency count
# Time: O(N)  Space: O(K), K=字符种类数
from collections import Counter
freq = Counter("abracadabra")
print(freq.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]
```

## 6. 习题（Exercises）

**练习 1：** 用字典实现一个简单的缓存（LRU 思路）。

**练习 2：** 给定两个列表，用集合运算找出共同元素和独有元素。

**练习 3：** 用 `defaultdict(list)` 实现分组操作（类似 Java 的 `Collectors.groupingBy`）。
