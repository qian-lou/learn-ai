# 列表与元组（ArrayList 对比）
# List and Tuple (vs ArrayList)

## 1. 背景（Background）

> Python 的 `list` 对应 Java 的 `ArrayList`，但更灵活——可以存储不同类型的元素。`tuple` 是不可变列表，类似 Java 14+ 的 `Record`。在大模型开发中，数据管道的核心操作就是列表处理。

## 2. 知识点（Key Concepts）

| 特性 | Java ArrayList | Python list | Python tuple |
|------|---------------|-------------|-------------|
| 可变性 | 可变 | 可变 | **不可变** |
| 类型约束 | 泛型约束 | 任意类型混合 | 任意类型混合 |
| 语法 | `new ArrayList<>()` | `[1, 2, 3]` | `(1, 2, 3)` |
| 切片 | 无 | `list[1:3]` | `tuple[1:3]` |

## 3. 内容（Content）

### 3.1 列表基础

```python
# 创建 / Creation
# Java: List<Integer> nums = new ArrayList<>(Arrays.asList(1, 2, 3));
nums = [1, 2, 3]
mixed = [1, "hello", 3.14, True]  # 可混合类型（Java 不行）

# 常用操作 / Common operations
nums.append(4)          # add(4)
nums.insert(0, 0)       # add(0, 0)
nums.extend([5, 6])     # addAll(Arrays.asList(5, 6))
nums.pop()              # remove(size()-1)
nums.remove(3)          # remove(Integer.valueOf(3))
length = len(nums)      # size()
```

### 3.2 切片（Python 独有杀手级特性）

```python
# 切片语法: list[start:stop:step]
# Slicing syntax: list[start:stop:step]
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

nums[2:5]     # [2, 3, 4]  （不含 stop）
nums[:3]      # [0, 1, 2]  （前 3 个）
nums[-3:]     # [7, 8, 9]  （后 3 个）
nums[::2]     # [0, 2, 4, 6, 8]  （每隔一个）
nums[::-1]    # [9, 8, ..., 0]  （反转！）
```

### 3.3 列表推导式

```python
# Time: O(N)  Space: O(N)
# Java Stream 等价操作
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### 3.4 元组

```python
# 不可变，常用于函数返回多值和字典键
point = (3, 4)
x, y = point  # 解构

# 命名元组（类似 Java Record）
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)
```

### 3.5 各操作的复杂度（性能心智模型 / performance mental model）

```python
# 关键：按"末尾"还是"头部/中间"操作，复杂度天差地别
# Key: cost depends on WHERE you mutate, not just WHAT you do
nums = list(range(1000))

nums.append(9)          # 均摊 O(1) / amortized O(1)（尾部追加）
nums.pop()              # O(1)（尾部弹出）
nums[500]               # O(1)（随机访问，本质是数组下标）/ random access
nums.insert(0, -1)      # O(N)！整体后移一格 / shifts all elements right
nums.pop(0)             # O(N)！整体前移一格 / shifts all elements left
9 in nums               # O(N)（线性扫描，list 无哈希索引）/ linear scan
nums.remove(500)        # O(N)（先查找再前移）

# 陷阱：频繁头部进出请用 deque（双端队列），两端均 O(1)
# Pitfall: for FIFO/head ops use deque — both ends are O(1)
# 对应 Java：ArrayList.add(0,x) 也是 O(N)，应换 ArrayDeque
from collections import deque
queue = deque([1, 2, 3])
queue.appendleft(0)     # O(1)，ArrayList 做不到 / ArrayDeque.offerFirst
queue.popleft()         # O(1)
```

### 3.6 浅拷贝陷阱（最高频 Bug / the #1 list footgun）

```python
import copy

# 陷阱 1：赋值不是拷贝，只是多一个引用 / assignment aliases, not copies
a = [1, 2, 3]
b = a                   # b 与 a 指向同一对象 / same object (like Java reference)
b.append(4)
print(a)                # [1, 2, 3, 4] —— a 也变了！

# 陷阱 2：[:] / list() / copy() 都是"浅拷贝"，只复制顶层
# Shallow copy duplicates the outer list but shares inner objects
matrix = [[0, 0], [0, 0]]
shallow = matrix[:]            # 或 matrix.copy() / list(matrix)
shallow[0][0] = 99
print(matrix)           # [[99, 0], [0, 0]] —— 内层列表被共享！

# 正解：嵌套结构必须用 deepcopy / use deepcopy for nested structures
deep = copy.deepcopy(matrix)
deep[0][0] = -1
print(matrix)           # 不受影响 / unaffected

# 陷阱 3：用乘法初始化二维列表 —— 经典翻车点
# Pitfall: [[0]*3]*2 makes 2 references to the SAME row
bad = [[0] * 3] * 2     # 两行其实是同一个对象 / both rows are one object
bad[0][0] = 1
print(bad)              # [[1, 0, 0], [1, 0, 0]] —— 不是你想要的！
good = [[0] * 3 for _ in range(2)]  # 推导式每次新建一行 / fresh row each time
good[0][0] = 1
print(good)             # [[1, 0, 0], [0, 0, 0]] 正确
```

### 3.7 元组的"不可变"边界（易错点 / common misconception）

```python
# 不可变指"绑定不可变"，不指"指向的对象不可变"
# Immutability means the bindings are fixed, NOT the referenced objects
t = (1, [2, 3])
# t[1] = [9]            # TypeError：不能替换元组的槽位 / can't rebind a slot
t[1].append(4)          # 但内层可变对象本身能改！/ inner mutable object is still mutable
print(t)                # (1, [2, 3, 4])
# 后果：含 list 的 tuple 不可哈希，不能做 dict 键 / unhashable, can't be a key

# 元组解包的进阶用法 / advanced unpacking
first, *rest = [1, 2, 3, 4]   # first=1, rest=[2, 3, 4]（星号收集）
a, b = 1, 2                    # 先赋值 / init
a, b = b, a                    # 一行交换，无需临时变量 / swap without temp（现在 a=2, b=1）
```

## 4. 详细推理（Deep Dive）

### 4.1 CPython 中 list 的真实内存布局

`list` 不是链表，而是**一个指向"指针数组"的对象**。CPython 的结构体 `PyListObject` 含三个关键字段：

- `ob_item`：指向一块连续内存（`PyObject**`），数组里存的是**指针**，不是值本身；
- `ob_size`：当前元素个数（`len()` 直接读这个字段，所以 `len()` 是 O(1)）；
- `allocated`：已申请的容量（≥ `ob_size`）。

这解释了为什么 `list` 能混存类型：数组里每个槽位都是 `PyObject*`，整数、字符串、子列表统统是堆对象的指针。代价是**缓存不友好**——元素散落在堆各处，遍历要不断解引用（这正是 NumPy 用连续原生数组能快 100x 的根因）。对比 Java：`ArrayList` 底层是 `Object[] elementData`，同样是引用数组，存基本类型要装箱（`Integer`），与 Python 的"万物皆对象"异曲同工。

### 4.2 动态数组扩容：为什么 append 是"均摊 O(1)"

当 `ob_size == allocated`、还要 `append` 时，CPython 调用 `list_resize`。它**不是**每次 +1，而是按下面这条公式**成倍式过度分配**（CPython `Objects/listobject.c`）：

```text
new_allocated = (newsize >> 3) + (newsize < 9 ? 3 : 6) + newsize
# 增长序列约为：0, 4, 8, 16, 25, 35, 46, 58, 72, 88, ...（约 1.125 倍 + 常数）
```

关键在于扩容把旧数组 `memcpy` 到新数组（一次 O(N)），但这种重排**只在容量耗尽时偶尔发生**。把 N 次 append 的总搬运成本摊到每次上：几何级数求和 `4+8+16+...+N ≈ 2N`，除以 N 次操作 = 平均 O(1)，这就是**均摊分析（amortized analysis）**。

> 工程启示：已知最终规模时，应一次性建好（如 `[None] * n` 或推导式），避免反复触发 `list_resize` 的 `memcpy`。这等价于 Java 里 `new ArrayList<>(expectedSize)` 预设 `initialCapacity`——区别是 Java 默认扩容因子是 1.5 倍，CPython 约 1.125 倍（更省内存、更频繁搬运）。

而 `insert(0, x)` / `pop(0)` 是 O(N) 的原因也清楚了：要把 `ob_item` 后面**所有指针**整体 `memmove` 一格，与元素个数成正比。

### 4.3 tuple 为何不可变、且能当键、还更快

- **不可变 ⇒ 可哈希**：`tuple.__hash__` 由其元素哈希组合而成。因为内容恒定，哈希值在生命周期内稳定，满足"作 `dict`/`set` 键期间哈希不变"的契约（`list` 正因可变才被故意不实现 `__hash__`）。
- **更省内存**：`tuple` 没有 `allocated` 字段、不预留扩容空间，长度固定，元数据更小。
- **小整数/小元组缓存**：CPython 维护**空元组单例**和小尺寸 tuple 的**空闲链表（free list）**，反复创建小元组开销极低。这与下一节字符串驻留是同一思想——用缓存换取小不可变对象的零成本复用。
- 对应 Java：`tuple` 的定位接近 `record`（不可变、值语义、天然适合做 Map 键），而 `List.of(...)` 返回的不可变列表仍**不是**为做键设计的，因为 `List` 的 `hashCode` 依赖内容、语义上仍可能被误用。

> 一句话决策：**数据是异质、固定字段的"一条记录"用 tuple/namedtuple；是同质、会增删的"一串元素"用 list。**

## 5. 例题（Worked Examples）

### 例题 1：矩阵转置 / Matrix transpose

```python
# Time: O(M*N)  Space: O(M*N)
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
# [[1, 4], [2, 5], [3, 6]]

# 工程化写法：zip(*matrix) 更快更短（C 层实现）/ idiomatic, C-level
transposed2 = [list(col) for col in zip(*matrix)]  # 同样结果 / same result
```

### 例题 2：把样本切成训练 mini-batch（大模型数据预处理常见操作）

```python
from typing import List, Iterator

# 场景：训练前要把样本 ID 列表按 batch_size 切块喂给 DataLoader
# Scenario: chunk a flat list of sample ids into fixed-size mini-batches.
# Time: O(N)  Space: O(batch_size)（生成器惰性产出，不一次性materialize）
def make_batches(samples: List[int], batch_size: int) -> Iterator[List[int]]:
    """将样本列表切分为定长 mini-batch（最后一批可能不满）。

    Generate fixed-size mini-batches; the last batch may be shorter.

    Args:
        samples: 扁平的样本 ID 列表 / flat list of sample ids.
        batch_size: 每批样本数，必须为正 / per-batch size, must be positive.

    Yields:
        每个 batch 的样本 ID 子列表 / a sublist of ids per batch.

    Raises:
        ValueError: 当 batch_size <= 0 时 / if batch_size is non-positive.
    """
    if batch_size <= 0:  # 防御式编程：拒绝非法入参 / defensive check
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # 切片越界是安全的：samples[a:b] 自动截断到末尾 / slicing never overflows
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


data = list(range(10))
for batch in make_batches(data, batch_size=4):
    print(batch)  # [0,1,2,3] -> [4,5,6,7] -> [8,9]（末批不满，符合预期）
```

要点：用**切片 + 生成器**避免把所有 batch 一次性堆进内存——这与 §4.2 的思路互补：不需要全量物化时用切片逐批产出，避免一次性堆入内存；切片对越界自动截断，省去手写边界判断。对应 Java 需手写 `subList` 加 `Math.min` 边界，远不如此简洁。

## 6. 习题（Exercises）

**练习 1：** 用切片实现字符串回文判断。

*参考答案*：
```python
# Time: O(N) Space: O(N)
def is_palindrome(s: str) -> bool:
    return s == s[::-1]  # 切片反转 / slice-reverse
```

**练习 2：** 用列表推导式生成九九乘法表。

*参考答案*：
```python
# Time: O(N^2) Space: O(N^2)，N=9
# 每行是一个 list，整体为嵌套列表 / nested list, one row per i
table = [[f"{i}x{j}={i*j}" for j in range(1, i + 1)] for i in range(1, 10)]
for row in table:
    print(" ".join(row))  # 打印下三角 / print lower-triangular table
```

**练习 3：** 解释为什么 `tuple` 可以作为字典键但 `list` 不行。

*参考答案*：

字典键必须可哈希（hashable），即拥有稳定不变的 `__hash__`。`tuple` 不可变，哈希值在生命周期内恒定，故可作键；`list` 可变，内容改变会使哈希失效，破坏哈希表定位，因此 Python 不给 `list` 实现 `__hash__`，用作键会抛 `TypeError: unhashable type: 'list'`。类比 Java：作 `HashMap` 键的对象其 `hashCode/equals` 不应随时间改变，可变对象作键同样危险。注意：含 `list` 的 `tuple` 仍不可哈希。
