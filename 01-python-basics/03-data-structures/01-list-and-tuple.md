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

## 4. 详细推理（Deep Dive）

- `list` 底层是动态数组（类似 `ArrayList`），`append` 均摊 O(1)，`insert(0)` 为 O(N)
- `tuple` 不可变，可作为 `dict` 的键（`list` 不行，因为 unhashable）
- 大数据场景使用 NumPy `ndarray` 替代 `list`，性能提升 100x+

## 5. 例题（Worked Examples）

```python
# 矩阵转置 / Matrix transpose
# Time: O(M*N)  Space: O(M*N)
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
# [[1, 4], [2, 5], [3, 6]]
```

## 6. 习题（Exercises）

**练习 1：** 用切片实现字符串回文判断。

**练习 2：** 用列表推导式生成九九乘法表。

**练习 3：** 解释为什么 `tuple` 可以作为字典键但 `list` 不行。
