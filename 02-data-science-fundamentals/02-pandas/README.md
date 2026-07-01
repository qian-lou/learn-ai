# 02-pandas — Pandas 数据处理

> **所属阶段**：阶段二 · 数据科学基础
> **学习目标**：掌握结构化数据处理，对标 Java 中的 JDBC + Stream 操作，能像写 SQL 一样清洗与分析数据
> **预估时长**：5-6 天

---

## 📋 知识点大纲

| 序号 | 文件 | 主题 | 简介 |
|------|------|------|------|
| 01 | [series-and-dataframe](./01-series-and-dataframe.md) | Series 与 DataFrame | 两大核心结构、`loc`（标签）vs `iloc`（位置）、布尔过滤、`apply` 派生列、`describe/info` |
| 02 | [data-io](./02-data-io.md) | 数据读写 | `read_csv` 的 `usecols/dtype/parse_dates/chunksize`；Parquet 列存/谓词下推；对接 `datasets` |
| 03 | [data-cleaning](./03-data-cleaning.md) | 数据清洗 | `fillna/dropna` 缺失值、`drop_duplicates` 去重、`clip`/IQR 异常值、`astype`、`.str` 字符串 |
| 04 | [groupby-and-aggregation](./04-groupby-and-aggregation.md) | 分组与聚合 | Split-Apply-Combine、`agg` 命名聚合、`transform` 组内变换、`pivot_table/unstack`、对比 SQL |
| 05 | [merge-and-join](./05-merge-and-join.md) | 合并与连接 | `merge` 四种 `how`（inner/left/right/outer）、`concat` 纵横拼接、多表链式合并、对比 SQL JOIN |

---

## 🔑 知识点详解

### 01 · Series 与 DataFrame
- **核心概念**：`Series` 是带标签的一维数组（一列），`DataFrame` 是共享行索引的二维表（多列），类似数据库表 / Excel。
- **关键 API**：列选 `df["col"]`（Series）、`df[["a","b"]]`（DataFrame）；行选 `df.loc[标签]` / `df.iloc[位置]`；条件过滤 `df[df["age"]>=30]`；派生列 `df["g"]=df["s"].apply(...)`。
- **易错点**：① `loc` 按**标签**、`iloc` 按**位置**，索引被重排后二者结果不同；② 多条件过滤每个条件必须用括号包住并用 `&`/`|`（不是 `and`/`or`），如 `df[(df.a>1) & (df.b<2)]`；③ 链式赋值 `df[df.a>1]["b"]=0` 可能触发 `SettingWithCopyWarning` 且不生效，应写 `df.loc[df.a>1, "b"]=0`。
- **Java 视角**：DataFrame ≈ `JDBC ResultSet + Stream API`；`df[...]` 过滤 ≈ `stream().filter(...)`，`apply` ≈ `map(...)`，但 Pandas 是列式向量化执行，别退化成逐行循环。
- **前置**：NumPy 01-02（DataFrame 每列底层是 ndarray）。

### 02 · 数据读写（Data I/O）
- **核心概念**：I/O 是流水线第一步；格式选择直接决定内存、速度与正确性。CSV 通用但慢，Parquet 是大数据/LLM 语料首选。
- **关键 API**：`pd.read_csv(path, usecols=…, dtype=…, parse_dates=…, chunksize=…)`；`pd.read_parquet(path, columns=…, filters=…)`；`df.to_parquet(..., compression="zstd")`；`pd.read_sql(sql, engine)`。
- **易错点**：① 含缺失值的整型列不能用 `int`（NaN 是浮点），要用可空整型 `"Int64"`（大写 I）；② 不指定 `dtype` 时 pandas 逐列推断，既慢又可能误判（邮编 `00123` 被读成 int 丢前导零）；③ `chunksize` 返回的是**迭代器**不是 DataFrame，要 `for chunk in reader:` 逐块处理。
- **Java 视角**：`read_sql/to_sql` ≈ `JdbcTemplate`（底层走 SQLAlchemy ≈ JDBC）；`read_json/to_json` ≈ Jackson `ObjectMapper`；Parquet/Arrow 是跨语言列式标准，Java 侧 `parquet-mr` 读写同一份文件。
- **前置**：01（读进来就是 DataFrame）。

### 03 · 数据清洗
- **核心概念**：数据科学 80% 时间在清洗——缺失/重复/异常/格式不一，不处理就是 "Garbage In, Garbage Out"。
- **关键 API**：`df.isnull().sum()` 盘点缺失；`fillna(值/中位数)` / `dropna()`；`drop_duplicates(subset=…, keep="first")`；`clip(lower, upper)` 盖帽；`.str.strip()/.lower()/.replace(regex=True)`。
- **易错点**：① 中位数 `median()` 比均值 `mean()` 更抗异常值，填充数值列时通常优先中位数；② `fillna/drop_duplicates` 等默认**返回新对象**，不加 `inplace=True` 或不重新赋值则原 df 不变（新版更推荐赋值而非 `inplace`）；③ `astype(int)` 遇到 NaN 会报错，需先填充或转 `"Int64"`。
- **Java 视角**：分组内填充 `groupby("cat")["price"].transform(lambda x: x.fillna(x.mean()))` ≈ SQL 窗口函数 `AVG(price) OVER (PARTITION BY cat)`——一步完成"按组均值回填"。
- **前置**：01、02。

### 04 · 分组与聚合
- **核心概念**：GroupBy = **Split-Apply-Combine**（按键分组 → 每组算 → 合并），等价 SQL `GROUP BY`，是模型评估按类别/数据集统计指标的主力。
- **关键 API**：命名聚合 `df.groupby("k").agg(avg=("col","mean"), cnt=("col","count"))`；组内变换 `groupby("k")["v"].transform(...)`（**保持原行数**，用于回填/标准化）；多列分组 `groupby(["a","b"])[...].mean().unstack()` 变交叉表。
- **易错点**：① `agg` 会把每组压成一行（聚合），`transform` 保持原形状（广播回每行）——按需选择；② `groupby` 默认丢弃 key 为 NaN 的组，需要保留传 `dropna=False`；③ 新版旧式 `agg({"col":"mean"})` 语法易列名混乱，优先用命名聚合。
- **Java 视角**：`groupby(...).agg(...).query("avg>0.9").sort_values(...)` 就是 `GROUP BY … HAVING … ORDER BY` 的链式版；`transform` ≈ 窗口函数 `OVER (PARTITION BY …)`。
- **前置**：01（DataFrame 与列运算）。

### 05 · 合并与连接
- **核心概念**：`merge` 按键关联多表 = SQL `JOIN`；`concat` 沿轴堆叠 = `UNION ALL`（纵向）或并列拼列（横向）。
- **关键 API**：`pd.merge(l, r, on="key", how="inner|left|right|outer")`；多键 `on=["k1","k2"]`；纵向 `pd.concat([d1,d2], ignore_index=True)`；横向 `pd.concat([...], axis=1)`。
- **易错点**：① 忘记 `how` 默认是 `inner`，会静默丢掉不匹配行——想保全左表要显式 `how="left"`；② 两表存在同名非键列时会自动加 `_x/_y` 后缀，可用 `suffixes=(...)` 控制；③ `concat` 纵向拼接不加 `ignore_index=True` 会保留各自旧索引导致重复索引。
- **Java 视角**：`merge` ≈ SQL JOIN / 内存里的哈希连接；"找未下单用户" = `LEFT JOIN` 后筛 `order_id IS NULL`（`merged[merged["order_id"].isna()]`）。
- **前置**：01（DataFrame）；概念上呼应 04（聚合后再关联是常见组合）。

---

## 🎯 学习要点

- **建立 SQL 映射直觉**：过滤=WHERE、选列=SELECT、`sort_values`=ORDER BY、`groupby.agg`=GROUP BY、`merge`=JOIN——每学一个操作都对上一条 SQL。
- **`loc`/`iloc` 分清标签与位置**：默认用 `loc` 做条件写入（`df.loc[mask, "col"]=v`），既避开 `SettingWithCopyWarning` 又语义清晰。
- **I/O 就上性能参数**：读大文件默认带 `usecols` + `dtype`，落盘归档优先 Parquet + `zstd`，超内存文件走 `chunksize` 流式；能省的列和精度尽早省。
- **清洗形成固定套路**：`isnull().sum()` → 决定填充/删除 → `drop_duplicates` → `clip`/IQR 处理异常 → `astype`/`.str` 规整，跑一遍就是一条可复用清洗管线。
- **`agg` 与 `transform` 刻意对比**：需要"每组一行汇总"用 `agg`，需要"回填/组内标准化且保持行数"用 `transform`，做几道题固化差异。
- **合并先想清 `how` 和键唯一性**：动手前确认用哪种 JOIN、键是否唯一（一对多会膨胀行数），并检查合并后行数是否符合预期。

---

## 🔗 关联

- **上一模块**：[01-numpy](../01-numpy/README.md)（DataFrame 每列底层就是 ndarray）
- **下一模块**：[03-matplotlib](../03-matplotlib/README.md)（清洗好的数据交给可视化）
- **本阶段总览**：[阶段二 · 数据科学基础](../README.md)
- **Agent 课程衔接**：清洗/读写技能直接用于 [agent-course · Day 15 数据查询 Agent](../../agent-course/Day-15-data-query-agent.md) 与 [Day 19 嵌入 ETL Pipeline](../../agent-course/Day-19-embedding-etl.md)（读入 → 清洗 → 切分 → 入库）。
