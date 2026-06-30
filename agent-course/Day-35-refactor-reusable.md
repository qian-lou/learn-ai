# Day 35 · 复盘 + 重构：把多 Agent 系统抽成可复用结构

> **今日目标**：复盘 Day 26–34 的代码，把散落的多 Agent 逻辑重构成清晰、可复用、可测试的结构，沉淀你自己的"编排脚手架"。
> **时长**：~2h ｜ **前置**：Day 26–34（LangGraph 全家桶 + CrewAI）
> **今日产出**：一个 `agent_kit/` 小模块（state schema / 节点工厂 / 建图函数分层），以及一份"多 Agent 工程 checklist"。

## 1. 为什么 & 是什么

学到这，你能搭出能跑的多 Agent 系统了——但大概率代码长这样：state schema、节点函数、建图、模型初始化**全挤在一个文件里**，复制粘贴改改就上。这在 demo 阶段没问题，**但这正是"只会写 demo 的人"和工程师的分水岭**。今天不学新 API，专门练**把它整理干净**。

借你 Java 的工程直觉，做三件事：

1. **分层（Separation of Concerns）**：state 定义、节点逻辑、建图装配、模型配置——各归各位。等价于 Java 的 `domain / service / config / assembler` 分层。
2. **抽可复用件（DRY）**：重复出现的模式（带工具的子 Agent、"只取结论"的回填、防死循环的 invoke）抽成工厂函数/工具函数。等价于抽 `BaseService` / 工具类。
3. **可测试 + 可配置**：节点纯函数化（输入 state、输出增量，不偷偷读全局），模型/参数从外部注入。等价于依赖注入 + 单测友好。

一句话目标：**让"加一个新 Agent / 换一个模型 / 调一条流程"从'改一大坨'变成'改一个小函数或一行配置'。**

## 2. 跟着做（Hands-on）

**Step 1 — 分层：拆出 `agent_kit/`**

把"一坨"拆成职责清晰的几块（先建目录，按下面四个文件归位）：

```
agent_kit/
├── state.py      # 只放 state schema（数据契约）/ data contracts only
├── nodes.py      # 节点逻辑（纯函数 + 工厂）/ node logic
├── graphs.py     # 建图装配（把节点连成 graph）/ graph assembly
└── llm.py        # 模型/配置集中管理 / centralized model config
```

```python
# ===== agent_kit/llm.py =====
"""集中管理模型，换模型只改这一处 / single place to swap models."""
from functools import lru_cache
from langchain_openai import ChatOpenAI

@lru_cache(maxsize=8)   # 缓存实例，避免每个节点重复 new / cache client instances
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """按名取模型实例 / get a cached chat model."""
    return ChatOpenAI(model=model, temperature=temperature)
```

```python
# ===== agent_kit/state.py =====
"""数据契约：所有 graph 共享的 state 形状 / shared state schemas."""
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class PipelineState(TypedDict):
    topic: str
    findings: Annotated[list[str], add]                 # 可并行追加 / append-safe
    analysis: str
    report: str
    messages: Annotated[list[AnyMessage], add_messages]
```

**Step 2 — 抽工厂：把"重复的节点模式"参数化**

```python
# ===== agent_kit/nodes.py =====
"""节点工厂：把"重复出现的节点模式"抽成可复用函数 / reusable node factories."""
from typing import Callable
from langchain_core.messages import HumanMessage, AIMessage
from .llm import get_llm
from .state import PipelineState


def make_llm_node(field: str, prompt_tmpl: str, model: str = "gpt-4o-mini") -> Callable:
    """工厂：生产一个"读 state → 调 LLM → 写指定字段"的节点。

    Factory: build a node that reads state, calls the LLM, writes one field.
    这样 研究员/分析师/报告员 只是"换 field + 换 prompt"，不再各写一遍。
    Args: field=写入的 state 字段；prompt_tmpl=用 {占位} 引用 state 的模板；model=模型名。
    Returns: 一个 (state) -> dict 的节点函数 / a node function.
    """
    def node(state: PipelineState) -> dict:
        prompt = prompt_tmpl.format(**state)          # 用 state 填模板 / fill from state
        text = get_llm(model).invoke(prompt).content
        return {field: text, "messages": [AIMessage(content=f"[{field}] done")]}
    return node


def extract_conclusion(sub_result: dict) -> str:
    """复用件：从子 Agent 结果里只取最终结论（防状态污染，见 Day 33）。

    Reusable: pull only the final conclusion from a sub-agent result.
    """
    return sub_result["messages"][-1].content
```

**Step 3 — 装配 + 可复用的安全 invoke**

```python
# ===== agent_kit/graphs.py =====
"""建图装配 + 安全运行封装 / graph assembly + safe runner."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from .state import PipelineState
from .nodes import make_llm_node


def build_research_pipeline(checkpointer=None):
    """装配 研究员→分析师→报告 流水线；checkpointer 可选注入。

    Assemble the research pipeline; optionally inject a checkpointer.
    """
    b = StateGraph(PipelineState)
    # 三个节点全部由工厂产出，只是 field+prompt 不同 / all three from the same factory
    b.add_node("researcher", make_llm_node(
        "findings", "你是研究员，围绕「{topic}」列 5 条关键事实。"))
    b.add_node("analyst", make_llm_node(
        "analysis", "你是分析师，基于资料提炼 3 条洞察：\n{findings}"))
    b.add_node("writer", make_llm_node(
        "report", "你是撰稿人，把洞察写成 200 字简报：\n{analysis}"))
    b.add_edge(START, "researcher")
    b.add_edge("researcher", "analyst")
    b.add_edge("analyst", "writer")
    b.add_edge("writer", END)
    return b.compile(checkpointer=checkpointer or InMemorySaver())


def run_safely(graph, init: dict, thread_id: str = "default", limit: int = 25):
    """复用件：带 thread_id + 防死循环的统一运行入口 / one safe entry point."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": limit}
    return graph.invoke(init, cfg)
```

```python
# ===== 用起来：main.py / putting it together =====
from agent_kit.graphs import build_research_pipeline, run_safely

graph = build_research_pipeline()
out = run_safely(graph,
                 {"topic": "向量数据库选型", "findings": [],
                  "analysis": "", "report": "", "messages": []},
                 thread_id="demo-1")
print(out["report"])
```

> 重构后的体感：**加一个新阶段** = 加一行 `make_llm_node(...)` + 一条边；**换模型** = 改 `llm.py` 一处；**改运行策略** = 动 `run_safely`。这就是"可复用结构"的价值——和你在 Spring 里抽 `BaseService`、用工厂/DI 是同一套工程审美。

## 3. 今日任务

1. **落地分层**：把你 Day 26–32 的代码按 `agent_kit/`（state/nodes/graphs/llm）重新归位，跑通流水线，行为和重构前一致。
2. **抽至少 2 个复用件**：除了 `make_llm_node`，再抽一个你自己重复写过的模式（如"带工具的子 Agent 工厂"或"只取结论的回填"）。
3. **加可配置**：让 `build_research_pipeline` 支持外部传入每个阶段的 prompt / 模型（不改函数体就能调流程）。
4. **写工程 checklist**（核心产出）：基于这 10 天的坑，列一份"我搭多 Agent 系统前要自查的清单"（10 条左右，参考下面的种子）。

**验收标准**：代码完成分层且行为不变；抽出 ≥2 个可复用件；流水线的 prompt/模型可外部配置；产出一份属于你自己的多 Agent 工程 checklist。

## 4. 自测清单

- [ ] 我的 state / 节点 / 建图 / 模型配置已经分层，不再挤在一个文件。
- [ ] 我把重复的节点模式抽成了工厂函数，节点是纯函数（不偷读全局）。
- [ ] "加一个 Agent / 换模型 / 调流程"现在只需小改动。
- [ ] 我沉淀了下面这份可复用的多 Agent 工程 checklist。

**🌱 多 Agent 工程 checklist（核心产出，自行增删）**
- [ ] state 里公共总线 vs 私有字段划清了吗？多方写的字段配对 reducer 了吗？
- [ ] 子 Agent 的结果是"只取结论"回填，还是把内部噪音全灌进来了？
- [ ] 每个会循环/交接的图都设了 `recursion_limit` 吗？关键写操作有 `interrupt` 闸吗？
- [ ] 生产用落库 checkpointer（SQLite/Postgres）而非内存吗？
- [ ] LLM 当路由器处输出收敛到枚举且有兜底吗？模型/prompt 是集中可配置吗？

## 5. 延伸 & 关联

- **Phase 3 上半程到此收官**：你已掌握 LangGraph 编排（状态/分支/循环/持久化/HITL）+ 多 Agent（三范式/流水线/共享状态）+ CrewAI 对比，并能把它工程化。
- 接下来 Day 36–40 进入**健壮性与协议**（重试/死循环检测/超时、MCP、A2A、性能），Day 41–45 用这套地基搭**主力项目「自动化研究 Agent」**——今天的 `agent_kit/` 就是那个项目的脚手架。
- 本仓库相关章节：
  - MLOps / 实验追踪（重构后的系统更易接监控与评估）：[../08-llm-engineering/03-mlops/01-experiment-tracking.md](../08-llm-engineering/03-mlops/01-experiment-tracking.md)
  - 多 Agent 流水线与共享状态回看 Day 32–33：[./Day-32-multi-agent-pipeline.md](./Day-32-multi-agent-pipeline.md)
  - 本系列总计划（看你在 70 天里的位置）：[../AI-Agent-每日学习计划.md](../AI-Agent-每日学习计划.md)
