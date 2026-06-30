# Day 31 · 多 Agent 三范式：graph / role / handoff

> **今日目标**：搞清三种多 Agent 编排范式（图编排 / 角色分工 / 直接交接）的区别、取舍与适用场景，各跑一个最小例子。
> **时长**：~2h ｜ **前置**：Day 26–30（StateGraph、条件边、Command）
> **今日产出**：一个 `day31_paradigms.py`，分别用 supervisor 和 handoff 两种方式让两个 Agent 协作，体会差异。

## 1. 为什么 & 是什么

单 Agent 啥都自己干，prompt 越堆越长、工具越挂越多，最后又笨又难调。**多 Agent** 的思路是**分而治之**：把大任务拆给各有专长的小 Agent。怎么"组织"这些 Agent，业界沉淀出三种范式：

| 范式 | 一句话 | Java 类比 | 控制权 | 适用 |
|---|---|---|---|---|
| **Graph（图/监督者）** | 一个 supervisor 节点决定"这步派给谁"，专家干完**回到** supervisor | 中心化的 `Orchestrator` / 网关路由 | 集中（supervisor 全程握权） | 多专家按需串联、要可控可追踪、研究/分析类 |
| **Role（角色/流水线）** | 预设固定角色和顺序，A→B→C 顺序流水 | 责任链 / 装配流水线 | 顺序（按预定义流程走） | 步骤固定的工序，如 研究员→分析师→报告 |
| **Handoff（交接/蜂群）** | Agent 之间**直接把控制权甩给对方**，没有中心 | 对象之间直接 `delegate()` 调用，去中心 | 分散（谁接手谁说了算） | 客服转接、对话要"换个人来谈"、灵活路由 |

直觉上：

```
Graph(Supervisor)         Role(Pipeline)            Handoff(Swarm)
                          
   ┌── Supervisor ──┐      研究员 → 分析师 → 报告      Triage ⇄ 专家A
   ▼     ▲    ▲     ▼      （固定单向流水）            ⇅      ⇅
  专家A  │    │    专家B                              专家B ⇄ 专家C
   └─────┘    └─────┘      
 （干完都回 supervisor）                              （彼此直接交接，无中心）
```

**关键认知**：这三种**不是互斥的框架，而是拓扑选择**——底层都能用 LangGraph 的 `StateGraph` + `Command` 表达：Graph 用条件边/handoff tool 派活、专家跑完连边回 supervisor；Role 就是 Day 26 那种写死的顺序边；Handoff 靠节点返回 `Command(goto="另一个agent", graph=Command.PARENT)` 直接转移控制权、不回中心。LangGraph 还提供 `langgraph-supervisor`（监督者）和 `langgraph-swarm`（蜂群/handoff）两个预制件，省去手搓。

## 2. 跟着做（Hands-on）

**Step 1 — Supervisor 范式（用预制件，最快上手）**

```bash
pip install -U langgraph-supervisor langgraph-swarm
```

```python
"""Day 31-A: Supervisor 范式 —— 一个监督者把活派给专家 / supervisor pattern."""

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def web_search(q: str) -> str:
    """联网检索（演示用桩）/ web search stub."""
    return f"[检索结果] 关于「{q}」：2026 年该领域增长显著……"

@tool
def calc(expr: str) -> float:
    """安全四则运算 / safe arithmetic."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
    def ev(n):  # 仅白名单，拒绝 eval / whitelist only, never eval()
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):    return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("不支持 / unsupported")
    return ev(ast.parse(expr, mode="eval").body)


# 两个专家 Agent：各管一摊 / two specialist agents
researcher = create_react_agent(
    model="openai:gpt-4o-mini", tools=[web_search],
    prompt="你是研究员，只负责检索资料。", name="researcher",
)
analyst = create_react_agent(
    model="openai:gpt-4o-mini", tools=[calc],
    prompt="你是分析师，负责计算与推理。", name="analyst",
)

# supervisor：决定每一步把活派给谁，专家干完回到它 / the supervisor routes work
team = create_supervisor(
    [researcher, analyst],
    model="openai:gpt-4o-mini",
    prompt=("你是团队主管。需要查资料就交给 researcher，需要算数就交给 analyst，"
            "拿到结果后整合成最终答复。"),
).compile()

if __name__ == "__main__":
    out = team.invoke({"messages": [HumanMessage(
        content="先查一下 2026 年 AI Agent 市场情况，再算 1500 * 1.3 是多少。")]})
    print(out["messages"][-1].content)
```

运行后观察消息流：supervisor 先把活甩给 researcher，再甩给 analyst，最后自己汇总——**控制权始终在 supervisor 手里**。

**Step 2 — Handoff 范式（用 Command 直接交接，看清机制）**

```python
"""Day 31-B: Handoff 范式 —— Agent 之间直接交接，无中心 / handoff via Command."""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def triage(state: State) -> Command:
    """分流 Agent：技术问题直接甩给 specialist，否则自己答 / hand off or answer."""
    text = state["messages"][-1].content
    if any(k in text for k in ("报错", "bug", "错误", "异常")):
        # ★ 返回 Command(goto=...)：控制权直接转移，不经过任何中心节点
        # transfer control straight to another agent — no central router
        return Command(goto="specialist",
                       update={"messages": [AIMessage(content="（转接技术专家…）")]})
    return Command(goto=END, update={"messages": [AIMessage(content="这是个一般问题，我来答：……")]})


def specialist(state: State) -> Command:
    """技术专家：处理被交接过来的技术问题 / handle the handed-off issue."""
    return Command(goto=END, update={"messages": [AIMessage(content="技术专家：你的报错通常是依赖版本不匹配，升级即可。")]})


builder = StateGraph(State)
builder.add_node("triage", triage)
builder.add_node("specialist", specialist)
builder.add_edge(START, "triage")
# 注意：没有"回 triage"的边——specialist 接手后自己走到 END / no edge back; specialist owns it
graph = builder.compile()

if __name__ == "__main__":
    for q in ["我的程序报错了怎么办", "今天天气适合出门吗"]:
        out = graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"Q: {q}\n→ {out['messages'][-1].content}\n")
```

> 对比关键：Supervisor 里专家干完**回中心**（中心再决定下一步）；Handoff 里 `triage` 把活甩给 `specialist` 后就**彻底交权**，`specialist` 自己走到 END。前者可控、好追踪；后者少一跳、更快，但容易"你交给我、我交给你"死循环——所以 swarm 一定要设 `recursion_limit` 兜底，并在每个 Agent 的 prompt 里写清"何时该交接、何时该自己答"。

## 3. 今日任务

1. 跑通 Supervisor 版，**打印完整 messages**，数清 supervisor 在专家之间切换了几次、每次派给谁。
2. 跑通 Handoff 版，确认技术问题走 specialist、普通问题 triage 直接答。
3. **造一个交接死循环**：写两个互相 `Command(goto=对方)` 的 Agent，设小 `recursion_limit`，确认会抛 `GraphRecursionError`——亲身体会 handoff 的风险。
4. **选型表**：用自己的话填一张"什么场景选 graph / role / handoff"的小表（至少各举一个你工作中可能遇到的例子）。

**验收标准**：两种范式都能跑；说得清"专家回中心 vs 直接交权"的差别；能复现并捕获 handoff 死循环；产出一张自己的选型表。

## 4. 自测清单

- [ ] 我能区分 graph / role / handoff 的控制权归属：supervisor 专家干完回中心，handoff 彻底交权。
- [ ] 我理解 `Command(goto=...)` 是实现 handoff 的机制。
- [ ] 我清楚 handoff/swarm 的最大风险是死循环，且知道怎么兜底。
- [ ] 我能根据任务特征选合适的范式。

## 5. 延伸 & 关联

- 明天用 **Role/Graph 范式**搭 研究员→分析师→报告 流水线（Day 32）；Day 33 再深入"多 Agent 怎么传数据、怎么避免状态污染"——这是多 Agent 真正难的地方。
- 本仓库相关章节：
  - LangChain Agent 与工具（单 Agent 是多 Agent 的细胞）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
  - 条件边/路由回看 Day 27（supervisor 本质是个超级路由）：[./Day-27-conditional-edges.md](./Day-27-conditional-edges.md)
