# Day 34 · CrewAI 速览：角色化框架，对比 LangGraph 取舍

> **今日目标**：用 CrewAI 这种"角色化"框架快速搭出多 Agent 协作，和昨天的 LangGraph 流水线对比，搞清各自的取舍与选型。
> **时长**：~2h ｜ **前置**：Day 31–33（多 Agent 范式、流水线、共享状态）
> **今日产出**：一个 `day34_crew.py`，用 CrewAI 复刻 研究员→撰写 协作，并写一段"何时选谁"的结论。

## 1. 为什么 & 是什么

LangGraph 是**显式状态机**：你画图、连边、管 state——强控制、强可观测，但**啰嗦**。很多任务其实就是"几个角色按部就班干活"，画图属实杀鸡用牛刀。

**CrewAI** 走另一条路：**角色扮演 + 声明式**。你不画图，而是描述"团队里有谁、各自什么职责、要交付什么"，框架自动协调。

它只有四个核心概念，对 Java 工程师极其友好——基本就是**给一个团队配 OKR**：

| CrewAI 概念 | 一句话 | Java / 职场类比 |
|---|---|---|
| **Agent** | 一个"员工"：有 role(岗位)、goal(KPI)、backstory(履历)、tools(技能) | 一个 `@Service` Bean，带职责描述 |
| **Task** | 派给某 Agent 的具体活：description(需求) + expected_output(验收标准) | 一张 Jira 工单 |
| **Crew** | 把 Agents + Tasks 装在一起的容器，决定执行流程 | 项目组 / `Orchestrator` |
| **Process** | 工作流引擎：`sequential`(顺序) / `hierarchical`(有经理动态派活) | 流程模式：流水线 vs 经理调度 |

**和 LangGraph 的根本差异**：
- LangGraph = **命令式**，你显式控制每一步流转（"先去A，满足X再去B"）。
- CrewAI = **声明式**，你描述角色和目标，把"怎么协作"交给框架。

类比：LangGraph 像你手写 `if/while` 的控制流；CrewAI 像写一份 Spring 配置 + 一堆 `@Task` 注解，让容器去编排。前者灵活到底，后者上手飞快。

## 2. 跟着做（Hands-on）

**Step 1 — 装 CrewAI（独立框架，不依赖 LangGraph）**

```bash
pip install -U crewai                    # 需 Python 3.10~3.13 / requires 3.10<=py<3.14
export OPENAI_API_KEY="sk-..."           # CrewAI 默认走 OpenAI / defaults to OpenAI
export CREWAI_TELEMETRY_OPT_OUT=true     # 生产/隐私：关遥测（环境变量控制）/ disable telemetry
```

**Step 2 — 用 Agent / Task / Crew 复刻 研究员→撰写**

```python
"""Day 34: CrewAI 角色化多 Agent —— 研究员 → 撰写 / role-based multi-agent."""

from crewai import Agent, Task, Crew, Process


# 1) 定义"员工"：role/goal/backstory 就是给 LLM 的角色设定
#    define agents: role/goal/backstory shape each agent's persona
researcher = Agent(
    role="资深市场研究员",
    goal="围绕 {topic} 挖出关键事实与数据",
    backstory="你有 20 年行业研究经验，擅长快速定位核心信息。",
    verbose=True,
    allow_delegation=False,   # 不许把活转包给别人 / no delegation
)

writer = Agent(
    role="内容撰写者",
    goal="基于研究结果写出清晰的中文简报",
    backstory="你是资深科技撰稿人，擅长把复杂信息讲明白。",
    verbose=True,
    allow_delegation=False,
)

# 2) 定义"工单"：description=需求，expected_output=验收标准
#    define tasks: description = the ask, expected_output = the bar
research_task = Task(
    description="研究「{topic}」，列出 5 条关键事实，注明大致来源。",
    expected_output="5 条带来源的要点清单。",
    agent=researcher,
)

write_task = Task(
    description="基于研究结果，就「{topic}」写一份 200 字以内的中文简报（标题/要点/结论）。",
    expected_output="一份结构完整的简报。",
    agent=writer,
    context=[research_task],   # ★ 关键：声明依赖上游任务的产出，框架自动把它注入上下文
                               # declare dependency → upstream output is injected as context
)

# 3) 组队：sequential 模式 = 按任务顺序逐个执行（≈ 昨天的流水线）
#    assemble the crew; sequential = run tasks in order
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)   # 关遥测靠环境变量 CREWAI_TELEMETRY_OPT_OUT，不是 Crew 参数 / opt out via env var


if __name__ == "__main__":
    # kickoff 注入变量、启动协作 / kickoff injects vars and runs the crew
    result = crew.kickoff(inputs={"topic": "2026 年 AI Agent 在企业客服的落地"})
    print("\n===== 最终产出 =====\n", result)
```

运行：`python day34_crew.py`。对比昨天 LangGraph 版——**同样是 研究→撰写，但你一条边都没画**，只声明了角色、工单和依赖（`context`）。

**Step 3 — 体会 hierarchical（有"经理"动态派活）**

```python
# 把 process 换成 hierarchical：CrewAI 自动加一个 manager agent 动态调度
# hierarchical: a manager agent dynamically delegates — 类似 LangGraph 的 supervisor
crew_h = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o-mini",   # 经理用哪个模型决策 / model the manager uses to route
    verbose=True,
)
# kickoff 后观察：经理会决定先调谁、要不要返工 / the manager decides ordering
```

> 对应关系一目了然：CrewAI 的 `sequential` ≈ 昨天 LangGraph 的固定流水线（Day 32）；`hierarchical` ≈ Day 31 的 supervisor 范式。**同样的编排思想，两种表达方式。**

## 3. 今日任务

1. 跑通 `day34_crew.py`，确认两个 Agent 顺序协作、`writer` 确实用到了 `researcher` 的产出（去掉 `context=[research_task]` 再跑，对比 writer 是否"瞎写"）。
2. **加第三个角色**：插一个"审校 editor" Agent + Task，`context` 依赖前两个，形成 研究→撰写→审校 三段。
3. **试 hierarchical**：把 process 换成 hierarchical，观察 manager 的调度日志，和你 Day 31 的 supervisor 体感对比。
4. **写选型结论**（核心产出）：用自己的话写 5~8 行"CrewAI vs LangGraph 何时选谁"，至少覆盖：控制粒度、可观测性、上手速度、状态/持久化/HITL 的掌控力这几个维度。

**验收标准**：CrewAi 三角色流水线跑通且 `context` 依赖生效；hierarchical 能看到 manager 调度；产出一份有维度、有依据的选型结论。

## 4. 自测清单

- [ ] 我能用"给团队配 OKR"解释 CrewAI 的 Agent/Task/Crew/Process。
- [ ] 我知道 `context=[...]` 是 CrewAI 里 Agent 间传递产出的机制。
- [ ] 我能把 CrewAI 的 sequential/hierarchical 对应到 LangGraph 的流水线/supervisor。
- [ ] 我说得清"声明式（CrewAI）vs 命令式（LangGraph）"的根本差异。
- [ ] 我有自己的选型判断，而不是"哪个新用哪个"。

## 5. 延伸 & 关联

- **选型速记**：要**强控制、复杂分支、断点续跑、细粒度 HITL、深度可观测** → LangGraph；要**快速起步、角色清晰、流程相对线性** → CrewAI。两者都能落地，差异在"你想自己握多少控制权"。社区数据也印证：CrewAI 在简单流程上更快上手，LangGraph 在复杂任务上更稳更可控。
- 明天：把这几天的多 Agent 代码**复盘重构**，抽出可复用结构——不管用哪个框架，好的抽象都是相通的。
- 本仓库相关章节：
  - LangChain 完整应用（另一种高层封装思路）：[../07-llm-applications/05-langchain/03-full-application.md](../07-llm-applications/05-langchain/03-full-application.md)
  - 多 Agent 范式回看 Day 31（看 CrewAI 落在哪个范式）：[./Day-31-multi-agent-paradigms.md](./Day-31-multi-agent-paradigms.md)
