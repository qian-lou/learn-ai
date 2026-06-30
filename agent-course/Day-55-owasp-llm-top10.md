# Day 55 · OWASP for LLM：逐条自查你的 Agent

> **今日目标**：用 **OWASP Top 10 for LLM Applications（2025）** 这把行业标尺，把你前面所有 Agent 逐条体检，产出一份自查报告。
> **时长**：~2h ｜ **前置**：Day 53（注入）、Day 54（权限/输出/脱敏）、Day 41~45（你的研究 Agent）
> **今日产出**：一份 `owasp_audit.md` 自查清单（对你的研究 Agent 逐条打分：已防护/部分/未防护 + 整改项），外加一个把清单跑成 CI 检查的小脚本骨架。

## 1. 为什么 & 是什么

前两天你学了注入、权限、guardrails——都是**单点**。今天换个视角：用一份**业界公认的全景清单**做系统性体检，确保没有漏掉的攻击面。

给 Java 工程师的对照：这就是 **OWASP Top 10（Web）** 的 LLM 版。你做 Web 安全会拿 OWASP 清单逐条核对（SQL 注入、XSS、CSRF……）；做 LLM 应用，对应的标尺就是 **OWASP Top 10 for LLM Applications**。**2025 版**（截至 2026 年仍是当前版）相比早期版本有重要更新——新增了"系统提示泄露""向量与嵌入弱点"，把"模型拒绝服务"扩展为"无限消耗"。

**核心心智：安全不是做一两个防护就完事，而是对照清单系统性地查漏。** 一条没防住，整个 Agent 就有缺口。今天的产出是一份**可复用的自查表**——以后每上线一个 Agent，都拿它过一遍。

## 2. 跟着做（Hands-on）

### OWASP Top 10 for LLM Applications（2025）逐条自查表

下面每条给出：**它是什么 → Agent 里的典型表现 → 你该怎么防（关联本课哪天）**。对照你的研究 Agent，逐条标注 ✅已防护 / ⚠️部分 / ❌未防护。

| ID | 标题 | 是什么 & 怎么防（关联本课） |
|---|---|---|
| **LLM01:2025** | **Prompt Injection（提示注入）** | 直接/间接注入劫持指令。防：分隔不可信数据、声明"数据非指令"、输出校验。→ Day 53 |
| **LLM02:2025** | **Sensitive Information Disclosure（敏感信息泄露）** | 模型吐出 PII、密钥、他人数据。防：入模型前脱敏、输出 guardrail 扫泄露、最小化上下文中的敏感数据。→ Day 54 |
| **LLM03:2025** | **Supply Chain（供应链）** | 第三方模型/插件/依赖/数据集被投毒或有漏洞。防：固定依赖版本、来源可信、SBOM、扫描第三方工具与 MCP server。→ Day 37 |
| **LLM04:2025** | **Data and Model Poisoning（数据与模型投毒）** | 训练/微调/RAG 语料被污染，植入后门或偏见。防：数据源审计、RAG 入库内容清洗与签名、隔离不可信来源。→ Day 19/23 |
| **LLM05:2025** | **Improper Output Handling（不当输出处理）** | 把模型输出**未经校验**直接喂给下游（执行 SQL/命令、渲染 HTML）。防：把输出当不可信、参数化、转义、结构化校验。→ Day 04/54 |
| **LLM06:2025** | **Excessive Agency（过度代理）** | Agent 工具权限过大，被诱导执行破坏性动作。防：工具最小权限、高危操作 HITL、白名单、沙箱。→ Day 54 |
| **LLM07:2025** | **System Prompt Leakage（系统提示泄露）** | 系统提示被套话泄露，暴露规则/密钥/绕过手段。防：**别把秘密放进 system prompt**、输出侧检测泄露、假设它终将泄露。→ Day 53 |
| **LLM08:2025** | **Vector and Embedding Weaknesses（向量与嵌入弱点）** | RAG 检索被投毒文档/嵌入反演攻击；跨租户向量泄露。防：来源隔离、检索内容过滤、向量库访问控制。→ Day 17/23 |
| **LLM09:2025** | **Misinformation（错误信息）** | 模型自信地胡说（幻觉），用户过度信任。防：RAG 引用溯源、"不知道就说不知道"、幻觉率 eval、关键场景人工复核。→ Day 22/50 |
| **LLM10:2025** | **Unbounded Consumption（无限消耗）** | 无限制的请求/超长上下文/递归调用拖垮服务或刷爆账单（含 DoS、钱包攻击）。防：限流、token 上限、超时、死循环检测、成本预算。→ Day 36/51 |

> **2025 版相对旧版的关键变化**（别再讲过时条目）：新增 **LLM07 系统提示泄露**、**LLM08 向量与嵌入弱点**；**LLM10** 从"模型拒绝服务"扩展为"无限消耗"；**LLM09** 从"过度依赖"演化为"错误信息"；旧版的"不安全插件设计""模型窃取"被并入供应链/过度代理。

### 把自查跑成 CI 检查（骨架）

清单不该只躺在文档里——把能自动化的项做成**每次发布前必跑的检查**：

```python
"""Day 55: OWASP 自查 CI 骨架 / a tiny OWASP self-audit harness.

把"可自动化"的 OWASP 项做成断言；人工项标 TODO 留给评审。
Encode the automatable OWASP items as checks; flag manual ones for review.
"""

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Check:
    """一条 OWASP 自查项 / one audit item."""

    owasp_id: str
    name: str
    probe: Callable[[], bool]  # 返回 True=通过 / True means defended


# 下面用占位函数示意；真实项目里换成对你 Agent 的实际探测
# placeholders — wire these to real probes against your agent
def has_injection_defense() -> bool:
    """LLM01：注入测试集的拦截率是否达标 / injection block-rate >= threshold."""
    return True  # TODO: 跑 Day53 的注入测试集，统计拦截率


def secrets_not_in_system_prompt() -> bool:
    """LLM07：系统提示里是否不含密钥 / no secrets in system prompt."""
    return True  # TODO: 扫描 system prompt 是否命中密钥正则


def has_rate_limit_and_token_cap() -> bool:
    """LLM10：是否配置了限流 + token 上限 + 超时 / consumption bounds set."""
    return True  # TODO: 检查配置项是否齐全


CHECKS: List[Check] = [
    Check("LLM01", "Prompt Injection", has_injection_defense),
    Check("LLM07", "System Prompt Leakage", secrets_not_in_system_prompt),
    Check("LLM10", "Unbounded Consumption", has_rate_limit_and_token_cap),
    # ...逐条补齐能自动化的项 / add the rest you can automate
]


def run_audit() -> int:
    """跑全部自查，返回未通过数（用作 CI 退出码）/ returns failure count.

    Returns:
        未通过的检查数量；0 表示全过 / number of failed checks.
    """
    failed = 0
    for c in CHECKS:
        ok = c.probe()
        print(f"{'✅' if ok else '❌'} {c.owasp_id} {c.name}")
        failed += 0 if ok else 1
    print(f"\n未通过 / failed: {failed}/{len(CHECKS)}")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_audit() else 0)  # 有未通过则 CI 失败 / non-zero exit fails CI
```

> 自动化能覆盖一部分（注入拦截率、限流配置、密钥扫描），但**供应链、数据投毒、过度代理**等很多要靠**人工评审 + 架构审查**。把可自动化的接进 CI，人工项写进 release checklist。

## 3. 今日任务

1. **逐条体检**：拿你的研究 Agent（Day 41~45），对 10 条逐一标注 ✅/⚠️/❌，写进 `owasp_audit.md`。
2. **挑两个最弱项整改**：选你标 ❌ 的两条，落地一个具体修复（如 LLM07 把密钥移出 system prompt；LLM10 加 token 上限 + 超时），并验证。
3. **跑通自查骨架**：把上面脚本里至少 3 个 `probe` 接成对你 Agent 的**真实**探测（哪怕粗糙），让 `run_audit` 输出真实结果。
4. **写一句话风险结论**：用一句话给你的 Agent 定个安全等级（如"可内部试用，上线前需补 LLM03/08"），并列出上线前的硬性整改清单。

**验收标准**：产出覆盖全部 10 条的自查表；至少完成两项实际整改并验证；自查脚本能对你的 Agent 跑出真实结果；有明确的"上线前整改清单"。

## 4. 自测清单

- [ ] 我能背出 OWASP LLM Top 10（2025）的 10 个条目大意。
- [ ] 我知道 2025 版的关键变化（新增系统提示泄露、向量弱点；无限消耗等）。
- [ ] 我能把每一条映射到本课对应的防护手段。
- [ ] 我对自己的 Agent 做过逐条体检，并知道最弱的两三项。
- [ ] 我能区分"可自动化进 CI"和"必须人工评审"的安全项。

## 5. 延伸 & 关联

- 本段（Day 46~55 生产化）到此完结：可观测（46~48）+ 评估（49~50）+ 成本（51~52）+ 安全（53~55）。接下来 Day 56~60 把它们**整合进可部署服务**。
- 官方清单原文：搜索 "OWASP Top 10 for LLM Applications 2025"（genai.owasp.org），建议收藏，每次上线对照。
- 本仓库相关章节：
  - 评估与监控（安全指标也应纳入持续监控）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - API 服务开发（限流 / 鉴权 / 输入校验是 LLM10/LLM02 的服务层落地）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
  - CI/CD（把 eval + 安全自查接进流水线）：[../08-llm-engineering/03-mlops/03-cicd.md](../08-llm-engineering/03-mlops/03-cicd.md)
