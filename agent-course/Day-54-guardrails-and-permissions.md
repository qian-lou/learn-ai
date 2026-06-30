# Day 54 · 安全②：工具权限边界、输出校验、guardrails、敏感数据

> **今日目标**：把安全防线从"输入"延伸到"动作与输出"——给工具划权限边界、对输出上 guardrails、对敏感数据做脱敏。
> **时长**：~2h ｜ **前置**：Day 53（prompt 注入）、Day 10（工具错误处理）
> **今日产出**：一个 `safe_agent.py`，演示工具最小权限 + 危险操作二次确认 + 输出 guardrail（PII/有害内容）+ 敏感数据脱敏。

## 1. 为什么 & 是什么

Day 53 解决"别被劫持"；今天解决"**就算被劫持，也炸不了**"。这是纵深防御的后半段——**对应 OWASP LLM 的 LLM06 过度代理（Excessive Agency）、LLM05 不当输出处理、LLM02 敏感信息泄露**（Day 55 过完整清单）。

为什么单防输入不够？因为防御从来不是 100%。攻击者总能找到新的注入方式，所以你必须假设"指令层迟早会被绕过一次"，然后让**第二道闸**（动作权限、输出校验）把伤害挡住。给 Java 工程师的对照：

| 今日概念 | Java / 安全世界 | 说明 |
|---|---|---|
| **工具最小权限** | RBAC / 最小权限原则 | Agent 只配它需要的工具，危险操作严格收口 |
| **危险操作二次确认** | 关键操作的人工审批 / 二次校验 | 删库/转账/发邮件前，HITL 或硬规则拦一道 |
| **输出 guardrails** | 出参校验 / 响应过滤器 | 拦 PII、有害内容、越权信息 |
| **敏感数据脱敏** | 日志脱敏 / 数据掩码 | 入模型前匿名化，出来再还原 |
| **沙箱执行** | 容器隔离 / 受限运行时 | 代码/命令类工具在隔离环境跑 |

**核心心智：把 Agent 当成一个"可能被社工的实习生"。** 你不会给实习生生产数据库的删除权限，也不会让他未经审批就给客户转账。同样地——**Agent 能调用的工具就是它的"权限"，必须按最小权限发放，危险动作必须加闸。**

## 2. 跟着做（Hands-on）

### Step 1 — 工具最小权限 + 危险操作收口

```bash
pip install "openai>=1.0" "pydantic>=2"
```

```python
"""Day 54: 工具权限边界 + 危险操作确认 / tool permission boundary + confirmation."""

from typing import Callable, Dict

# 工具按风险分级，高危工具默认需要人工确认 / high-risk tools need approval
DANGEROUS_TOOLS = {"send_email", "delete_record", "transfer_money"}  # 有副作用 / side effects


def get_weather(city: str) -> str:
    """只读工具，无副作用 / read-only tool."""
    return f"{city} 晴 22°C"


def send_email(to: str, body: str) -> str:
    """高危工具，真的会发邮件 / dangerous tool with side effects."""
    return f"[已发送] to={to} body={body[:20]}..."


REGISTRY: Dict[str, Callable] = {"get_weather": get_weather, "send_email": send_email}


def dispatch(tool_name: str, args: dict, *, human_approved: bool = False) -> str:
    """带权限闸的工具分发；越权工具抛 PermissionError / permission-gated dispatch."""
    # 闸 1：白名单——模型只能调注册过的工具，杜绝"凭空捏造工具名"
    if tool_name not in REGISTRY:
        raise PermissionError(f"未授权的工具 / unauthorized tool: {tool_name}")

    # 闸 2：高危工具必须有人工批准（HITL）/ dangerous ops require approval
    if tool_name in DANGEROUS_TOOLS and not human_approved:
        return f"[已拦截] 高危操作 {tool_name} 需人工确认 / blocked: needs human approval"

    return REGISTRY[tool_name](**args)


if __name__ == "__main__":
    print(dispatch("get_weather", {"city": "上海"}))                  # 放行 / allowed
    print(dispatch("send_email", {"to": "a@b.com", "body": "hi"}))     # 拦截：未批准 / blocked
    print(dispatch("send_email", {"to": "a@b.com", "body": "hi"}, human_approved=True))  # 批准放行
    # dispatch("rm_rf_root", {}) → 抛 PermissionError（越权工具）/ raises on unauthorized tool
```

**要点**：白名单（杜绝模型臆造工具）+ 风险分级（高危需 HITL）。这就是 Day 30 人在回路在安全语境下的落地——**破坏性动作永远不让模型独自拍板**。

### Step 2 — 输出 guardrails：拦 PII / 有害内容

生产里用成熟库，而不是自己写正则。2026 当前做法（`Guard().use(...)`，**老的 RAIL XML 已弃用**）：

```bash
pip install guardrails-ai
guardrails configure                                  # 配置 Hub key（免费）
guardrails hub install hub://guardrails/detect_pii    # 从 Hub 装 validator
guardrails hub install hub://guardrails/toxic_language
```

```python
"""Day 54 Step2: 用 Guardrails AI 校验输出 / validate output with Guardrails AI."""

from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage  # 装好后从 hub 导入

# 组合多个 validator：PII 自动脱敏，有害内容直接抛异常
# compose validators: fix PII, raise on toxic content
guard = Guard().use(
    DetectPII(["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.FIX),
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION),
)

if __name__ == "__main__":
    outcome = guard.validate("有问题请联系 john@example.com 或 13800138000。")
    print(outcome.validated_output)  # FIX 模式下邮箱/电话被脱敏 / PII redacted
```

> `on_fail` 可选 `fix`（脱敏/修正）、`exception`（抛错拦截）、`reask`（让模型重答）、`refrain`（拒答）。底层 PII 检测用微软 Presidio。

### Step 3 — 敏感数据脱敏：入模型前匿名、出来后还原

对"必须给模型看，但不能泄露"的数据（用户姓名、手机号），用 **Presidio** 或 **LLM Guard** 在入模型前打码：

```python
# pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_sm
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()


def anonymize(text: str) -> str:
    """把文本里的 PII 替换成占位符再交给模型 / strip PII before the LLM sees it."""
    results = analyzer.analyze(text=text, language="en")  # 识别 PII 实体 / detect entities
    return anonymizer.anonymize(text=text, analyzer_results=results).text


# anonymize("My name is John Smith, email john.smith@example.com")
# → "My name is <PERSON>, email <EMAIL_ADDRESS>"
```

> **另一个利器：LLM Guard**（`pip install llm-guard`），`scan_prompt` / `scan_output` 一站式串起"注入扫描 + 匿名化 + 输出泄露检测"，`Anonymize`+`Vault` 还能在输出阶段把占位符**还原**回真实值——适合"脱敏进、还原出"的闭环。

**纵深防御总图（输入→处理→输出全链路）：**

```
用户/外部内容 ─[Day53: 注入防御]→ ─[脱敏]→ LLM ─[工具白名单+HITL]→ 动作
                                                  └─[输出 guardrail: PII/有害]→ ─[还原]→ 用户
```

## 3. 今日任务

1. 跑通 Step 1，确认：只读工具放行、高危工具未批准被拦、批准后放行、臆造工具名抛 `PermissionError`。
2. 跑通 Step 2（或 Step 3），让一段含邮箱/电话的文本被自动脱敏；再喂一段"有害内容"确认被 `exception` 拦截。
3. **串成闭环**：把 Day 53 的注入防御 + 今天的工具白名单 + 输出 guardrail 接成一条链，跑一个"被注入但没造成破坏"的端到端 demo。
4. **设计你的权限矩阵**：为你的研究 Agent（Day 41~45）列一张表——每个工具的风险级别、是否需 HITL、输出要不要过 guardrail。

**验收标准**：工具权限闸四种情况都正确；PII 能被脱敏、有害内容能被拦；端到端 demo 证明"即便指令被绕过，破坏性动作仍被挡住"；产出一张工具权限矩阵。

## 4. 自测清单

- [ ] 我理解"防输入不够，还要防动作与输出"，并能说出对应的 OWASP 项。
- [ ] 我会用白名单 + 风险分级 + HITL 给工具划权限边界。
- [ ] 我知道破坏性操作绝不让模型独自拍板。
- [ ] 我会用 Guardrails AI（`.use()`）/ Presidio / LLM Guard 做输出校验与脱敏。
- [ ] 我能把注入防御 + 工具权限 + 输出 guardrail 串成一条纵深防线。

## 5. 延伸 & 关联

- 明天用 **OWASP LLM Top 10**（今天对应 LLM02/05/06）逐条自查：[Day-55-owasp-llm-top10.md](./Day-55-owasp-llm-top10.md)；工具权限的"人工确认"正是 Day 30 人在回路（HITL）在安全语境的应用。
- 本仓库相关章节：
  - 评估与监控（把"拦截率/脱敏覆盖率"做成可监控指标）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
  - API 服务开发（鉴权 / 限流 / 输入校验等服务层防护）：[../08-llm-engineering/02-model-serving/02-api-service.md](../08-llm-engineering/02-model-serving/02-api-service.md)
