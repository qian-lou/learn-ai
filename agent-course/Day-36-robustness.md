# Day 36 · Agent 健壮性：重试 / 死循环 / token 超限 / 整体超时

> **今日目标**：把一个"happy path 能跑"的 agent 改造成"线上不会失控"的 agent，掌握四道防线。
> **时长**：~2h ｜ **前置**：Day 26–28（LangGraph 状态机与循环）、Day 10（工具错误处理）
> **今日产出**：一个 `day36_robust_agent.py`，自带「重试 + 死循环检测 + token 预算 + 整体超时」，在工具反复失败时优雅终止而非崩溃或烧钱。

## 1. 为什么 & 是什么

Demo 和生产 agent 的差距，80% 在这一天。LLM 的循环本质上是**不可信的外部系统在驱动你的控制流**——它可能反复调用同一个失败的工具、陷入"思考→行动→思考"的死循环、或把上下文越堆越大直到超出窗口。生产 agent 必须有**四道独立的防线**：

| 防线 | 解决的问题 | Java 类比 |
|---|---|---|
| **重试 + 退避** | 工具/网络瞬时失败（429、超时） | Spring Retry `@Retryable` + 指数退避；Resilience4j |
| **死循环检测** | 模型反复做同一个无效动作 | 熔断器（CircuitBreaker）+ 最大步数兜底 |
| **token 预算** | 上下文无限膨胀 → 撞窗口 / 烧钱 | 连接池/线程池的**容量上限**，满了就拒绝 |
| **整体超时** | 单次 run 永远不返回 | `CompletableFuture.orTimeout(...)`；Hystrix 超时 |

关键心智：**这四个是正交的**。重试管"单步瞬时失败"，死循环检测管"多步在原地打转"，token 预算管"空间维度膨胀"，整体超时管"时间维度失控"。少任何一个，都有一类故障能让 agent 失控。

> 退避策略首选**指数退避 + 抖动（jitter）**：第 n 次等待 `base * 2^n + random()`。抖动是为了避免大量客户端在同一时刻重试造成"惊群"（thundering herd）——这点和高并发后端完全一致。

## 2. 跟着做（Hands-on）

用 `tenacity`（Python 事实标准重试库）+ 一个轻量 agent 循环演示四道防线。

```bash
pip install "openai>=1.40" "tenacity>=9.0"
```

```python
"""Day 36: 健壮 agent 循环 / a robust agent loop with four guardrails."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

client = OpenAI()


# ---- 防线 1：重试 + 指数退避 + 抖动 / retry with exponential backoff + jitter ----
class TransientToolError(RuntimeError):
    """可重试的瞬时错误（如 429、网络抖动）/ retryable transient error."""


@retry(
    retry=retry_if_exception_type(TransientToolError),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),  # 0.5s 起，封顶 8s，自带抖动
    stop=stop_after_attempt(4),                          # 最多 4 次 / at most 4 tries
    reraise=True,                                        # 耗尽后抛原始异常 / re-raise on exhaustion
)
def call_flaky_tool(query: str) -> str:
    """模拟一个 30% 概率瞬时失败的外部工具 / a tool that fails transiently 30% of the time."""
    if random.random() < 0.3:  # noqa: S311 — 演示用 / demo only
        raise TransientToolError("upstream 503")
    return f"结果({query})"


# ---- 防线 2/3/4：预算 + 死循环 + 整体超时统一收口到一个 RunBudget ----
@dataclass
class RunBudget:
    """单次 run 的资源预算守卫 / per-run resource budget guard."""

    max_steps: int = 8                    # 步数上限（兜底死循环）/ hard step cap
    max_tokens: int = 20_000              # token 预算 / token budget
    deadline_s: float = 30.0              # 整体超时 / overall wall-clock timeout
    _start: float = field(default_factory=time.monotonic)
    _tokens_used: int = 0
    _recent_actions: list[str] = field(default_factory=list)

    def charge_tokens(self, n: int) -> None:
        """累计 token 并在超预算时熔断 / accrue tokens, trip when over budget."""
        self._tokens_used += n
        if self._tokens_used > self.max_tokens:
            raise RuntimeError(f"token 预算超限 {self._tokens_used}/{self.max_tokens}")

    def check_deadline(self) -> None:
        """整体超时检查 / overall timeout check."""
        if time.monotonic() - self._start > self.deadline_s:
            raise TimeoutError(f"整体超时 {self.deadline_s}s")

    def detect_loop(self, action_signature: str, window: int = 3) -> None:
        """死循环检测：连续 window 次相同动作即判定打转 / loop detection."""
        # 时间 O(1) 空间 O(window)
        self._recent_actions.append(action_signature)
        if len(self._recent_actions) > window:
            self._recent_actions.pop(0)
        if (
            len(self._recent_actions) == window
            and len(set(self._recent_actions)) == 1
        ):
            raise RuntimeError(f"死循环：连续 {window} 次相同动作 {action_signature!r}")


def run_agent(task: str) -> str:
    """带四道防线的最小 agent 循环 / minimal agent loop with all four guardrails."""
    budget = RunBudget()
    for step in range(budget.max_steps):  # 步数上限本身就是死循环兜底
        budget.check_deadline()           # 防线 4
        action = "search"                 # 真实场景由模型决定 / model decides in reality
        budget.detect_loop(f"{action}:{task}")  # 防线 2

        observation = call_flaky_tool(task)      # 防线 1（内部自动重试）

        # 真实场景这里调 LLM；演示用固定 token 估算 / estimate tokens per step
        budget.charge_tokens(2_500)              # 防线 3
        print(f"[step {step}] {action} -> {observation}")

        if observation:                          # 满足终止条件就退出 / stop condition
            return f"完成：{observation}"
    return "达到步数上限，安全退出 / hit step cap, exited safely"


if __name__ == "__main__":
    try:
        print(run_agent("2026 年 AI agent 健壮性最佳实践"))
    except (RuntimeError, TimeoutError) as exc:
        # 健壮性：任何一道防线触发都给出明确原因，而非裸栈
        # any guardrail trip surfaces a clear reason, not a raw traceback
        print(f"agent 被安全中止 / aborted safely: {exc}")
```

运行多跑几次：你会看到工具偶发失败被静默重试掉；若把 `max_tokens` 调到 `5000`，会看到"token 预算超限"被干净地拦下。

### Agent 可靠性语义：重试一个已扣款的工具会怎样？（2026 补充，Java 后端强项）

防线 1 的重试有个隐含前提：**工具是无副作用的读操作**。一旦工具会扣款、发邮件、建工单，"超时后重试"就可能变成"扣两次款"——因为**超时 ≠ 失败**，请求可能已在对端执行成功。分布式系统的经典结论在此完全适用：网络之上没有免费的 exactly-once，只能用 **at-least-once 重试 + 幂等副作用** 拼出等价效果：

| 语义 | 含义 | 做法 |
|---|---|---|
| at-most-once | 宁可丢失，不重复 | 写操作不重试 |
| at-least-once | 宁可重复，不丢失 | 重试 + 对端幂等 |
| exactly-once（等价实现） | 恰好生效一次 | at-least-once + 幂等键去重 |

**幂等键（idempotency key）设计**：调用方为"同一个逻辑操作"生成稳定的 key，重试/重放时 key 不变；服务端按 key 去重，重复请求直接返回首次结果（Stripe 的 `Idempotency-Key` 请求头就是这个模式）。Agent 场景的关键是 key 必须由**确定性上下文**推导，不能用随机 UUID——否则重试时 key 变了，去重直接失效：

```python
# 幂等键 = (会话, 步骤, 工具, 规范化参数) 的稳定哈希
# idempotency key: stable hash of (thread_id, step, tool, canonical args)
key = hashlib.sha256(
    f"{thread_id}:{step_index}:{tool_name}:{json.dumps(args, sort_keys=True)}".encode()
).hexdigest()
executed: dict[str, str] = {}  # 演示用；生产换成带 TTL 的 Redis 或 DB 唯一索引

def execute_tool_once(key: str, tool, args: dict) -> str:
    """幂等执行：同 key 只真正执行一次 / side effect fires at most once per key."""
    # 时间 O(1) 空间 O(去重表大小)
    if key in executed:          # 命中 → 返回首次结果，不重复执行 / replay-safe
        return executed[key]
    executed[key] = tool(**args)  # 生产需"记录+执行"原子化，DB 唯一索引兜底
    return executed[key]
```

**与 Day 29 checkpointer 重放的联动**：断点续跑本质是"从 checkpoint 重放后续步骤"。LangGraph 恢复时已完成节点的结果直接从 checkpoint 读取、不会重跑，但**崩溃/中断时正在执行的那个节点会从头重跑**——该节点内的工具调用若有副作用，就必须靠幂等键保证**重放安全（replay-safe）**：重放可以发生任意次，外部动作只生效一次。上面 key 里放 `thread_id + step_index`，正是为了让"同一断点的重放"命中同一个 key。

**失败补偿 / saga 回滚**：有些副作用既无法幂等化也已经发生（款已扣、邮件已发），多步流程中途失败时只能**补偿**：为每个写操作定义逆操作（扣款↔退款、建单↔关单），失败后按逆序执行补偿——即分布式事务的 **Saga 模式**（Java 世界对应 Seata Saga / TCC）。注意补偿是"业务上的对冲"而非时光倒流：邮件发出去收不回，只能补发更正。因此真正不可逆的高危动作（真实扣款），在 agent 里通常还要叠加 Day 30 的 HITL 人工审批门控。

## 3. 今日任务

1. 跑通 `day36_robust_agent.py`，连跑 5 次，观察 `call_flaky_tool` 的瞬时失败被重试吸收。
2. **触发每一道防线**：分别把 `max_tokens=5000`、`deadline_s=0.001`、把 `action` 写死成常量后注释掉终止条件，确认四种中止理由都能被打印出来。
3. **加一个"重试耗尽"路径**：把 `call_flaky_tool` 的失败概率改成 `0.95`，确认 4 次重试耗尽后异常被最外层捕获，打印的是友好中文提示。

**验收标准**：四道防线各能被独立触发并打印明确原因；重试耗尽时不抛裸栈；正常路径下偶发工具失败对最终结果无影响。

## 4. 自测清单

- [ ] 我能说清重试 / 死循环 / token 预算 / 整体超时各自管哪一类故障，以及为什么它们正交。
- [ ] 我理解指数退避为什么要加抖动（jitter）。
- [ ] 我知道"最大步数"是死循环的最后一道兜底，即使语义检测失效也不会无限循环。
- [ ] 我能解释 token 预算和后端"线程池容量上限"的相似之处。
- [ ] 我的 agent 在任何一道防线触发时，对外都是一条可读的中止信息，而非崩溃。
- [ ] 我能回答"重试一个已扣款的工具会怎样"，并说清 exactly-once 为什么只能靠 at-least-once + 幂等键拼出等价效果。
- [ ] 我知道幂等键为什么必须由确定性上下文（thread_id + 步骤 + 参数）推导而非随机 UUID，以及它如何让 Day 29 的断点重放不重复执行副作用。

## 5. 延伸 & 关联

- 工具异常的基础处理（本仓库第 8 天打底）：[../07-llm-applications/05-langchain/02-agents-and-tools.md](../07-llm-applications/05-langchain/02-agents-and-tools.md)
- LangGraph 里实现循环与上限的官方做法：本课程 Day 28（循环）会把这里的 `max_steps` 对应到 graph 的 `recursion_limit`。
- 生产监控视角（为什么这些中止理由必须能被 trace 到）：[../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md](../08-llm-engineering/03-mlops/02-evaluation-and-monitoring.md)
- 上下文超长截断策略（token 预算的"软"对策）：本课程 Day 11（记忆与会话状态）。
- 可靠性语义 × 断点续跑：Day 29 的 checkpointer 提供"重放"能力，本日的幂等键保证"重放不重复执行副作用"——两者合起来才是可安全恢复的 agent。
