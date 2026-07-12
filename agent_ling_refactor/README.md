# Agent Ling Refactor

这是与原实现并存的重构版本。它不改变 Agent Ling 的功能边界，重点重做认知区域之间的交接、Prompt 组织和运行时编排。

## 核心原则

普通交流仍保留两段认知过程（代码中的通用职责名为 Understanding / Expression，对应 Wernicke / Broca 区域）：

```text
用户话语
  → Understanding（Wernicke）：形成自然语言理解
  → Expression（Broca）：生成唯一对外话语
```

当理解区域发现行动、承诺或重要判断时，它会额外发出 `request_decision` 能力信号：

```text
                         ┌→ Expression → 尽快对外回复
Understanding ──────────┤
                         └→ Decision → Task / Action / Event
```

Expression 与 Decision 在理解完成后并发。Decision 的工具调用仍是结构化能力边界，但区域之间传递的内容都是 `NaturalMessage.text`，即自然语言。

事件总线和模型激活是两个独立层次：事件可以高频到达并持续更新状态，但只有携带新证据、解除等待或开启新目标的事件才能获得一次模型激活。内部任务更新、重复结果、自身广播和确定性的调度拒绝只记录，不请求模型。

## 与原版相比

| 方面 | 原版 | 重构版 |
|---|---|---|
| 普通交流 | Wernicke → Broca | 保留 Wernicke → Broca |
| 行动型交流 | 理解、决策、表达按事件串行 | 理解后，表达与决策并发 |
| 内部认知交接 | Understanding 对象、GeneratorDecision、命令 JSON、自然语言并存 | 统一为自然语言 `NaturalMessage` |
| Prompt | 包含大量 Runtime、任务树、分页、上下文协议 | 每个区域只描述自身职责和共同边界 |
| 模型唤醒 | 多类事件直接唤醒 Decision | 价值门槛、重复指纹、等待抑制和单链路预算 |
| 对外表达 | Broca 唯一生成 | 保持不变 |
| 工具执行 | ActionSystem / TaskSystem | 复用已验证实现 |
| 状态与记忆 | JSON 状态、对话、Markdown 记忆 | 保持兼容 |
| Star Protocol | StarSession | 保持兼容 |

## 目录

```text
agent_ling_refactor/
  messages.py           # 自然语言区域消息
  activation.py         # 模型激活价值门槛、去重和退避
  scheduling.py         # 等待条件修复和用户响应解锁
  settings.py           # 精简配置
  prompts.py            # 小型、按职责生成的 Prompt
  context.py            # 按区域裁剪的自然语言上下文
  model_gateway.py      # 统一模型与工具边界
  conversation.py       # 对话记录、主动续话和区域交接
  runtime.py            # 事件编排、区域并发、行动闭环
  app.py                # 应用装配
  config/default_agent.toml
  entrypoints/
  tests/
```

任务树、行动幂等、持久化、记忆、情绪和 Star 适配器是原项目中已经通过测试的稳定能力，本版本通过组合复用。这样避免复制数千行基础设施，也让重构风险集中在真正需要改变的 Prompt 和编排层。

## 运行

直接控制台：

```bash
uv run agent-ling-refactor
```

接入 Star Protocol：

```bash
uv run agent-ling-refactor-star \
  --agent-id agent_ling_refactor \
  --hub-url ws://localhost:8000 \
  --env-id demo_env
```

现有 `web_ui` 无需复制。它通过 Star Hub 与 Agent 通信，启动重构版 Star Agent 后仍可使用原 Web 场景。

```bash
uv run agent-ling-refactor-web \
  --hub-url ws://localhost:8000 \
  --env-id demo_env \
  --agent-id agent_ling_refactor
```

也可以直接使用 Python：

```python
from agent.runtime.persistence_system import JsonStateStore
from agent_ling_refactor import create_refactored_runtime

application = create_refactored_runtime(
    agent_id="ling",
    store=JsonStateStore(".agent_state_refactor"),
)
runtime = application.runtime
```

## 测试

```bash
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s agent_ling_refactor/tests -v
```

重构测试覆盖：

- 普通回复严格经过 Understanding → Expression 两轮；
- Understanding 触发 Decision 时，Expression 与 Decision 并发；
- 同步 Action 从用户请求、任务创建、事件恢复到 Broca 最终回复完整跑通；
- Expression 上下文不混入任务、行动和记忆协议；
- 所有认知 Prompt 保持小型且不包含 GeneratorDecision、命令 JSON 等运行时协议。
- 内部任务更新、不可运行失败、自身广播、重复结果不会请求模型；
- 新用户消息能确定性解除 `human_response` 等待，空等待条件不会持久化；
- 每条目标链有模型激活预算，429 后进入持久化指数退避；
- 启动目标直接进入 Decision，不再向虚假的 `startup` recipient 回复。

## 配置与 Prompt

默认配置位于 `config/default_agent.toml`。Prompt 分为五个通用职责：

- `understanding`：理解话语，并在必要时请求决策；
- `expression`：把自然语言理解或表达意图变成对外话语；
- `decision`：推进事件、任务和行动；
- `reflection`：空闲回顾；
- `memory`：经验归纳。

Profile 仍保留 Ling 的身份、价值观、表达方式和边界，但 Prompt 编译器只给当前区域必要的字段。Runtime 细节由代码保证，不再让模型背诵。

`[activation]` 提供模型唤醒策略：

- `duplicate_ttl_seconds`：等价证据在窗口内只评估一次；
- `max_decision_hops`：单个用户输入或自主目标最多派生的 Decision 跳数；
- `backoff_initial_seconds` / `backoff_max_seconds`：限流和临时错误的指数退避范围。

## 兼容与回退

- 原 `agent/`、`agent_ling/` 和 `web_ui/` 未被替换；
- 新状态默认写入 `.agent_state_refactor`，不会覆盖旧状态；
- 新命令使用 `agent-ling-refactor*` 前缀；
- 如果需要回退，继续运行原 `agent-ling-star` 或 `agent-ling-console` 即可。
