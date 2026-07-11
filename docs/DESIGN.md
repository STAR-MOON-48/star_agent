# 整体设计文档：Event-driven Task-centric Agent Runtime MVP

## 1. 背景

经典 tool loop 通常是：

```text
User -> Assistant -> Tool -> Assistant -> Tool -> Assistant(final)
```

这个结构适合短链路、同步、低延迟的工具调用，但不适合一个真正存在于时间中的自主交互 agent。原因是：

- 工具执行可能很慢，不能一直阻塞模型和请求线程。
- 外部事件可能随时到来，比如用户补充、取消、webhook、定时器、工具进度。
- agent 需要等待，但等待不应该等于同步阻塞。
- 长任务完成后，agent 需要恢复当时的语义，而不是只拿到一个裸工具结果。
- 用户交互不是一次 final answer，而是可以持续中断、查询、修改目标。

因此，这个 MVP 把架构从 `model-driven tool loop` 改成：

```text
Event -> AgentRuntime -> GeneratorDecision -> Commands -> Runtime effects
```

也就是：Runtime 是主循环；Generator 只是被事件激活的认知决策器。

---

## 2. 目标

这个 MVP 的目标不是一次性做完整 Agent OS，而是验证最小地基：

```text
Generator + Action + Task + EventBus + Workspace + Persistence
```

其中：

- **Generator**：根据 context 生成结构化 decision。
- **Workspace**：作为 working memory，同时为 Generator 构造 context。
- **ActionSystem**：描述和执行能力，负责同步/异步 action。
- **TaskSystem**：管理 agent 的活动单元，类似操作系统 PCB。
- **EventBus**：统一承载用户输入、工具结果、进度、取消、定时器等事件。
- **Persistence**：保存状态和 checkpoint，让 agent 可以跨时间恢复。

---

## 3. 总体结构

```text
                  ┌────────────────────┐
User / Tool / Timer│      EventBus       │
External Events ─▶│   asyncio.Queue MVP │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │    AgentRuntime    │
                  │ load state/checkpt │
                  └─────────┬──────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
   ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
   │  TaskSystem    │ │  Workspace   │ │RuntimePolicy │
   │ apply event    │ │build context │ │wake or not   │
   └────────────────┘ └──────┬───────┘ └──────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │     Generator      │
                  │ real model adapter │
                  └─────────┬──────────┘
                            │ decision
                            ▼
                  ┌────────────────────┐
                  │  Command Executor  │
                  └──────┬───────┬─────┘
                         │       │
                         ▼       ▼
                ┌────────────┐ ┌────────────┐
                │TaskSystem  │ │ActionSystem│
                │state patch │ │sync/async  │
                └────────────┘ └─────┬──────┘
                                     │
                                     ▼
                              future events
```

核心变化是：

```text
传统 tool loop：
  LLM 调用工具，并等待工具返回，然后继续调用 LLM。

这个 MVP：
  Runtime 接收事件，必要时唤醒 Generator。
  Generator 输出 command。
  Runtime 执行 command。
  长 action 未来通过 EventBus 返回结果。
```

---

## 4. Agent = Profile + Runtime

### Profile

Profile 是稳定身份和行为边界：

```text
SystemProfile   agent 的系统级身份
PersonaProfile  agent 的人格风格
BehaviorProfile agent 的行为约束和默认策略
```

MVP 中对应 `AgentProfile`。

### Runtime

Runtime 是 agent 的运行系统：

```text
EventBus
RuntimePolicy
Interfaces
GeneratorRuntime
Workspace / ContextBuilder
PerceptionSystem
Generator
TaskSystem
ActionSystem
Persistence
```

Runtime 持有 actor 的执行权，负责事件顺序、锁、状态读写、checkpoint、命令执行。

Interfaces 是 runtime 与外界交互的边界：

- `ModelInterface`：模型访问边界，当前实现 `StarModel`，底层使用 MengLong `Model`，默认模型是 `xiaomi/mimo-v2.5`。
- `ProtocolInterface`：通信协议边界，当前实现 `StarSession`，底层使用 Star Protocol `AgentClient`。

PerceptionSystem 是 runtime 的整体感知入口。它把本地输入、Star Protocol action/event/outcome/stream 归一化为内部 `AgentEvent`，再通过 `EventBus` 分发给 Runtime 和其他系统。

`GeneratorRuntime` 是 kernel 中的模型/生成管理器。它管理一个位于 kernel 的 `GeneratorSession` actor，把 runtime context 转成 MengLong `Context`，再把请求投递给 session。其他子系统以后需要模型能力时，也应该通过 `GeneratorRuntime` 发起，而不是直接持有模型 adapter。

---

## 5. Workspace 的角色

我们在讨论中进一步明确：MVP 不能缺少 Workspace。

Workspace 不是普通 memory，也不是全部 state 的别名。它更像 agent 的 working memory：

```text
Workspace = 当前工作记忆 + context builder 输入材料 + 当前任务焦点
```

MVP 中 Workspace 负责：

- 持久保存 transcript 原始记录。
- 保存当前 task focus。
- 保存 notes 和 variables。
- 保存 last_decision_summary。
- 为 ContextBuilder 提供上下文材料。

Generator 不直接接收完整数据库或所有内部对象，而是只接收 Workspace/ContextBuilder 生成的 decision pack。完整 action schema 不放在 context JSON 中，而是通过模型请求的 `tools` 字段传入。

完整持久化不等于每轮全量发送。ContextBuilder 将 task、ActionRun、transcript、note 和 ActionSpec 转成带引用、优先级和 token 估算的候选单元，在每个 GeneratorSession 独立的 context policy 下组装请求。当前事件、focus task、active/trigger action 和滚动摘要是高优先级信息；完整 tool-call/result 始终成对选择。`runtime.execution_memory` 额外保存最近动作的紧凑结果与相同 action/args 的重复尝试计数，避免完整历史未入选时重新走已失败路径。

没有进入本轮请求的数据仍保留在 Store。Generator 可通过 `search_workspace`、`read_task`、`read_action_run`、`search_actions` 精确回读。小型外部 action catalog 在 tool budget 内整体发送，避免关键环境能力因动态排序而消失；大型目录才执行预算选择。旧阶段材料超过阈值且摘要源发生变化时，Runtime 才调用 ContextBuilderSystem 生成滚动 Markdown 摘要，并受冷却时间限制。

这意味着：

```text
Workspace 是 Generator 的感知边界之一。
ContextBuilder 是 Workspace 到 Generator 的协议转换层；模型请求工具 schema 是 GeneratorRuntime 到 ModelInterface 的独立通道。
```

---

## 6. Generator 的角色

Generator 是有状态 actor 的认知生成部分，但它不直接产生副作用。

它的输入：

```text
decision_pack = trigger event + focus task/action runs + selected evidence + tool names
model_request.tools = selected action schemas
context_selection = budget + selected refs + available-by-reference counts + summary version
```

它的输出有三种路径：

```text
1. model_request.tools/tool_calls -> Runtime 封装为内部 start_action
2. Decision 普通自然语言文本 -> ConversationSystem 封装为 speech intent
3. GeneratorDecision JSON -> 仅用于内部任务控制
```

Decision 生成的 command 可以包含：

- `reply`
- `create_task`
- `start_action`
- `wait`
- `update_task`
- `complete_task`
- `cancel_task`

Generator 不能直接修改外部世界。能力调用优先通过模型请求 `tools` 字段产生 tool_call，再由 Runtime 封装为 task/action。

用户话语不再直接进入 DecisionSystem。对话事件链为：

```text
user.message
→ conversation.understanding.requested
→ WernickeSystem（可调用内部检索工具）
→ conversation.understanding.ready
→ conversation.decision.requested（由 Wernicke 按需选择）
→ conversation.speech.requested
→ BrocaSystem
→ conversation.utterance.ready
→ ProtocolInterface
→ conversation.utterance.sent
```

Wernicke understanding 是带 speaker 来源、置信度和待验证状态的解释，不直接写成世界事实。ConversationManager 只保证 turn 顺序、状态和关联关系，不决定理解内容。普通 turn 默认只发送一次正式回复；确实需要阶段性通知时必须显式使用 `progress_response`。较新的 Human turn 尚未回复时，旧 turn 已排队但未发送的话语会被记录为 suppressed，避免重启恢复或并发 action outcome 造成连续重复发言。

这个边界很重要：

```text
Generator decides what should happen.
Runtime decides how to execute it safely.
```

---

## 7. ActionSystem 的角色

ActionSystem 不是传统意义上的单个 tool call，而是 agent 的能力系统。

MVP 中包含：

```text
ActionRegistry   提供 ActionSpec
ActionExecutor   执行 sync/async action
ActionRun        记录一次具体 action 执行
```

ActionSpec 描述：

- action 名称
- 输入 schema
- 执行模式：sync / async
- timeout
- cancelable
- side effect level
- 是否需要审批

MVP 先实现两个 action：

```text
project_analysis     async 本地长任务
query_task_status    sync 本地短任务
```

---

## 8. TaskSystem 的角色

Task 是 agent 活动的基本抽象，可以类比为操作系统中的 PCB。

Task 不是简单 todo，而是一个可调度、可等待、可恢复、可取消的活动单元。

MVP 中 Task 包含：

- goal
- purpose
- status
- active_action_runs
- waiting_on
- parent_task_id / child_task_ids
- dependencies
- scheduling（TaskSystem 派生的可运行、等待、阻塞和可完成判断）
- progress
- result
- error
- continuation

其中 `continuation` 很关键：它保存未来恢复时的语义，例如：

```text
这个任务为什么存在？
这个 action 完成后 agent 应该做什么？
这个结果服务哪个用户目标？
```

没有 continuation，异步结果回来时就只是一个裸 result，Generator 很容易丢失任务语义。

TaskSystem 对任务树执行统一 reconcile。叶子任务在依赖满足时进入 runnable；父任务在子任务推进期间进入 waiting；当子任务全部终态后，父任务重新进入 runnable 做结果归纳和完成确认。`complete_task` 只有在整个子树没有非终态任务、active action、等待或未完成依赖时才会成功，因此 root task 不会因一次模型误判被提前关闭。

---

## 9. EventBus 的角色

EventBus 是 agent 的时间和中断系统。

所有输入统一为 event：

```text
user.message

action.started
action.progress
action.completed
action.failed
action.cancelled

timer.fired
system.wakeup
```

关键原则：

```text
外部数据不是插入 tool loop，而是进入 agent mailbox/event bus。
```

Runtime 读取事件后先更新状态，再由 RuntimePolicy 判断是否需要唤醒 Generator。

---

## 10. RuntimePolicy：避免模型调用风暴

不是所有事件都应该唤醒模型。

MVP 策略：

```text
user.message       -> activate Generator
action.completed   -> activate Generator
action.failed      -> activate Generator
action.cancelled   -> activate unless silent
timer.fired        -> activate Generator
action.started     -> state-only
action.progress    -> state-only
```

这能避免长任务的高频 progress 事件不断调用模型。

---

## 11. 同步与异步 action

### 同步 action

适合短时间、低成本、结果是后续推理直接依赖的动作。

流程：

```text
Generator -> start_action(sync)
Runtime -> ActionExecutor executes immediately
ActionExecutor -> action.completed event
EventBus -> Runtime -> Generator resumes
```

### 异步 action

适合耗时任务。

流程：

```text
Generator -> create_task
Generator -> start_action(async)
Generator -> reply
Generator -> wait(action_completed)
Runtime persists state and releases control

Async worker -> action.progress events
Runtime updates task state without waking Generator

Async worker -> action.completed
Runtime activates Generator
Generator replies with final report and completes task
```

---

## 12. MVP 闭环

最小闭环是：

```text
1. EventBus 收到 user.message。
2. Runtime 加载 AgentState。
3. TaskSystem 把事件应用到状态。
4. RuntimePolicy 决定唤醒 Generator。
5. Workspace/ContextBuilder 构造 context。
6. GeneratorRuntime 把 runtime context 转成 MengLong Context，投递给 GeneratorSession actor。
7. LLMGeneratorSession 输出 tool_calls、自然语言文本或内部 GeneratorDecision。
8. Runtime 将其封装为内部 commands 并执行。
9. ActionSystem 启动 sync/async action。
10. TaskSystem 更新 Task 状态。
11. Store 保存 state 和 checkpoint。
12. Action worker 未来发布 progress/completed event。
13. Runtime 重新被事件唤醒。
```

---

## 13. 可靠性原则

MVP 中已经预留或部分实现这些原则：

### 1. 副作用必须通过 ActionExecutor

Generator 不直接执行工具。

### 2. 事件需要幂等

`processed_event_ids` 用来跳过重复事件。

### 3. 状态需要 checkpoint

每次事件处理后写入：

```text
state.json
checkpoints/{agent_id}.jsonl
```

### 4. 同一个 agent 串行处理事件

`AgentRuntime` 使用 actor lock，避免同一个 agent 的并发状态写冲突。

### 5. Progress 不唤醒模型

RuntimePolicy 把 progress 当成 state-only event。

---

## 14. 当前实现和生产版本的差距

当前 MVP 是教学/验证版本：

```text
EventBus          asyncio.Queue
Persistence       JSON file
Generator         GeneratorRuntime + LLMGeneratorSession + StarModel
Action workers    asyncio task
Lock              in-process asyncio.Lock
```

生产版本可替换为：

```text
EventBus          Kafka / NATS / Redis Streams / SQS / Postgres queue
Persistence       Postgres / FoundationDB / DynamoDB / Redis + snapshot
Generator         real LLM adapter
Action workers    distributed workers / Temporal / Celery / custom executor
Lock              DB advisory lock / distributed lock / actor placement
```

协议层不需要大改。

---

## 15. 最小设计结论

这个 MVP 的核心不是让工具循环更复杂，而是改变 agent 的运行范式：

```text
Agent does not run continuously.
Agent is activated by events.
Generator decides briefly.
Runtime executes safely.
Workspace preserves working memory.
Task represents long-running activity.
Action represents capability execution.
Future events resume the agent.
```

一句话概括：

```text
Generator 是认知执行器，Workspace 是工作记忆，Action 是能力边界，Task 是活动 PCB，EventBus 是时间和中断系统。
```
