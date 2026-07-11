# 协议文档：Agent 内部数据结构与协作协议

本文档描述 MVP 中各模块之间的数据结构协议。代码实现位于：

```text
agent/protocols.py
```

---

## 1. 设计原则

### 1.1 Event first

所有外部输入和内部异步结果都统一为 `AgentEvent`。

```text
用户消息是 event。
工具进度是 event。
工具完成是 event。
定时器触发是 event。
取消和失败也是 event。
```

### 1.2 Generator only receives Context

Generator 不直接读取数据库或运行时内部对象。Runtime 通过 Workspace/ContextBuilder 构造 context。

```text
AgentState -> Workspace/ContextBuilder -> Context -> Generator
```

### 1.3 Generator outputs Commands, not side effects

Generator 输出结构化 command。Runtime 解释 command，并通过 TaskSystem/ActionSystem 执行。

### 1.4 Task is the activity unit

长任务、等待、取消、恢复、依赖推进都挂在 Task 上。

### 1.5 ActionRun is one concrete execution

Task 是语义活动；ActionRun 是一次具体能力执行。

---

## 2. AgentEvent

### 2.1 Schema

```ts
type AgentEvent = {
  event_id: string
  agent_id: string
  type: string
  source: string
  payload: object

  task_id?: string
  action_run_id?: string

  correlation_id?: string
  causation_id?: string
  idempotency_key?: string

  priority: number
  created_at: string
}
```

### 2.2 字段说明

| 字段 | 含义 |
|---|---|
| `event_id` | 事件唯一 ID，用于幂等处理 |
| `agent_id` | 目标 agent |
| `type` | 事件类型 |
| `source` | 事件来源，如 user、action_executor、local_tool_worker |
| `payload` | 事件载荷 |
| `task_id` | 事件关联的 Task |
| `action_run_id` | 事件关联的 ActionRun |
| `correlation_id` | 跨多个事件的关联 ID |
| `causation_id` | 当前事件由哪个事件引发 |
| `idempotency_key` | 幂等键，生产版本应强化 |
| `priority` | 事件优先级 |
| `created_at` | 创建时间 |

### 2.3 MVP 事件类型

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

### 2.4 示例

```json
{
  "event_id": "evt_abc",
  "agent_id": "agent_demo",
  "type": "action.completed",
  "source": "local_tool_worker",
  "task_id": "task_123",
  "action_run_id": "run_456",
  "payload": {
    "action_name": "project_analysis",
    "result": {
      "summary": "分析完成"
    }
  },
  "causation_id": "evt_started",
  "priority": 100,
  "created_at": "2026-07-02T00:00:00+00:00"
}
```

---

## 3. AgentState

### 3.1 Schema

```ts
type AgentState = {
  agent_id: string
  profile: AgentProfile
  workspace: Workspace
  tasks: Record<string, AgentTask>
  action_runs: Record<string, ActionRun>
  processed_event_ids: string[]
  version: number
  created_at: string
  updated_at: string
}
```

### 3.2 说明

`AgentState` 是持久化 actor 状态。MVP 中每次事件处理后写入 JSON。

生产版本里可以换成数据库，但协议不应该依赖具体存储。

---

## 4. AgentProfile

### 4.1 Schema

```ts
type AgentProfile = {
  agent_id: string
  name: string
  system_profile: string
  identity_profile: string
  background_profile: string
  persona_profile: string
  values_profile: string
  behavior_profile: string
  voice_profile: string
  speech_profile: string
  relationship_profile: string
  self_boundaries: string
}
```

### 4.2 说明

Profile 是稳定 Self Model，不应该在每次事件中频繁变化。它用于约束 Generator 的身份、判断、关系和表达风格。

- `identity_profile` 与 `background_profile` 定义“我是谁、为何形成现在的取向”；其中背景是规范性自我叙事，不自动构成外部世界证据。
- `persona_profile`、`values_profile`、`behavior_profile` 与 `self_boundaries` 约束判断和行动。
- `voice_profile` 定义整体声色，`speech_profile` 定义措辞和节奏，主要由 BrocaSystem 用于最终表达。
- `relationship_profile` 规定默认关系姿态；具体关系变化来自带来源的对话和记忆，不应反向随意改写稳定 Profile。

持久化旧状态缺少新增字段时按空字符串兼容。Runtime 每次处理事件时会用当前配置中的非空值刷新稳定 Profile，因此角色档案升级会作用于旧状态；任务、记忆、情绪和关系等经历状态不会被清空。

---

## 5. Workspace

### 5.1 Schema

```ts
type Workspace = {
  workspace_id: string
  current_task_id?: string
  notes: string[]
  variables: Record<string, any>
  transcript: Array<{
    role: "user" | "assistant" | "system"
    content: string
    event_id?: string
    created_at: string
  }>
  last_decision_summary: string
  updated_at: string
}
```

### 5.2 说明

Workspace 是 working memory，负责承载：

- 当前 task focus
- 完整对话原始记录
- 工作 notes
- 临时变量
- 上一次 decision summary

ContextBuilder 会从 Workspace、Task、ActionSpec 中构造 Generator context。

Workspace/AgentState 是完整数据源，Generator context 是预算内视图。每个候选单元具有稳定引用，未进入本轮模型请求的数据仍可通过 ActionSystem 的只读回读工具访问。`context_selection` 公开预算、选中数量、可回读数量和摘要版本；完整 selection manifest 写入 generator trace，不进入模型正文。

模型请求不拆分 `assistant(tool_call) + tool(result)` 消息对。messages 和 tools 在 ModelInterface 调用前统一估算，超过 session 配置的输入预算时显式报错。

### 5.3 Workspace 与 State 的关系

MVP 中 `AgentState` 是持久化总状态，`Workspace` 是其中的 working memory 层。

可以这样理解：

```text
AgentState = agent actor 的完整可恢复状态
Workspace  = generator 当前工作记忆和 context builder 输入区
```

---

## 6. AgentTask

### 6.1 Schema

```ts
type AgentTask = {
  task_id: string
  agent_id: string

  title: string
  goal: string
  purpose: string

  status:
    | "created"
    | "runnable"
    | "running"
    | "waiting"
    | "blocked"
    | "completed"
    | "failed"
    | "cancelled"

  parent_task_id?: string
  child_task_ids: string[]
  dependencies: string[]

  active_action_runs: string[]
  waiting_on: AwaitCondition[]
  scheduling: {
    root_task_id: string
    depth: number
    classification: string
    reason: string
    can_run: boolean
    can_complete: boolean
    pending_dependency_ids: string[]
    failed_dependency_ids: string[]
    nonterminal_child_ids: string[]
    completion_blockers: object[]
  }

  progress: object
  result_ref?: string
  result?: object
  error?: object

  workspace_ref?: string
  continuation: object

  created_at: string
  updated_at: string
  version: number
}
```

### 6.2 状态机

MVP 使用较简单状态机：

```text
created -> runnable -> running -> waiting -> runnable -> completed
                                \-> failed
                                \-> cancelled
```

含义：

| 状态 | 含义 |
|---|---|
| `created` | 刚创建，还未进入调度 |
| `runnable` | 可被 Generator/Runtime 推进 |
| `running` | 正在执行同步 action 或刚开始执行 |
| `waiting` | 等待 action、用户、timer 或外部事件 |
| `blocked` | 依赖未满足或权限不足 |
| `completed` | 完成 |
| `failed` | 失败 |
| `cancelled` | 取消 |

### 6.3 continuation

`continuation` 用于恢复语义。

示例：

```json
{
  "expected_next_step": "当 project_analysis 完成后，根据结果生成报告并完成任务。",
  "resume_context_summary": "用户希望分析项目，任务可能较耗时，所以应异步执行。"
}
```

它回答三个问题：

```text
这个任务为什么存在？
这个异步结果回来后应该做什么？
这个结果服务哪个用户目标？
```

---

## 7. ActionSpec

### 7.1 Schema

```ts
type ActionSpec = {
  name: string
  description: string
  input_schema: object
  mode: "sync" | "async" | "stream" | "subscription"
  timeout_ms: number
  cancelable: boolean
  requires_approval: boolean
  side_effect_level: "none" | "read" | "write" | "external_effect"
  source: "local" | "star_protocol" | string
  target?: string
  metadata: object
}
```

### 7.2 MVP 支持

MVP 只实现：

```text
sync
async
```

`source="local"` 表示由本地 `ActionExecutor` 执行；`source="star_protocol"` 表示由 `StarSession` 下发到 Star Protocol 环境。`target` 通常是 Environment ID。

预留但未实现：

```text
stream
subscription
```

---

## 8. ActionRun

### 8.1 Schema

```ts
type ActionRun = {
  action_run_id: string
  agent_id: string
  task_id: string
  action_name: string
  args: object
  mode: "sync" | "async"

  status: "created" | "running" | "succeeded" | "failed" | "cancelled"

  progress: object
  result?: object
  error?: object

  created_at: string
  started_at?: string
  finished_at?: string
  idempotency_key?: string
}
```

### 8.2 Task 与 ActionRun 的关系

```text
Task = 有目标和语义的活动单元
ActionRun = Task 下的一次具体能力执行
```

一个 Task 可以包含多个 ActionRun。

当前实现使用 `task_id + action_name + canonical JSON(args)` 的 SHA-256
摘要生成稳定 `idempotency_key`。Runtime 重启后，同一 Task 中已经成功的副作用
Action 会直接回放已持久化结果，不会再次下发；失败或取消的 Action 允许重试。
读取类 Action 不回放成功结果，以免返回已经过期的环境状态。

---

## 9. GeneratorDecision

### 9.1 Schema

```ts
type GeneratorDecision = {
  decision_summary: string
  commands: Command[]
}
```

### 9.2 示例

```json
{
  "decision_summary": "User requests long-running analysis; create a Task and start async action.",
  "commands": [
    {
      "type": "create_task",
      "task_ref": "main",
      "title": "分析大型项目并生成报告",
      "goal": "分析用户指定的大型项目，完成后给出结构化报告。",
      "purpose": "展示异步任务、等待、恢复与用户中断能力。"
    },
    {
      "type": "start_action",
      "task_ref": "main",
      "action_name": "project_analysis",
      "args": {
        "target": "demo-large-project"
      },
      "mode_hint": "async"
    },
    {
      "type": "reply",
      "content": "我已经创建了一个后台分析任务。"
    },
    {
      "type": "wait",
      "task_ref": "main",
      "condition": {
        "kind": "action_completed",
        "action_name": "project_analysis"
      }
    }
  ]
}
```

---

## 10. Command 协议

### 10.1 reply

```ts
type ReplyCommand = {
  type: "reply"
  content: string
}
```

向用户发送消息，同时写入 Workspace transcript。

---

### 10.2 create_task

```ts
type CreateTaskCommand = {
  type: "create_task"
  task_ref?: string
  title: string
  goal: string
  purpose: string
  parent_task_id?: string
  dependencies?: string[]
  continuation?: object
}
```

`task_ref` 是同一个 decision 内的临时引用，方便后续 command 引用刚创建的 task。

---

### 10.3 start_action

```ts
type StartActionCommand = {
  type: "start_action"
  task_id?: string
  task_ref?: string
  action_name: string
  args: object
  mode_hint?: "sync" | "async"
}
```

Runtime 会创建 ActionRun，并交给 ActionExecutor 执行。

---

### 10.4 wait

```ts
type WaitCommand = {
  type: "wait"
  task_id?: string
  task_ref?: string
  condition: AwaitCondition
}
```

等待不是阻塞，而是把条件写入 Task。

---

### 10.5 update_task

```ts
type UpdateTaskCommand = {
  type: "update_task"
  task_id?: string
  task_ref?: string
  patch: object
}
```

MVP 限制可更新字段，避免 Generator 任意改内部结构。

---

### 10.6 complete_task

```ts
type CompleteTaskCommand = {
  type: "complete_task"
  task_id?: string
  task_ref?: string
  result?: object
}
```

这是一个受 TaskSystem 保护的完成请求，不是直接赋值。子树仍有非终态 task、active ActionRun、未满足等待、未完成依赖或未确认的 failed/cancelled 子任务时，返回 `deferred=true` 和结构化 `blockers`，task 保持非完成状态。

若替代路径已经让父目标成功，可以在 `result.accepted_terminal_task_ids` 中明确确认已处理的 failed/cancelled 子任务，再次请求完成。

---

### 10.7 cancel_task

```ts
type CancelTaskCommand = {
  type: "cancel_task"
  task_id?: string
  task_ref?: string
  reason: string
}
```

Runtime 会取消该 Task 子树中的 active ActionRun，并把仍非终态的 Task 子树标记为 cancelled。

---

## 11. AwaitCondition

MVP 使用开放对象：

```ts
type AwaitCondition = {
  kind: string
  action_name?: string
  action_run_id?: string
  event_type?: string
  timeout_at?: string
  metadata?: object
}
```

当前主要使用：

```json
{
  "kind": "action_completed",
  "action_name": "project_analysis",
  "action_run_id": "run_123"
}
```

未来可扩展：

```text
user_reply
external_event
timer
all/any condition
approval_result
subscription_match
```

---

## 12. RuntimePolicy 协议

`RuntimePolicy` 决定事件是否激活 Generator。

当前规则：

| Event type | 是否激活 Generator |
|---|---|
| `user.message` | 是 |
| `action.completed` | 是 |
| `action.failed` | 是 |
| `action.cancelled` | 取决于 `payload.silent` |
| `timer.fired` | 是 |
| `action.started` | 否 |
| `action.progress` | 否 |

这个策略是 MVP 成本控制的核心。

---

## 13. Checkpoint 协议

每次事件处理完成后写入 checkpoint：

```ts
type CheckpointRecord = {
  created_at: string
  agent_id: string
  state_version: number
  event: AgentEvent
  decision?: GeneratorDecision
  comment: string
}
```

MVP 存储为 JSONL：

```text
checkpoints/{agent_id}.jsonl
```

---

## 14. 协议不变量

### 14.1 Generator 不直接产生副作用

只输出 command。

### 14.2 ActionExecutor 是副作用边界

所有外部工具和长任务必须通过 ActionExecutor。

### 14.3 TaskSystem 负责 Task 状态机

Generator 不直接写底层 Task 对象。

TaskSystem 在每次相关事件和任务变更后执行 reconcile：校验父子关系与依赖，派生 `scheduling`，优先选择可运行的深层任务，并阻止父/root task 提前完成。依赖未完成时 task 为 waiting，依赖失败、缺失或成环时 task 为 blocked；如果子任务全部 blocked，父任务恢复为 runnable 以便重新规划。

### 14.4 Event 是恢复入口

异步 action 完成后，不调用某个 suspended function，而是发布 `action.completed` event。

### 14.5 Progress 默认不激活模型

防止异步工具高频事件导致模型调用风暴。

### 14.6 事件需要幂等

MVP 使用 `processed_event_ids`。生产版本应使用数据库唯一键和 outbox pattern。

### 14.7 完整存储与模型上下文分离

持久化 Store 不因 context budget 删除原始记录。默认 generator context window 为 1,000,000 tokens，900,000 tokens 是紧急压缩触发线，压缩目标为 300,000 tokens。ConversationSystem 使用分层历史：最近 6 轮保持原始 user/assistant 消息，之前 12 轮保留 Wernicke 语义理解和 Agent 表达意图，更早的活动历史合并为带 turn_id 的摘要。ContextBuilder 对 workspace 也按目标预算选择。未进入本轮请求的内容保留 Store 引用，可通过搜索和精确读取能力恢复。滚动摘要是工作记忆，不替代原始事实，精确参数、工具结果和 task 状态必须按 id 回读。

### 14.8 对外话语必须经过 ConversationSystem

`user.message` 不直接激活 DecisionSystem。Wernicke 先提交带 speaker attribution 的 understanding；是否咨询 DecisionSystem 由 Wernicke 的工具选择决定。Decision 的普通文本是 speech intent，不是外部消息。只有 Broca 生成的 `conversation.utterance.ready` 可以由 ProtocolInterface 发送。

核心事件：

```text
conversation.understanding.requested
conversation.understanding.ready
conversation.decision.requested
conversation.speech.requested
conversation.utterance.ready
conversation.utterance.sent
```

所有事件使用 `conversation_id`、`turn_id`、`correlation_id` 和 `causation_id` 保持因果链。普通 turn 默认只允许一条正式 outbound utterance，显式 `progress_response` 才能形成多条阶段性话语；被更新 turn 或既有正式回复挡住的 speech intent 会写入 `suppressed_speech_intents` 供审计。最新已发送话语同时写入 `response_text` 便于快速读取。

---

## 15. 未来协议扩展

优先扩展方向：

```text
schedule_timer command
subscribe_event command
request_approval command
spawn_child_task command
stream action events
external perception events
budget and permission policy
outbox/inbox persistent queue
```
