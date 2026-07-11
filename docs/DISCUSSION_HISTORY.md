# 思路文档：从传统 Tool Loop 到事件驱动 Agent Runtime

本文档记录我们这轮讨论中形成的设计脉络。它不是隐藏推理过程，而是对显式讨论内容的整理，方便后续继续迭代架构。

---

## 1. 起点：传统 tool loop 的局限

我们最开始讨论的是经典 agent tool loop：

```text
user -> assistant -> tool -> assistant -> tool -> assistant(final)
```

这个模式的优点是简单、容易实现、适合短任务。但它的问题也很明显：

1. **严格串行**：必须等待工具返回才能继续。
2. **同步阻塞**：长工具会卡住整个 loop。
3. **缺少时间维度**：等待、定时器、外部事件很难自然进入。
4. **外部中断困难**：用户中途补充、取消、修改目标时，传统 loop 不好处理。
5. **无法自然支持自主交互**：真实 agent 不是每次都由 user message 开始，也不是每次都在一次调用里完成。

我们因此把问题定义为：

```text
需要一个存在于时间中的自主可交互 agent，而不是一次性 inference chain。
```

---

## 2. 第一次架构转向：Runtime 驱动，而不是模型驱动

随后我们形成一个核心判断：

```text
LLM 不应该是主循环。
Agent Runtime 才应该是主循环。
```

传统 tool loop 是：

```text
model-driven loop
```

更适合自主 agent 的模式是：

```text
runtime-driven reactive agent
```

也就是：

```text
Event -> Runtime -> LLM/Generator decision -> Commands -> Runtime
```

模型只在关键事件到来时短暂运行，Runtime 长期存在并管理状态、事件、任务、工具和恢复。

---

## 3. 第二次抽象：Agent 是带状态 actor

我们把 agent 看成一个长期存在的 actor：

```text
Agent = state + mailbox + runtime + decision capability
```

它不再是一次函数调用，而是一个有收件箱、状态和活动任务的实体。

这个 actor 可以被不同事件激活：

```text
user.message
tool result
tool progress
timer fired
external webhook
human approval
cancellation
```

关键思想：

```text
外部数据不再插入 tool loop，而是作为 event 进入 mailbox。
```

---

## 4. 第三次抽象：等待是一等公民

我们进一步讨论了“等待”这个问题。

传统 tool loop 中没有真正的等待，只有同步阻塞。自主 agent 中的等待应该是显式状态：

```text
await job completed
await user reply
await timer
await external event
await approval
```

等待不应该占用模型调用，也不应该占用请求线程。它应该被持久化为 condition，等未来事件满足后再恢复 agent。

因此形成了：

```text
Wait/AwaitCondition 是 runtime primitive。
```

---

## 5. 第四次抽象：工具不止一种执行模式

我们区分了不同工具类型：

```text
sync tool          短工具，立即返回
async job          长任务，返回 job_id/action_run_id，未来完成
stream tool        持续产生 progress/log/partial result
subscription tool  订阅外部事件，未来触发
```

MVP 先实现：

```text
sync action
async action
```

这样既保留传统短工具的效率，又支持长任务不阻塞 Runtime。

---

## 6. 用户提出的架构图与关键想法

你提出了自己的架构图，核心是：

```text
Agent = Profile + Runtime(default)
```

并提出最小 MVP 可以从这几个部分开始：

```text
generator + action + task + event bus
```

其中：

- **generator**：使用模型根据 context 生成内容，本身是 actor 模式、有状态。
- **action**：向 generator 提供动作能力描述，并执行生成的动作。
- **task**：管理 agent 的所有任务，同步、异步、依赖推进、调度通知。
- **EventBus**：事件驱动。

你还提出：

```text
Task 是 Agent 活动的基本抽象，类似操作系统 PCB。
```

这个类比非常关键。它让 Task 不再只是 todo，而是 agent 活动的控制块。

---

## 7. 对 generator 的进一步共识

我们进一步明确：

```text
带状态的 actor 可以被视为 generator。
```

但这个 generator 不等于裸模型。更准确地说：

```text
Generator = model interface + context builder + state/workspace boundary + generation policy + output protocol
```

Generator 边界代表 agent 的能力边界之一。它能看到什么 context、知道哪些 action、能操作哪些 task、受哪些 policy 约束，都会影响它的能力。

---

## 8. Workspace 的补充共识

你随后指出：MVP 里不能缺少 Workspace。

我们把之前笼统说的 state 进一步细化为：

```text
Workspace 像 working memory。
Workspace 甚至应该负责 context builder 的材料。
Generator 只接收 context。
```

因此最终 MVP 中加入：

```text
Workspace
ContextBuilder
```

Workspace 负责当前工作记忆、短 transcript、当前 task focus、notes、variables、last decision summary。

---

## 9. 最终 MVP 共识

我们最终收敛到这套最小结构：

```text
EventBus
  -> AgentRuntime
  -> TaskSystem.apply_event
  -> RuntimePolicy.should_activate_generator
  -> Workspace/ContextBuilder.build
  -> Generator.generate
  -> CommandExecutor.apply
  -> ActionSystem / TaskSystem
  -> Persistence checkpoint
  -> EventBus future events
```

核心模块：

```text
Generator     认知决策器，输出 commands
Workspace     working memory + context builder input
ActionSystem  能力描述和执行边界
TaskSystem    活动 PCB、等待、取消、恢复
EventBus      时间、中断、异步结果入口
Persistence   状态和 checkpoint
RuntimePolicy 是否唤醒 Generator 的策略门
```

---

## 10. 这个 MVP 验证的行为模式

### 10.1 长异步任务

```text
用户请求分析大型项目
-> Generator 创建 Task
-> 启动 async action
-> 回复用户任务已开始
-> wait action completed
-> Runtime 释放控制权
-> 工具完成后发布 action.completed
-> Generator 恢复并生成报告
```

### 10.2 用户中途询问进度

```text
工具 progress 事件只更新 Task，不唤醒模型
用户问“进度怎么样”
-> user.message 唤醒 Generator
-> Generator 从 Task.progress 回复
```

### 10.3 用户中断取消

```text
用户说“不用做了，取消”
-> user.message 唤醒 Generator
-> Generator 输出 cancel_task
-> Runtime 取消 active ActionRun
-> Task 标记 cancelled
```

### 10.4 同步 action

```text
Generator 启动 sync action
-> ActionExecutor 立即生成 action.completed event
-> Runtime 重新唤醒 Generator
-> Generator 根据结果回复
```

---

## 11. 设计原则总结

最终形成的几个原则：

### 11.1 Runtime 是主循环

不是模型 while loop，而是 event-driven runtime。

### 11.2 Generator 只做决策

Generator 输出 command，不直接产生副作用。

### 11.3 Workspace 是工作记忆

Generator 只看 ContextBuilder 产出的 context。

### 11.4 Action 是能力边界

所有工具、副作用、长任务都通过 ActionSystem。

### 11.5 Task 是活动 PCB

等待、进度、取消、依赖、恢复都挂在 Task 上。

### 11.6 EventBus 是时间系统

外部中断和异步结果通过事件进入 agent。

### 11.7 Progress 默认不唤醒模型

防止长任务导致模型调用风暴。

### 11.8 continuation 保存恢复语义

异步结果回来时，agent 需要知道“为什么做这个任务”和“下一步应该做什么”。

---

## 12. 这个版本之后可以继续讨论的问题

当前 MVP 只是地基。下一步可以继续深入：

```text
真实 LLM adapter 怎么接入
GeneratorDecision schema 是否要更严格
Task dependency graph 怎么设计
Workspace 和长期 Memory 如何分层
RuntimePolicy 如何做预算、权限、审批
stream/subscription action 怎么进入协议
多 agent / 多 runtime worker 如何协调
事件 outbox/inbox 如何保证可靠投递
如何支持 human-in-the-loop approval
如何设计 UI 层的状态通知与交互协议
```
