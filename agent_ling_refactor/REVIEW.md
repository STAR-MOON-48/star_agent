# Agent Ling 复盘与重构审查

## 审查范围

本次沿真实入口审查了配置、Runtime、Generator、Conversation、ContextBuilder、Task、Action、Memory、Emotion、Persistence、Star Protocol、Web UI 和现有测试。原 Python 核心约 1.9 万行，基线 `unittest` 24 项全部通过。

这说明原项目的主要问题不是功能错误，而是认知职责、运行时协议和 Prompt 之间耦合过深，继续扩展的成本偏高。

## 主要问题

### 0. 高频事件被错误地等同于高频认知

实际运行日志显示，Decision 请求中大量来源于 `action.internal.completed`、`task_not_runnable`、自身 Star 广播和持久化成功重放。这些事件需要落状态，但没有独立的认知价值。等待条件没有被用户消息确定性满足，又造成“动作被拒绝 → 失败事件唤醒模型 → 再次尝试”的反馈环。

重构新增独立模型激活门槛：

- EventBus 继续完整接收和持久化事件；
- 内部状态变更不反向唤醒 Decision；
- `task_not_runnable` 由调度器解释，不交给模型重试；
- 等价证据通过稳定指纹和 TTL 去重；
- 等待中的任务只由其等待条件对应的外部事件唤醒；
- 单条用户输入或自主目标拥有有限的 Decision hop budget；
- 429/timeout 建立跨事件持久化退避，后台反思也尊重退避；
- Memory Reflection 和 DMN 都使用 single-flight 标记，且 DMN 不打断未完成任务。

启动目标也不再伪装成 sender=`startup` 的用户消息，而是以 `runtime.objective` 直接进入 Decision。这同时消除了虚假 recipient 404 和两次无意义的 Understanding/Expression 调用。

### 1. Prompt 承担了过多 Runtime 正确性

原默认配置约 1.7 万字符。Decision Prompt 同时解释角色边界、工具协议、内部任务工具、任务树调度、完成门槛、重试、上下文选择、长期记忆、Star 工具来源和外部状态判断。

这些规则中，大量内容应由类型、调度器和执行器保证。把它们放进 Prompt 会带来三个问题：

- 每轮重复消耗上下文；
- 模型需要同时扮演认知区域和 Runtime 解释器；
- 代码行为与 Prompt 描述容易在迭代中漂移。

重构后，系统 Prompt 只包含区域职责、Ling 的必要自我约束和少量共同规则。当前五类系统 Prompt 实际渲染长度约 350–470 字。

### 2. 内部协议层次重叠

原链路同时存在：

- `AgentEvent.payload`；
- `ConversationUnderstanding`；
- `speech_intent`；
- `GeneratorDecision.commands`；
- 模型原生 tool call；
- 普通自然语言 decision text。

同一个意图可能经历多次结构转换。尤其是 tool call 先变成 GeneratorDecision，再变成 Runtime command，增加了解析、恢复和兜底分支。

重构将边界分成两类：

- 认知区域之间：`NaturalMessage`，内容只能是自然语言；
- Runtime 和外部能力之间：Event、Task、ActionSpec、tool call，继续保持结构化。

这样不牺牲机器执行的可靠性，也不让内部认知区域互相依赖私有 JSON 方言。

### 3. 对话区域与运行时编排耦合

原 `AgentRuntime` 约 1900 行，负责生命周期、协议循环、记忆循环、DMN、对话事件状态机、模型日志、重试、决策应用、任务恢复和对外发送。Conversation 相关实现另有约 1280 行，Generator/Context 层也承担大量对话格式转换。

重构把职责拆为：

- `ConversationLedger`：只负责持久对话和自然语言交接；
- `PromptCompiler`：只负责区域 Prompt；
- `ContextComposer`：按区域给最小必要上下文；
- `ModelGateway`：只负责一次模型边界调用；
- `RefactoredRuntime`：只做事件路由和并发编排。

### 4. Broca 上下文被无关运行时信息污染

Broca 的职责是表达。任务调度、行动目录、分页检索和完整记忆协议不应进入它的 Prompt 或上下文。

重构后的 Expression 上下文只有：

- 自然语言表达交接；
- 最近交流；
- 当前情绪等表达状态。

任务、行动和相关记忆只进入 Decision 或 Understanding 的相应上下文。

### 5. “响应速度”不等于压缩认知区域

普通对话保留 Wernicke → Broca 是合理设计：先理解，再形成符合人格和关系状态的表达。真正可优化的是行动型对话中不必要的串行等待。

重构采用：

1. Understanding 先完成自然语言理解；
2. 若需要 Decision，同时启动 Decision 与 Expression；
3. Broca 先完成并发送对外话语；
4. Decision 在后台推进任务和行动；
5. Action 结果到来后，Decision 的自然语言表达意图再次交给 Broca。

这保留了认知分区，也降低了对外回复等待其他区域处理的概率。

## 保留而非重写的部分

以下实现已有明确测试和状态兼容价值，本次选择复用：

- `AgentEvent`、`AgentState`、`AgentTask`、`ActionRun`；
- `TaskSystem` 的任务树、依赖、等待和完成门槛；
- `ActionExecutor` 的同步/异步执行和副作用幂等；
- `JsonStateStore`、`ConversationStore`、`MemoryStore`；
- `MemorySystem` 的捕获、脱敏和检索；
- `EmotionSystem`；
- `StarModel`、`StarSession` 和 `PerceptionSystem`；
- 独立的 Web UI / Star World 场景。

这是一种渐进式重构：替换高耦合编排和 Prompt，而不是把已经验证的基础能力复制一遍后重新制造兼容问题。

## 仍然存在的边界

- EventBus 仍是进程内队列，不是持久消息队列；进程在事件入队但未 checkpoint 时退出，仍可能需要外部重投。
- 真实模型和真实 Star Hub 的联调依赖外部服务与凭据；本地测试使用可控 ModelInterface 验证完整状态闭环。
- 新 Runtime 复用原 `agent` 基础包，因此当前适合在同一仓库并行评估。若未来要拆成独立发布包，可以在接口稳定后再提取公共 core。
- 多个模型区域并发依赖 ModelInterface/provider 允许并发请求；如果某 provider 强制串行，可以在 ModelGateway 外增加按 provider 配置的并发限流，而不改变区域消息协议。

## 建议迁移顺序

1. 先用独立 agent id 和 `.agent_state_refactor` 跑本地对话回归；
2. 接入测试 Star 环境，验证工具发现、异步 outcome 和 recipient 路由；
3. 对比同一组会话的 Prompt tokens、首条回复耗时、工具成功率和重复行动率；
4. 确认状态与行为后，再把部署入口从 `agent-ling-star` 切到 `agent-ling-refactor-star`；
5. 保留旧入口一个发布周期作为回退。
