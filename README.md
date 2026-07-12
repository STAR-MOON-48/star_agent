# Agent Tool Loop MVP：事件驱动、任务中心的自主交互 Agent Runtime

并行重构版本位于 [`agent_ling_refactor/`](agent_ling_refactor/README.md)。它保留 Wernicke → Broca 的普通对话链路，改为自然语言区域交接、精简 Prompt，并让理解完成后的表达与决策按需并发；原实现仍可独立运行和回退。

这是一个最小可运行版本，用来验证我们讨论的 agent 架构地基：

```text
Agent = Profile + Runtime
Runtime = Kernel + Interfaces + StateSystems + PerceptionSystems
        + CognitionSystem + ActionSystems + PersistenceSystem
```

这个 MVP 不追求完整 Agent OS，而是先把核心闭环跑通，并在终端展示结构化 trace：

```text
Event -> Runtime -> Workspace/ContextBuilder -> DecisionGenerator -> Commands
      -> ActionExecutor / TaskSystem -> EventBus -> future activation
```

## 这个版本验证了什么

1. **Generator 不是 while tool loop 的 owner**：Runtime 才是主循环，Generator 只在事件到来时被激活。
2. **Workspace 是 working memory 和 context builder 的基础**：Generator 只接收 context，不直接读完整内部状态。
3. **Task 是 agent 活动的基本抽象**：长任务、等待、取消、恢复都围绕 Task 管理。
4. **Action 是能力边界**：Generator 只能输出 action intent，副作用由 ActionExecutor 执行。
5. **EventBus 是时间与中断机制**：用户消息、工具进度、工具完成、取消都变成事件。
6. **同步和异步 action 可并存**：短 action 立即完成，长 action 异步执行并通过事件恢复。

## 目录结构

```text
agent_ling/
  README.md
  pyproject.toml
  uv.lock
  agent/
    __init__.py
    protocols.py             # Event/Task/Workspace/ActionRun/AgentState 协议对象
    config/
      loader.py              # 通用 AgentConfig schema 与显式 TOML 加载
    profile/
      __init__.py            # AgentProfile 导出入口
    runtime/
      __init__.py            # AgentRuntime/JsonStateStore 导出入口
      kernel/
        runtime.py           # AgentRuntime 与 RuntimePolicy
        event_bus.py         # 内存 EventBus
        generator_runtime.py # GeneratorRuntime，管理 GeneratorSession actor
        generator_session.py # GeneratorSession actor 及 LLMGeneratorSession
      interfaces/
        model.py             # ModelInterface 和通用模型返回结构
        star_model.py        # MengLong-backed StarModel
        protocol.py          # ProtocolInterface
        star_session.py      # Star Protocol-backed StarSession
      state_systems/
        workspace.py         # ContextBuilder
        context_policy.py    # session token 预算、候选单元选择与请求估算
        memory_system/       # 情景记忆捕获、检索与自主经验反思
      perception_systems/
        perception.py        # PerceptionSystem，统一感知入口
      cognition_system/
        conversation_system/ # ConversationManager / Wernicke / Broca
        decision_system/     # 决策上下文、模型决策与审计历史
        emotion_system/      # 可衰减情绪、心境和社会情感状态
        dmn.py               # Default Mode Network
      action_systems/
        actions.py           # ActionSpec registry 与 sync/async executor
        task_system.py       # Task 树、依赖、调度派生状态与完成门槛
      persistence_system/
        store.py             # JSON 状态与 checkpoint 持久化
        conversation_store.py # 独立对话持久化
        memory_store.py      # Markdown 长期记忆与 JSON 检索索引
  agent_ling/
    __init__.py              # 具体 app 导出入口
    app.py                   # AgentApplication = profile/config + AgentRuntime
    config/
      default_agent.toml     # 默认 Agent profile、模块 prompt、入口目标配置
      loader.py              # app 默认配置加载与用户覆盖
    entrypoints/
      star_agent.py          # Star Protocol agent 启动入口
      web_ui.py              # Web UI 启动桥接入口
      console_ui.py          # 旧 Textual UI（兼容命令 agent-ling-console-tui）
  web_ui/                    # Star World + Star Protocol Web 可视化场景
  demo/
    task_tree_demo.py
    context_budget_demo.py
    context_continuity_demo.py
    conversation_context_compaction_demo.py
    generator_demo.py
    conversation_demo.py
  docs/
    ARCHITECTURE.md        # 面向实现的架构与模块边界文档
    DESIGN.md              # 整体设计文档
    PROTOCOL.md            # 数据结构和协议文档
    DISCUSSION_HISTORY.md  # 我们前面讨论的思路演进文档
```

## 运行方式

项目使用 `uv` 管理 Python 版本、依赖和锁文件。需要 Python 3.13+。

```bash
cd agent_ling
uv sync
uv run python demo/task_tree_demo.py
```

Agent 的 profile、Generator prompt、Star 启动目标不写在代码里，默认配置在：

```bash
agent_ling/config/default_agent.toml
```

其中 Profile 描述 Agent 的稳定 Self Model，包括身份、规范性背景、性格、价值观、行为原则、声色、说话习惯、关系立场和自我边界；Generator prompt 和 context policy 按 session/system 拆分。DecisionSystem 使用 `generator.sessions.decision`，ContextBuilder 使用 `context_builder`，MemorySystem 使用 `memory_reflection` 自主沉淀 Markdown 经验。

默认 Ling 的角色档案位于 `[agent.profile]`。这里的 `background_profile` 是 Ling 用来维持自我连续性的规范性历史，不等于当前外部世界事实；情景记忆同样保留来源和置信度，需要采取副作用动作时仍要核实现状。`voice_profile` 定义整体声色，`speech_profile` 定义具体措辞习惯，最终都由 BrocaSystem 落实，而不是散落在各业务模块里。

可以用自定义 TOML 覆盖：

```bash
uv run python demo/generator_demo.py --agent-config ./my_agent.toml
uv run agent-ling-star --agent-config ./my_agent.toml
```

模块边界：

- `Kernel`：`AgentRuntime`、`RuntimePolicy`、`EventBus`。
- `agent/`：只放可复用模块，包括协议、runtime、interfaces、state/action/persistence/perception 系统和通用配置 schema。
- `agent_ling/`：具体 agent 应用，负责加载默认 profile/prompt 配置，并装配 `AgentApplication = AgentConfig + AgentRuntime`。
- `Config`：`agent_ling/config/default_agent.toml` 提供默认 AgentProfile、按 session 拆分的 Generator prompts、入口默认目标。
- `StateSystems`：实现 `Workspace/ContextBuilder/MemorySystem`。MemorySystem 自动捕获关键事件、脱敏保存情景记忆，并在空闲时固化可复用经验。
- `Kernel.GeneratorRuntime`：按 `generator_session` 管理 generator session actor，并把 runtime context 转成 MengLong `Context`。
- `Kernel.GeneratorSession`：事件驱动的 generator session actor，当前真实实现是 `LLMGeneratorSession`。
- `CognitionSystem`：实现 `ConversationSystem/Wernicke/Broca/DecisionSystem/EmotionSystem/DMN`；DecisionSystem 组合记忆、情绪、任务与工具证据后调用独立 generator session。
- `ActionSystems`：当前实现 `ActionRegistry`、`ActionExecutor`、`TaskSystem`。
- `PersistenceSystem`：当前实现 `JsonStateStore` 和独立 `ConversationStore`。
- `MemoryStore`：每条长期记忆以 Markdown 为源文件，并用 JSON 索引提供 `search_memory/read_memory`。
- `Interfaces`：当前实现 `StarModel` 和 `StarSession`。
- `PerceptionSystems`：当前实现 `PerceptionSystem`，负责把本地输入、Star Protocol action/event/outcome/stream 归一化为内部 `AgentEvent`，并通过 `EventBus` 发布给 runtime。

真实模型已接入：`GeneratorRuntime` 默认使用 `StarModel`，模型 ID 是 `xiaomi/mimo-v2.5`。需要通过 MengLong 配置或环境变量提供小米模型访问配置，例如 `XIAOMI_API_KEY`，可选 `XIAOMI_BASE_URL`。

不访问模型即可观察任务树调度和 root 完成门槛：

```bash
uv run python demo/task_tree_demo.py
```

不访问模型即可构造大量历史，观察完整 Store 与预算内模型请求的区别：

```bash
uv run python demo/context_budget_demo.py
```

不访问模型即可复现并验证连续任务中的能力目录、重复尝试记忆和 root task 进度语义：

```bash
uv run python demo/context_continuity_demo.py
```

使用真实模型观察两轮 Wernicke 理解、对话记忆和 Broca 表达：

```bash
uv run python demo/conversation_demo.py
```

`user.message` 会先进入 ConversationSystem。Wernicke 通过内部检索工具理解说话者；只有涉及行动、任务、承诺或重要判断时才请求 DecisionSystem。Decision 的自然语言只作为表达意图，最终对外文本统一由 Broca 根据稳定 Self Model、当前情绪、关系状态和最近对话生成。完整 turn、understanding 和 outbound utterances 保存在 `ConversationStore`。

关键事件会同步进入 MemorySystem；DecisionSystem 自动召回相关长期记忆，Wernicke 也可主动检索。EmotionSystem 根据话语、行动成功/失败和任务完成情况更新持久情绪，状态随时间回归基线，并同时提供给 Wernicke、DecisionSystem、DMN 和 Broca。

上下文采用“完整持久化、预算内选择、按引用回读”的策略。Decision、ContextBuilder、MemoryReflection 和 DMN 分别在 `default_agent.toml` 中配置 context policy。模型请求前会估算 messages 与 tools 的输入长度；超过配置预算时显式失败，不依赖模型服务静默裁剪。小型外部 action catalog 在 tool budget 内会完整发送；大型目录才按相关性选择并通过 `search_actions` 回读。最近动作与重复尝试会形成紧凑 execution memory，完整 tool result 仍保留在 Store。旧阶段材料超过阈值时，ContextBuilderSystem 低频生成滚动 Markdown 工作摘要，原始 task/action/transcript/note 仍可通过 `search_workspace`、`read_task`、`read_action_run` 和 `search_actions` 回读。

启动接入 Star Protocol 的 agent：

```bash
uv run agent-ling-star --agent-id agent_ling --hub-url ws://localhost:8000 --env-id demo_env
```

启动后默认会在 Star 工具 discover 完成后提交一个 startup objective，要求 agent 使用外部 `star_protocol` 工具观察环境、列出任务和活动，并持续推进到完成、失败、取消或阻塞。如果只想挂起等待外部用户消息：

```bash
uv run agent-ling-star --no-startup-objective
```

也可以通过 Python 模块直接运行同一个入口：

```bash
uv run python -m agent_ling.entrypoints.star_agent --agent-id agent_ling --hub-url ws://localhost:8000 --env-id demo_env
```

Star 侧用户输入可以以 event/action 形式发送给 agent，例如 action 名为 `user_message`，参数里带 `content="..."`。`PerceptionSystem` 会把它转换成内部 `user.message`，后续回复会通过 Star event `assistant.message` 发回原 sender。

真实接口用法：

```python
from agent import JsonStateStore
from agent.runtime.interfaces import StarModel, StarSession
from agent_ling.app import create_agent_runtime

model = StarModel()  # default_model_id="xiaomi/mimo-v2.5"

session = StarSession(
    hub_url="ws://localhost:8000",
    env_id="demo_env",
)

application = create_agent_runtime(
    agent_id="agent_loop",
    store=JsonStateStore(".agent_state"),
    model_interface=model,
    protocol_interface=session,
)
runtime = application.runtime
```

`StarSession` 启动后会通过 Star Protocol 加入环境、discover 工具，并把环境工具注册成 `source="star_protocol"` 的 `ActionSpec`。当 generator 发起这些 action 时，`ActionExecutor` 会通过 Star Protocol 下发；环境返回的 outcome 会转换成内部 `action.completed` / `action.failed` 事件。

## Web 交互式 NPC 对话场景

`agent-ling-console` 现在启动 Web 可视化场景。它仍是独立场景客户端，不会在 UI 进程里创建或直接调用 `AgentRuntime`：

```text
Browser ─ WebSocket ─ DialogueWorld (@star-world/core)
                         ├─ EnvironmentClient ─┐
                         └─ HumanClient       ─┼─ Star Hub ─ AgentClient
```

首次使用先安装 Web 依赖：

```bash
cd web_ui
npm install
cd ..
```

Hub 已启动时，启动 Web UI。它会创建 Environment，并用命令行提供的资料模拟一个已经登录的 Human：

```bash
uv run agent-ling-console \
  --hub-url ws://localhost:8000 \
  --env-id npc-dialogue-lab \
  --agent-id npc_agent \
  --human-id human_web \
  --human-name 沈岚 \
  --human-role 星港研究所的值班工程师 \
  --human-background "熟悉设备维护，第一次与这个 NPC 见面" \
  --relationship "初次见面，可能长期协作"
```

然后打开启动日志显示的 Web 地址（默认 `http://127.0.0.1:4173`；端口被占用时会自动顺延）。Web UI 支持中央 ECS 场景拓扑、参与者在线状态、Human/Agent 对话、环境 action/outcome 动画、协议载荷检查和断线重连。

右上角 `VOICE LIVE` 可开启持续语音对话：浏览器用本地 VAD 自动截取话段，小米 `mimo-v2.5-asr` 转成文字后作为正常 `user_message` 交给 Agent，Agent 的每条 `assistant.message` 再由 `mimo-v2.5-tts` 流式合成并播放。播放期间暂停有效收音，结束后自动恢复，直至手动关闭 Live。通过 `uv run agent-ling-console` 启动时会复用环境变量或 MengLong Xiaomi provider 中的 API Key，凭据不会发送到浏览器。

再在另一个终端启动独立 Agent client：

```bash
uv run agent-ling-star \
  --hub-url ws://localhost:8000 \
  --env-id npc-dialogue-lab \
  --agent-id npc_agent \
  --no-startup-objective
```

Web UI 收到目标 Agent 的加入事件后才开放输入，避免把消息发给尚未在线的 recipient。每次 Human action 都携带 `speaker_context`、`scene_context` 和稳定的 `conversation_id`；Agent 回复使用 Star event `assistant.message`。浏览器只连接本地 `/live` 通道，真实 Star websocket 由服务端的 TypeScript SDK 管理。

每条 Web Human 输入还携带稳定 `message_id`，Agent 会立即返回 action outcome，并以该 ID 对重投消息去重。短时间连续输入会跳过已经被更新话轮取代、尚未开始理解的旧消息，减少无意义模型调用。如果 Human 明确说“不用等我回复”“不用一问一答”或“继续说”，ConversationSystem 会按配置间隔主动续说一个有限、可打断的自然话语段；新输入会切换到最新话轮，“先别说了”等指令会立即取消待发送续话。

场景 Environment 向 Agent discover 三项真实 Star 工具：`observe_social_scene`、`read_human_profile`、`perform_social_action`。可以用 `--initial-message` 在 Agent 加入后自动发送第一句话。旧 Textual 版本仍可用 `uv run agent-ling-console-tui` 启动。

`demo/generator_demo.py` 和 `demo/conversation_demo.py` 使用真实的
`GeneratorRuntime + LLMGeneratorSession + StarModel`，会实际访问默认模型
`xiaomi/mimo-v2.5`。其余回归 Demo 不访问模型。

长程任务没有回退到传统 `while tool-loop`。当前等价闭环是：

```text
event -> generator -> command -> action/tool -> event -> generator
```

只要任务没有进入 `completed` / `failed` / `cancelled` 终态，runtime 就持续消费事件；`action.completed` 会重新激活 generator，让任务继续推进到最终结果。

可直接运行的离线回归 Demo：

```bash
uv run python demo/task_tree_demo.py
uv run python demo/context_budget_demo.py
uv run python demo/context_continuity_demo.py
uv run python demo/conversation_context_compaction_demo.py
```

真实模型 Demo：

```bash
uv run python demo/generator_demo.py --model xiaomi/mimo-v2.5
uv run python demo/generator_demo.py --agent-config ./my_agent.toml
uv run python demo/conversation_demo.py
```

`agent-ling-star` 默认在 `.agent_state/` 下生成状态、checkpoint、模型调用日志和对话记录：

```text
.agent_state/
  {agent_id}.state.json
  checkpoints/{agent_id}.jsonl
  logs/{agent_id}.generator.jsonl
  conversations/{agent_id}.conversation.json
  memories/{agent_id}/index.json
  memories/{agent_id}/episodic/*.md
  memories/{agent_id}/semantic/*.md
```

## Demo 场景

### Task tree

验证任务树、依赖、等待条件、子任务完成门槛和 root 收口。

### Context budget / continuity

验证完整持久化、预算内上下文选择、引用回读、动作目录连续性和重复尝试记忆。

### Conversation context compaction

验证对话远期摘要、近期原文保留和模型边界压缩，不修改 `ConversationStore` 中的原始记录。

## 这个 MVP 刻意没有做的事

- 没有真实消息队列，`EventBus` 是内存 `asyncio.Queue`。
- 没有复杂权限、安全、预算、审批。
- 没有多 agent、多 worker 分布式锁。
- 没有 stream/subscription action，只先实现 sync/async。

这些都可以在协议稳定后逐步替换或扩展。

## 后续扩展方向

最自然的下一步是把 `EventBus` 替换为持久队列，把 `JsonStateStore` 替换为数据库，并把 ActionSpec 扩展为真正的 tool/action registry。
