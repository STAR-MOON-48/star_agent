# Agent Ling Web UI

这是原 `agent-ling-console` 的 Web 可视化版本。它沿用原终端 UI 的通信边界：Web UI 进程只创建独立的 Star Protocol Environment 与 Human client，不会在进程内直接调用 `AgentRuntime`。

## 技术结构

```text
Browser UI ── /live WebSocket ── Node Web Server
                                      ├── DialogueWorld (@star-world/core ECS)
                                      ├── EnvironmentClient (star-protocol)
                                      └── HumanClient (star-protocol)
                                                    │
                                                Star Hub
                                                    │
                                            agent-ling-star
```

- `@star-world/core`：管理场景参与者、在线状态、活动、布局与动态图脉冲。
- `star-protocol`：承载真实 Environment/Human/Agent 消息，不使用 UI 内部短路。
- 浏览器：显示实时场景拓扑、身份、聊天、动作、outcome 和协议载荷。
- Voice Live：浏览器本地 VAD 持续监听，调用小米 `mimo-v2.5-asr` 转文字并发送给 Agent；Agent 的 `assistant.message` 使用 `mimo-v2.5-tts` 流式合成 PCM16 并在浏览器播放。

## 启动

首次安装：

```bash
cd web_ui
npm install
```

Voice Live 使用与 Agent 相同的小米凭据。通过 Python 兼容命令启动时，会优先读取 `MIMO_API_KEY`、`XIAOMI_API_KEY`，没有环境变量时再复用 MengLong 的 `[providers.xiaomi]` 配置。直接运行 `npm start` 时请设置：

```bash
export XIAOMI_API_KEY="your-key"
# 可选：export XIAOMI_BASE_URL="https://api.xiaomimimo.com/v1"
```

启动 Web UI：

```bash
npm start -- \
  --hub-url ws://localhost:8000 \
  --env-id npc-dialogue-lab \
  --agent-id npc_agent \
  --human-id human_web \
  --human-name 林舟
```

浏览器打开 <http://127.0.0.1:4173>。如果指定端口已被占用，服务会自动顺延到下一个可用端口，并在启动日志中打印实际地址。也可以从 Python 项目根目录运行兼容命令：

```bash
uv run agent-ling-console --hub-url ws://localhost:8000 --env-id npc-dialogue-lab --agent-id npc_agent
```

随后在另一个终端启动 Agent：

```bash
uv run agent-ling-star \
  --hub-url ws://localhost:8000 \
  --env-id npc-dialogue-lab \
  --agent-id npc_agent \
  --no-startup-objective
```

可用参数与原终端版一致，包括 `--scene-title`、`--scene-background`、`--human-role`、`--human-background`、`--relationship`、`--conversation-id`、`--initial-message`、`--monitorable` 和 `--no-auto-reconnect`。Web 服务另支持 `--host` 与 `--port`。

语音参数：`--voice` 选择音色（默认“冰糖”），`--voice-language` 选择 `auto`、`zh` 或 `en`，`--voice-style` 控制朗读方式，`--mimo-base-url` 覆盖 API 地址。点击页面右上角 `VOICE LIVE` 后浏览器会请求麦克风权限；一句话结束后自动执行 ASR → Agent → TTS，播放期间暂停有效收音以避免回声，播放结束继续监听，直到点击“关闭 LIVE”。API Key 始终只保留在服务端。

## 验证

```bash
npm run build
npm test
```
