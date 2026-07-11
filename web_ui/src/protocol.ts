import { randomUUID } from "node:crypto";

import {
  EnvironmentClient,
  HumanClient,
  type MessageActionContent,
  type MessageEventContent,
  type MessageOutcomeContent,
  type MessageStreamContent,
  type SystemErrorContent,
  type ToolDefinition,
  type UnknownRecord,
} from "star-protocol";

import { DialogueWorld } from "./dialogue-world.js";

export interface DialogueProtocolOptions {
  onAssistantMessage?: (
    sender: string,
    content: string,
    metadata: Record<string, unknown>,
  ) => void;
}

export const DIALOGUE_TOOLS: readonly ToolDefinition[] = [
  {
    name: "observe_social_scene",
    description: "观察当前 NPC 对话实验场景、参与者和可公开的人类身份背景。",
    parameters: {
      type: "object",
      properties: { focus: { type: "string", description: "希望重点观察的内容，可留空。" } },
      additionalProperties: false,
    },
    tags: ["scene", "social", "observe"],
  },
  {
    name: "read_human_profile",
    description: "读取当前场景中已登录 Human 的公开身份、背景和与 Agent 的关系设定。",
    parameters: {
      type: "object",
      properties: { human_id: { type: "string", description: "Human client id；留空表示当前登录 Human。" } },
      additionalProperties: false,
    },
    tags: ["human", "profile", "social"],
  },
  {
    name: "perform_social_action",
    description: "在场景中执行可观察的非语言社会动作，例如点头、挥手、递出物品或转身。",
    parameters: {
      type: "object",
      properties: {
        action: { type: "string", description: "社会动作名称。" },
        target: { type: "string", description: "动作对象 client id。" },
        description: { type: "string", description: "动作的自然语言细节。" },
      },
      required: ["action"],
      additionalProperties: false,
    },
    tags: ["social", "action", "nonverbal"],
  },
];

class WebEnvironmentClient extends EnvironmentClient {
  constructor(private readonly dialogue: DialogueWorld) {
    super(dialogue.config.scene.envId, {
      autoReconnect: dialogue.config.autoReconnect,
      monitorable: dialogue.config.monitorable,
    });
  }

  override async onConnected(): Promise<void> {
    this.dialogue.protocol("environment_connected", "Environment 已连接 Hub", {
      client_id: this.clientId,
    }, "success");
  }

  override async onDisconnected(): Promise<void> {
    this.dialogue.setConnection("disconnected", false);
    this.dialogue.protocol("environment_disconnected", "Environment 已断开 Hub", {
      client_id: this.clientId,
    }, "warning");
  }

  override async onDiscover(sender: string): Promise<readonly ToolDefinition[]> {
    this.dialogue.agentDiscovered(sender);
    this.dialogue.protocol("environment_discover", `${sender} 获取场景工具`, {
      sender,
      tool_names: DIALOGUE_TOOLS.map((tool) => tool.name),
    });
    return DIALOGUE_TOOLS;
  }

  override async onAction(sender: string, content: MessageActionContent): Promise<void> {
    const name = content.name || "unknown";
    const actionId = content.id ?? "";
    const params = asRecord(content.params);
    this.dialogue.environmentAction(sender, name, actionId, params);

    let success = true;
    let data: Record<string, unknown>;
    if (name === "observe_social_scene") {
      data = {
        scene: this.dialogue.sceneContext(),
        human_public_profile: this.dialogue.humanContext(),
        observation: `${this.dialogue.config.scene.title}。${this.dialogue.config.scene.background} 当前参与者：${this.dialogue.participantIds().sort().join("、") || "无"}。`,
      };
    } else if (name === "read_human_profile") {
      const requestedId = String(params.human_id ?? this.dialogue.config.human.humanId);
      if (requestedId !== this.dialogue.config.human.humanId) {
        success = false;
        data = { error: `场景中没有公开资料属于 ${requestedId}` };
      } else {
        data = {
          profile: this.dialogue.humanContext(),
          source: "scenario_login_context",
          verified_by_environment: true,
        };
      }
    } else if (name === "perform_social_action") {
      data = {
        actor: sender,
        target: params.target ?? this.dialogue.config.human.humanId,
        action: params.action,
        description: params.description,
        scene: this.dialogue.config.scene.envId,
        performed: true,
      };
      await this.broadcastToEnvironment(this.dialogue.config.scene.envId, "social_action_performed", data);
    } else {
      success = false;
      data = { error: `Unknown dialogue-scene action: ${name}` };
    }

    await this.sendOutcome(sender, {
      ref_id: actionId,
      success,
      ...(success ? { data } : { error: String(data.error ?? "Action failed") }),
    });
    this.dialogue.environmentOutcome(sender, name, success, data);
  }

  override async onEvent(sender: string, content: MessageEventContent): Promise<void> {
    this.dialogue.protocol("environment_event", `${sender} 向环境发送 ${content.name}`, {
      sender,
      content: content as unknown as Record<string, unknown>,
    });
  }

  override async onClientJoined(clientId: string): Promise<void> {
    this.dialogue.participantJoined(clientId);
  }

  override async onClientLeft(clientId: string, reason = "leave"): Promise<void> {
    this.dialogue.participantLeft(clientId, reason);
  }

  override async onError(error: SystemErrorContent): Promise<void> {
    this.dialogue.protocol("environment_error", `Environment 协议错误：${error.msg}`, {
      code: error.code,
      msg: error.msg,
    }, "error");
  }
}

class WebHumanClient extends HumanClient {
  constructor(
    private readonly dialogue: DialogueWorld,
    private readonly callbacks: DialogueProtocolOptions,
  ) {
    super(dialogue.config.human.humanId, {
      autoReconnect: dialogue.config.autoReconnect,
      monitorable: dialogue.config.monitorable,
    });
  }

  override async onConnected(): Promise<void> {
    this.dialogue.protocol("human_connected", "Human 已连接 Hub", { client_id: this.clientId }, "success");
  }

  override async onDisconnected(): Promise<void> {
    this.dialogue.protocol("human_disconnected", "Human 已断开 Hub", { client_id: this.clientId }, "warning");
  }

  override async onEvent(sender: string, content: MessageEventContent): Promise<void> {
    const data = asRecord(content.data);
    if (content.name === "assistant.message") {
      const text = String(data.content ?? "");
      this.dialogue.assistantMessage(sender, text, data);
      this.callbacks.onAssistantMessage?.(sender, text, data);
      return;
    }
    this.dialogue.protocol("human_event", `${sender} 发送事件 ${content.name}`, {
      sender,
      name: content.name,
      data,
    });
  }

  override async onOutcome(sender: string, content: MessageOutcomeContent): Promise<void> {
    this.dialogue.protocol("human_outcome", `Human 收到 ${sender} 的 outcome`, {
      sender,
      content: content as unknown as Record<string, unknown>,
    });
  }

  override async onStream(sender: string, content: MessageStreamContent): Promise<void> {
    this.dialogue.protocol("human_stream", `收到 ${sender} 的流式数据`, {
      sender,
      content: content as unknown as Record<string, unknown>,
    });
  }

  override async onBroadcastEvent(sender: string, content: MessageEventContent): Promise<void> {
    this.dialogue.protocol("human_broadcast", `场景广播：${content.name}`, {
      sender,
      content: content as unknown as Record<string, unknown>,
    });
  }

  override async onError(error: SystemErrorContent): Promise<void> {
    this.dialogue.protocol("human_error", `Human 协议错误：${error.msg}`, {
      code: error.code,
      msg: error.msg,
    }, "error");
  }
}

export class DialogueProtocol {
  private environment: WebEnvironmentClient | null = null;
  private human: WebHumanClient | null = null;
  private connecting: Promise<void> | null = null;
  private generation = 0;
  private stopped = false;
  private pendingInitialMessage: string | undefined;

  constructor(
    private readonly dialogue: DialogueWorld,
    private readonly callbacks: DialogueProtocolOptions = {},
  ) {
    this.pendingInitialMessage = dialogue.config.initialMessage;
  }

  async connect(): Promise<void> {
    if (this.connecting) return this.connecting;
    this.stopped = false;
    const generation = ++this.generation;
    this.connecting = this.connectGeneration(generation).finally(() => {
      if (generation === this.generation) this.connecting = null;
    });
    return this.connecting;
  }

  async reconnect(): Promise<void> {
    await this.disconnect(false);
    await this.connect();
  }

  async disconnect(final = true): Promise<void> {
    if (final) this.stopped = true;
    this.generation += 1;
    this.connecting = null;
    const human = this.human;
    const environment = this.environment;
    this.human = null;
    this.environment = null;
    await Promise.allSettled([human?.stop(), environment?.stop()]);
    this.dialogue.setConnection("disconnected", false, "Star clients 已断开");
  }

  async sendHumanMessage(raw: string): Promise<{ ok: true; messageId: string } | { ok: false; error: string }> {
    const content = raw.trim();
    const agentId = this.dialogue.agentId;
    const conversationId = this.dialogue.conversationId;
    if (!content) return { ok: false, error: "消息不能为空" };
    if (!this.human?.isConnected) return { ok: false, error: "Human 尚未连接 Star Hub" };
    if (!this.dialogue.isOnline(agentId)) {
      return { ok: false, error: `Agent ${agentId} 尚未加入场景` };
    }
    try {
      const messageId = `human-${Date.now()}-${randomUUID().slice(0, 8)}`;
      await this.human.sendAction(agentId, "user_message", {
        content,
        message_id: messageId,
        conversation_id: conversationId,
        speaker_context: this.dialogue.humanContext(),
        scene_context: this.dialogue.sceneContext(),
      });
      this.dialogue.humanMessage(content, messageId, agentId);
      return { ok: true, messageId };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.dialogue.protocol("send_error", `消息发送失败：${message}`, { target: agentId }, "error");
      return { ok: false, error: message };
    }
  }

  async sendInitialMessageIfReady(): Promise<void> {
    if (!this.pendingInitialMessage || !this.dialogue.isOnline(this.dialogue.agentId)) return;
    const content = this.pendingInitialMessage;
    this.pendingInitialMessage = undefined;
    const result = await this.sendHumanMessage(content);
    if (!result.ok) this.pendingInitialMessage = content;
  }

  private async connectGeneration(generation: number): Promise<void> {
    this.dialogue.setConnection("connecting", false, "正在连接 Star Hub");
    let environment: WebEnvironmentClient | null = null;
    let human: WebHumanClient | null = null;
    try {
      environment = new WebEnvironmentClient(this.dialogue);
      await environment.connect(this.dialogue.config.hubUrl);
      if (generation !== this.generation || this.stopped) {
        await environment.stop();
        return;
      }
      await environment.start();
      this.environment = environment;

      human = new WebHumanClient(this.dialogue, this.callbacks);
      await human.connect(this.dialogue.config.hubUrl);
      if (generation !== this.generation || this.stopped) {
        await human.stop();
        await environment.stop();
        return;
      }
      await human.start();
      await human.joinEnvironment(this.dialogue.config.scene.envId);
      this.human = human;
      this.dialogue.setConnection("connected", true, "Environment 与 Human 已连接 Star Hub");
      await this.sendInitialMessageIfReady();
    } catch (error) {
      if (generation !== this.generation || this.stopped) return;
      await Promise.allSettled([human?.stop(), environment?.stop()]);
      if (this.human === human) this.human = null;
      if (this.environment === environment) this.environment = null;
      const message = error instanceof Error ? error.message : String(error);
      this.dialogue.setConnection("error", false, `连接失败：${message}`);
    }
  }
}

function asRecord(value: unknown): UnknownRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as UnknownRecord
    : {};
}
