import { App, component, type Entity } from "@star-world/core";

import { Activity, Identity, Layout, Presence, participantQuery } from "./components.js";
import { SceneMotionSystem } from "./systems.js";
import type {
  DialogueConfig,
  DialogueMessage,
  DialoguePersistenceState,
  DialogueSession,
  DialogueSnapshot,
  ParticipantRole,
  ProtocolNotice,
  ScenePulse,
} from "./types.js";

const MAX_MESSAGES = 120;
const MAX_NOTICES = 180;
const MAX_PULSES = 32;
const PULSE_LIFETIME_MS = 12_000;

export class DialogueWorld {
  readonly app = new App({ world: { deterministic: true } });

  private readonly entities = new Map<string, Entity>();
  private readonly sessions = new Map<string, DialogueSession>();
  private readonly notices: ProtocolNotice[] = [];
  private readonly pulses: ScenePulse[] = [];
  private revision = 0;
  private actionCount = 0;
  private browserClients = 0;
  private connectionState: DialogueSnapshot["connection"]["state"] = "connecting";
  private hubConnected = false;
  private readonly discoveredAgentIds = new Set<string>();
  private targetAgentId: string;
  private manualTargetSelected: boolean;

  constructor(
    readonly config: DialogueConfig,
    private readonly now = Date.now,
    restored?: DialoguePersistenceState,
  ) {
    this.targetAgentId = restored?.activeAgentId || config.agentId;
    this.manualTargetSelected = Boolean(restored?.manualTargetSelected);
    this.restoreSessions(restored);
    this.app.addSystem(new SceneMotionSystem(), { stage: "update" });
    this.upsertParticipant(config.scene.envId, "environment", config.scene.title, true);
    this.upsertParticipant(config.human.humanId, "human", config.human.displayName, true);
    this.upsertParticipant(config.agentId, "agent", config.agentId, false);
    if (this.targetAgentId !== config.agentId) {
      this.upsertParticipant(this.targetAgentId, "agent", this.targetAgentId, false);
    }
    this.notice("boot", "Web 场景已创建，正在连接 Star Hub", {
      hub_url: config.hubUrl,
      environment_id: config.scene.envId,
    });
  }

  tick(deltaSeconds: number): void {
    this.app.step(Math.max(0, Math.min(deltaSeconds, 0.5)));
    const cutoff = this.now() - PULSE_LIFETIME_MS;
    const expired = this.pulses.findIndex((pulse) => pulse.at < cutoff);
    if (expired >= 0) this.pulses.splice(expired);
  }

  setBrowserClients(count: number): void {
    this.browserClients = Math.max(0, count);
    this.touch();
  }

  setConnection(
    state: DialogueSnapshot["connection"]["state"],
    hubConnected: boolean,
    detail?: string,
  ): void {
    this.connectionState = state;
    this.hubConnected = hubConnected;
    this.setOnline(this.config.scene.envId, hubConnected, hubConnected ? "hosting" : "offline");
    this.setOnline(this.config.human.humanId, hubConnected, hubConnected ? "joined" : "offline");
    if (!hubConnected) this.setOnline(this.targetAgentId, false, "waiting");
    if (detail) {
      this.notice(
        state === "error" ? "connection_error" : `connection_${state}`,
        detail,
        { hub_url: this.config.hubUrl },
        state === "error" ? "error" : state === "connected" ? "success" : "info",
      );
    }
    this.touch();
  }

  participantJoined(clientId: string): void {
    const role = this.roleFor(clientId);
    this.upsertParticipant(clientId, role, this.labelFor(clientId), true);
    this.setActivity(clientId, role === "agent" ? "已进入对话场景" : "在线");
    this.notice("participant_joined", `${this.labelFor(clientId)} 加入场景`, { client_id: clientId }, "success");
  }

  agentDiscovered(clientId: string): void {
    this.discoveredAgentIds.add(clientId);
    this.upsertParticipant(clientId, "agent", this.labelFor(clientId), true);
    this.setActivity(clientId, "已完成协议发现");

    if (
      clientId !== this.targetAgentId
      && !this.manualTargetSelected
      && !this.isOnline(this.targetAgentId)
    ) {
      this.switchTargetAgent(clientId, false);
    }

    this.notice("agent_discovered", `${this.labelFor(clientId)} 已识别为 Agent`, {
      client_id: clientId,
      active_target: this.targetAgentId,
    }, "success");
  }

  participantLeft(clientId: string, reason = "leave"): void {
    this.upsertParticipant(clientId, this.roleFor(clientId), this.labelFor(clientId), false);
    this.setOnline(clientId, false, "offline");
    this.setActivity(clientId, "已离开场景");
    this.notice("participant_left", `${this.labelFor(clientId)} 离开场景`, {
      client_id: clientId,
      reason,
    }, "warning");
  }

  selectAgent(clientId: string): { ok: true } | { ok: false; error: string } {
    const entity = this.entities.get(clientId);
    const identity = entity === undefined ? undefined : this.app.world.get(entity, Identity);
    if (!identity || identity.role !== "agent") {
      return { ok: false, error: `${clientId} 不是可对话的 Agent` };
    }
    this.manualTargetSelected = true;
    this.switchTargetAgent(clientId, true);
    return { ok: true };
  }

  humanMessage(content: string, messageId: string, recipient = this.targetAgentId): void {
    const session = this.sessionFor(recipient);
    session.lastHumanSentAt = this.now();
    session.updatedAt = this.now();
    session.messages.unshift({
      id: messageId,
      speakerId: this.config.human.humanId,
      speakerLabel: this.config.human.displayName,
      role: "human",
      content,
      at: this.now(),
      status: "sent",
    });
    session.messages.splice(MAX_MESSAGES);
    this.setActivity(this.config.human.humanId, "刚刚发出消息");
    this.setActivity(recipient, "正在理解消息…");
    this.pulse(this.config.human.humanId, recipient, "message", "user_message");
    this.notice("human_sent", "Human 消息已通过 Hub 发送", {
      message_id: messageId,
      recipient,
    }, "success");
  }

  assistantMessage(sender: string, content: string, metadata: Record<string, unknown>): void {
    const at = this.now();
    const session = this.sessionFor(sender);
    const responseConversationId = metadata.conversation_id;
    if (typeof responseConversationId === "string" && responseConversationId) {
      session.conversationId = responseConversationId;
    }
    session.latencyMs = session.lastHumanSentAt === null ? null : Math.max(0, at - session.lastHumanSentAt);
    session.updatedAt = at;
    session.messages.unshift({
      id: String(metadata.turn_id ?? metadata.message_id ?? `assistant-${at}`),
      speakerId: sender,
      speakerLabel: this.labelFor(sender),
      role: "agent",
      content,
      at,
      status: "received",
    });
    session.messages.splice(MAX_MESSAGES);
    this.setActivity(sender, "已回复，保持自主在线");
    this.setActivity(this.config.human.humanId, "收到 Agent 回复");
    this.pulse(sender, this.config.human.humanId, "message", "assistant.message");
    this.notice("assistant_message", "收到 Agent 回复", { sender, ...metadata }, "success");
  }

  environmentAction(sender: string, name: string, actionId: string, params: Record<string, unknown>): void {
    this.actionCount += 1;
    this.setActivity(sender, `正在执行 ${name}`);
    this.pulse(sender, this.config.scene.envId, "action", name);
    this.notice("environment_action", `${this.labelFor(sender)} 执行 ${name}`, {
      sender,
      action_name: name,
      action_id: actionId,
      params,
    });
  }

  environmentOutcome(sender: string, name: string, success: boolean, data: Record<string, unknown>): void {
    this.setActivity(sender, success ? `${name} 已完成` : `${name} 失败`);
    this.pulse(this.config.scene.envId, sender, "outcome", name);
    this.notice("environment_outcome", `环境${success ? "已返回" : "拒绝"} ${name}`, {
      recipient: sender,
      action_name: name,
      success,
      data,
    }, success ? "success" : "error");
  }

  protocol(kind: string, summary: string, payload: Record<string, unknown>, level: ProtocolNotice["level"] = "info"): void {
    this.notice(kind, summary, payload, level);
  }

  isOnline(clientId: string): boolean {
    const entity = this.entities.get(clientId);
    return entity === undefined ? false : Boolean(this.app.world.get(entity, Presence)?.online);
  }

  get agentId(): string {
    return this.targetAgentId;
  }

  get conversationId(): string {
    return this.sessionFor(this.targetAgentId).conversationId;
  }

  exportState(): DialoguePersistenceState {
    return {
      version: 1,
      activeAgentId: this.targetAgentId,
      manualTargetSelected: this.manualTargetSelected,
      sessions: [...this.sessions.values()].map((session) => ({
        ...session,
        messages: session.messages.map((message) => ({ ...message })),
      })),
    };
  }

  participantIds(): string[] {
    return [...this.entities.keys()].filter((id) => this.isOnline(id));
  }

  sceneContext(): Record<string, unknown> {
    return {
      environment_id: this.config.scene.envId,
      title: this.config.scene.title,
      background: this.config.scene.background,
      participants: this.participantIds().sort(),
      scene_type: "interactive_npc_dialogue_experiment",
    };
  }

  humanContext(): Record<string, unknown> {
    const human = this.config.human;
    return {
      human_id: human.humanId,
      display_name: human.displayName,
      role: human.role,
      background: human.background,
      relationship_to_agent: human.relationshipToAgent,
      authenticated: true,
      authentication_source: "web_ui_scenario",
      login_session: human.loginSession,
    };
  }

  snapshot(): DialogueSnapshot {
    const activeSession = this.sessionFor(this.targetAgentId);
    const participants = [...this.app.world.query(participantQuery)].map((row) => ({
      id: row.identity.id,
      label: row.identity.label,
      role: row.identity.role,
      online: row.presence.online,
      state: row.presence.state,
      activity: row.activity.label,
      x: Number(row.layout.x.toFixed(3)),
      y: Number(row.layout.y.toFixed(3)),
      updatedAt: row.presence.updatedAt,
    }));
    participants.sort((left, right) => roleOrder(left.role) - roleOrder(right.role) || left.id.localeCompare(right.id));
    return {
      revision: this.revision,
      serverTime: this.now(),
      connection: {
        state: this.connectionState,
        hubConnected: this.hubConnected,
        browserClients: this.browserClients,
        hubUrl: this.config.hubUrl,
      },
      scene: { ...this.config.scene },
      human: { ...this.config.human },
      agentId: this.targetAgentId,
      conversationId: activeSession.conversationId,
      sessions: [...this.sessions.values()]
        .map((session) => ({
          agentId: session.agentId,
          conversationId: session.conversationId,
          messageCount: session.messages.length,
          updatedAt: session.updatedAt,
        }))
        .sort((left, right) => right.updatedAt - left.updatedAt),
      participants,
      messages: activeSession.messages.map((message) => ({ ...message })),
      notices: this.notices.map((notice) => ({ ...notice, payload: { ...notice.payload } })),
      pulses: this.pulses.map((pulse) => ({ ...pulse })),
      metrics: {
        online: participants.filter((participant) => participant.online).length,
        messages: activeSession.messages.length,
        actions: this.actionCount,
        latencyMs: activeSession.latencyMs,
      },
    };
  }

  private upsertParticipant(id: string, role: ParticipantRole, label: string, online: boolean): void {
    const existing = this.entities.get(id);
    if (existing !== undefined) {
      const identity = this.app.world.get(existing, Identity);
      const presence = this.app.world.get(existing, Presence);
      if (identity) {
        identity.role = role;
        identity.label = label;
      }
      if (presence) {
        presence.online = online;
        presence.state = online ? defaultState(role) : "offline";
        presence.updatedAt = this.now();
      }
      this.reflowTargets();
      this.touch();
      return;
    }
    const position = positionFor(role, this.entities.size);
    const entity = this.app.world.spawn(
      component(Identity, { id, label, role }),
      component(Presence, {
        online,
        state: online ? defaultState(role) : "waiting",
        updatedAt: this.now(),
      }),
      component(Activity, { label: initialActivity(role, online), count: 0 }),
      component(Layout, { x: 0.5, y: 0.5, targetX: position.x, targetY: position.y }),
    );
    this.entities.set(id, entity);
    this.reflowTargets();
    this.touch();
  }

  private setOnline(id: string, online: boolean, state: string): void {
    const entity = this.entities.get(id);
    if (entity === undefined) return;
    const presence = this.app.world.get(entity, Presence);
    if (!presence) return;
    presence.online = online;
    presence.state = state;
    presence.updatedAt = this.now();
  }

  private setActivity(id: string, label: string): void {
    if (!this.entities.has(id)) this.upsertParticipant(id, this.roleFor(id), this.labelFor(id), true);
    const entity = this.entities.get(id);
    if (entity === undefined) return;
    const activity = this.app.world.get(entity, Activity);
    if (activity) {
      activity.label = label;
      activity.count += 1;
    }
    this.touch();
  }

  private notice(
    kind: string,
    summary: string,
    payload: Record<string, unknown>,
    level: ProtocolNotice["level"] = "info",
  ): void {
    const at = this.now();
    this.notices.unshift({ id: `${kind}-${at}-${this.notices.length}`, kind, summary, payload, level, at });
    this.notices.splice(MAX_NOTICES);
    this.touch();
  }

  private pulse(from: string, to: string, kind: ScenePulse["kind"], label: string): void {
    this.pulses.unshift({ id: `${kind}-${this.now()}-${this.pulses.length}`, from, to, kind, label, at: this.now() });
    this.pulses.splice(MAX_PULSES);
    this.touch();
  }

  private reflowTargets(): void {
    this.setLayoutTarget(this.config.scene.envId, positionFor("environment", 0));
    this.setLayoutTarget(this.config.human.humanId, positionFor("human", 0));
    this.setLayoutTarget(this.targetAgentId, positionFor("agent", 0));
    const extras = [...this.entities.keys()].filter((id) => ![
      this.config.scene.envId,
      this.config.human.humanId,
      this.targetAgentId,
    ].includes(id));
    extras.forEach((id, index) => {
      const entity = this.entities.get(id);
      const layout = entity === undefined ? undefined : this.app.world.get(entity, Layout);
      if (!layout) return;
      const angle = (index / Math.max(extras.length, 1)) * Math.PI * 2 - Math.PI / 2;
      layout.targetX = 0.5 + Math.cos(angle) * 0.39;
      layout.targetY = 0.5 + Math.sin(angle) * 0.39;
    });
  }

  private roleFor(id: string): ParticipantRole {
    if (id === this.config.scene.envId) return "environment";
    if (id === this.config.human.humanId) return "human";
    if (id === this.config.agentId || id === this.targetAgentId || this.discoveredAgentIds.has(id)) return "agent";
    return "client";
  }

  private switchTargetAgent(clientId: string, manual: boolean): void {
    const previousAgentId = this.targetAgentId;
    if (clientId === previousAgentId) return;
    this.targetAgentId = clientId;
    this.sessionFor(clientId);
    this.reflowTargets();
    this.setActivity(clientId, this.isOnline(clientId) ? "当前对话目标，可开始交流" : "已选择，等待上线");
    this.notice("agent_target_switched", `${manual ? "已选择" : "已自动选择"} ${this.labelFor(clientId)} 作为对话目标`, {
      previous_agent_id: previousAgentId,
      agent_id: clientId,
      conversation_id: this.conversationId,
      selection: manual ? "manual" : "automatic",
    }, "success");
  }

  private sessionFor(agentId: string): DialogueSession {
    const existing = this.sessions.get(agentId);
    if (existing) return existing;
    const session: DialogueSession = {
      agentId,
      conversationId: agentId === this.config.agentId
        ? this.config.conversationId
        : `${this.config.scene.envId}:${this.config.human.humanId}:${agentId}`,
      messages: [],
      lastHumanSentAt: null,
      latencyMs: null,
      updatedAt: this.now(),
    };
    this.sessions.set(agentId, session);
    return session;
  }

  private restoreSessions(restored?: DialoguePersistenceState): void {
    if (restored?.version !== 1 || !Array.isArray(restored.sessions)) {
      this.sessionFor(this.config.agentId);
      return;
    }
    for (const candidate of restored.sessions) {
      if (!candidate || typeof candidate.agentId !== "string" || typeof candidate.conversationId !== "string") continue;
      this.sessions.set(candidate.agentId, {
        agentId: candidate.agentId,
        conversationId: candidate.conversationId,
        messages: Array.isArray(candidate.messages)
          ? candidate.messages.slice(0, MAX_MESSAGES).map((message) => ({ ...message }))
          : [],
        lastHumanSentAt: typeof candidate.lastHumanSentAt === "number" ? candidate.lastHumanSentAt : null,
        latencyMs: typeof candidate.latencyMs === "number" ? candidate.latencyMs : null,
        updatedAt: typeof candidate.updatedAt === "number" ? candidate.updatedAt : this.now(),
      });
    }
    this.sessionFor(this.config.agentId);
    this.sessionFor(this.targetAgentId);
  }

  private setLayoutTarget(id: string, position: { x: number; y: number }): void {
    const entity = this.entities.get(id);
    const layout = entity === undefined ? undefined : this.app.world.get(entity, Layout);
    if (!layout) return;
    layout.targetX = position.x;
    layout.targetY = position.y;
  }

  private labelFor(id: string): string {
    if (id === this.config.scene.envId) return this.config.scene.title;
    if (id === this.config.human.humanId) return this.config.human.displayName;
    return id;
  }

  private touch(): void {
    this.revision += 1;
  }
}

function positionFor(role: ParticipantRole, index: number): { x: number; y: number } {
  if (role === "environment") return { x: 0.5, y: 0.48 };
  if (role === "human") return { x: 0.18, y: 0.72 };
  if (role === "agent") return { x: 0.82, y: 0.28 };
  const angle = index * 2.18;
  return { x: 0.5 + Math.cos(angle) * 0.39, y: 0.5 + Math.sin(angle) * 0.39 };
}

function defaultState(role: ParticipantRole): string {
  if (role === "environment") return "hosting";
  if (role === "human") return "joined";
  return "online";
}

function initialActivity(role: ParticipantRole, online: boolean): string {
  if (!online) return "等待加入场景";
  if (role === "environment") return "提供场景与工具";
  if (role === "human") return "已登录，等待 Agent";
  return "在线";
}

function roleOrder(role: ParticipantRole): number {
  return { environment: 0, human: 1, agent: 2, client: 3 }[role];
}
