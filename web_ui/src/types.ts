export type ParticipantRole = "environment" | "human" | "agent" | "client";

export interface HumanProfile {
  humanId: string;
  displayName: string;
  role: string;
  background: string;
  relationshipToAgent: string;
  loginSession: string;
}

export interface SceneConfig {
  envId: string;
  title: string;
  background: string;
}

export interface DialogueConfig {
  hubUrl: string;
  host: string;
  port: number;
  agentId: string;
  conversationId: string;
  scene: SceneConfig;
  human: HumanProfile;
  initialMessage?: string;
  autoReconnect: boolean;
  monitorable: boolean;
}

export interface ParticipantSnapshot {
  id: string;
  label: string;
  role: ParticipantRole;
  online: boolean;
  state: string;
  activity: string;
  x: number;
  y: number;
  updatedAt: number;
}

export interface DialogueMessage {
  id: string;
  speakerId: string;
  speakerLabel: string;
  role: "human" | "agent" | "system";
  content: string;
  at: number;
  status?: "sending" | "sent" | "received" | "failed";
}

export interface DialogueSession {
  agentId: string;
  conversationId: string;
  messages: DialogueMessage[];
  lastHumanSentAt: number | null;
  latencyMs: number | null;
  updatedAt: number;
}

export interface DialoguePersistenceState {
  version: 1;
  activeAgentId: string;
  manualTargetSelected: boolean;
  sessions: DialogueSession[];
}

export interface DialogueSessionSnapshot {
  agentId: string;
  conversationId: string;
  messageCount: number;
  updatedAt: number;
}

export interface ProtocolNotice {
  id: string;
  kind: string;
  summary: string;
  payload: Record<string, unknown>;
  level: "info" | "success" | "warning" | "error";
  at: number;
}

export interface ScenePulse {
  id: string;
  from: string;
  to: string;
  kind: "message" | "action" | "outcome" | "event";
  label: string;
  at: number;
}

export interface DialogueSnapshot {
  revision: number;
  serverTime: number;
  connection: {
    state: "connecting" | "connected" | "disconnected" | "error";
    hubConnected: boolean;
    browserClients: number;
    hubUrl: string;
  };
  scene: SceneConfig;
  human: HumanProfile;
  agentId: string;
  conversationId: string;
  sessions: DialogueSessionSnapshot[];
  participants: ParticipantSnapshot[];
  messages: DialogueMessage[];
  notices: ProtocolNotice[];
  pulses: ScenePulse[];
  metrics: {
    online: number;
    messages: number;
    actions: number;
    latencyMs: number | null;
  };
}
