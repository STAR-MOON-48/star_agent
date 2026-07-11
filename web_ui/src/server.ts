import { randomUUID } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { mkdir, rename, stat, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { dirname, extname, resolve } from "node:path";

import WebSocket, { WebSocketServer } from "ws";

import { DialogueWorld } from "./dialogue-world.js";
import { listenOnAvailablePort } from "./port.js";
import { DialogueProtocol } from "./protocol.js";
import type { DialogueConfig, DialoguePersistenceState } from "./types.js";
import { MimoVoiceService } from "./voice.js";

const cliArguments = process.argv.slice(2);
if (cliArguments.includes("--help") || cliArguments.includes("-h")) {
  printHelp();
  process.exit(0);
}
const config = readConfig(cliArguments);
const publicDirectory = resolve(process.env.PUBLIC_DIR ?? "public");
const defaultSessionDirectory = resolve(publicDirectory, "..", "..", ".agent_state_web_ui");
const sessionFile = resolve(process.env.LING_WEB_SESSION_FILE ?? resolve(
  defaultSessionDirectory,
  `${safeFileSegment(config.scene.envId)}--${safeFileSegment(config.human.humanId)}.json`,
));
const voiceService = new MimoVoiceService({
  apiKey: process.env.MIMO_API_KEY ?? process.env.XIAOMI_API_KEY ?? "",
  baseUrl: argumentValue(cliArguments, "mimo-base-url")
    ?? process.env.MIMO_BASE_URL
    ?? process.env.XIAOMI_BASE_URL,
  voice: argumentValue(cliArguments, "voice") ?? process.env.MIMO_TTS_VOICE,
  style: argumentValue(cliArguments, "voice-style") ?? process.env.MIMO_TTS_STYLE,
  language: voiceLanguage(
    argumentValue(cliArguments, "voice-language") ?? process.env.MIMO_ASR_LANGUAGE,
  ),
});
const world = new DialogueWorld(config, Date.now, loadDialogueState(sessionFile));
const browserClients = new Set<WebSocket>();
const voiceClients = new Set<WebSocket>();
const voiceInputBusy = new Set<WebSocket>();
let speechQueue = Promise.resolve();
const protocol = new DialogueProtocol(world, {
  onAssistantMessage(sender, content) {
    if (sender === world.agentId) enqueueAssistantSpeech(sender, content);
  },
});
let shuttingDown = false;
let lastRevision = -1;
let lastTick = Date.now();
let persistenceTimer: NodeJS.Timeout | null = null;
let persistenceWrite = Promise.resolve();

const server = createServer((request, response) => void handleHttp(request, response));
const websocketServer = new WebSocketServer({ noServer: true });

server.on("upgrade", (request, socket, head) => {
  const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
  if (pathname !== "/live") {
    socket.destroy();
    return;
  }
  websocketServer.handleUpgrade(request, socket, head, (websocket) => {
    websocketServer.emit("connection", websocket, request);
  });
});

websocketServer.on("connection", (websocket) => {
  browserClients.add(websocket);
  world.setBrowserClients(browserClients.size);
  sendSnapshot(websocket);
  sendVoice(websocket, {
    type: "voice_capabilities",
    data: voiceService.capabilities(),
  });
  websocket.on("message", (raw) => void handleBrowserMessage(websocket, raw.toString()));
  websocket.on("close", () => removeBrowserClient(websocket));
  websocket.on("error", () => removeBrowserClient(websocket));
});

const requestedPort = config.port;
config.port = await listenOnAvailablePort(server, {
  host: config.host,
  preferredPort: requestedPort,
});

console.log(`\n  Ling 对话场景 Web UI 已启动`);
if (config.port !== requestedPort) {
  console.log(`  Port:  ${requestedPort} 已占用，自动顺延到 ${config.port}`);
}
console.log(`  Web:   http://${config.host}:${config.port}`);
console.log(`  Env:   ${config.scene.envId}`);
console.log(`  Human: ${config.human.humanId}`);
console.log(`  Agent: ${config.agentId}`);
console.log(`  Hub:   ${config.hubUrl}\n`);
console.log(
  voiceService.available
    ? `  Voice: MiMo ASR + TTS (${voiceService.voice}) ready`
    : `  Voice: disabled — set MIMO_API_KEY or XIAOMI_API_KEY`,
);
console.log(`  Agent 启动命令:`);
console.log(`  uv run agent-ling-star --agent-id ${config.agentId} --hub-url ${config.hubUrl} --env-id ${config.scene.envId} --no-startup-objective\n`);

const frameTimer = setInterval(() => {
  const now = Date.now();
  world.tick((now - lastTick) / 1_000);
  lastTick = now;
  const snapshot = world.snapshot();
  const revisionChanged = snapshot.revision !== lastRevision;
  if (revisionChanged || snapshot.pulses.length > 0) {
    lastRevision = snapshot.revision;
    broadcastSnapshot(snapshot);
    if (revisionChanged) scheduleSessionSave();
  }
}, 100);
frameTimer.unref();

const connectionTimer = setInterval(() => {
  if (shuttingDown) return;
  const connection = world.snapshot().connection;
  if (!connection.hubConnected && connection.state !== "connecting") void protocol.connect();
  void protocol.sendInitialMessageIfReady();
}, 2_000);
connectionTimer.unref();

void protocol.connect();
process.once("SIGINT", () => void shutdown());
process.once("SIGTERM", () => void shutdown());

async function handleHttp(request: IncomingMessage, response: ServerResponse): Promise<void> {
  try {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (request.method !== "GET") return sendJson(response, 405, { error: "Method not allowed" });
    if (url.pathname === "/api/snapshot") return sendJson(response, 200, world.snapshot());
    if (url.pathname === "/health") {
      const snapshot = world.snapshot();
      return sendJson(response, 200, {
        ok: true,
        hub_connected: snapshot.connection.hubConnected,
        voice_available: voiceService.available,
        environment_id: snapshot.scene.envId,
        revision: snapshot.revision,
      });
    }
    const filePath = staticPath(url.pathname);
    if (!filePath) return sendJson(response, 404, { error: "Not found" });
    const info = await stat(filePath);
    if (!info.isFile()) throw Object.assign(new Error("Not found"), { code: "ENOENT" });
    const extension = extname(filePath);
    response.writeHead(200, {
      "content-type": contentType(filePath),
      "cache-control": [".html", ".css", ".js"].includes(extension) ? "no-cache" : "public, max-age=3600",
      "x-content-type-options": "nosniff",
    });
    createReadStream(filePath).pipe(response);
  } catch (error) {
    if (isMissingFile(error)) sendJson(response, 404, { error: "Not found" });
    else {
      console.error(error);
      sendJson(response, 500, { error: "Internal server error" });
    }
  }
}

async function handleBrowserMessage(websocket: WebSocket, raw: string): Promise<void> {
  let command: Record<string, unknown>;
  try {
    command = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return sendCommandResult(websocket, "invalid", false, "无效的 JSON 消息");
  }
  const type = String(command.type ?? "");
  if (type === "send_message") {
    const result = await protocol.sendHumanMessage(String(command.content ?? ""));
    sendCommandResult(websocket, type, result.ok, result.ok ? undefined : result.error, result.ok ? { message_id: result.messageId } : undefined);
    broadcastSnapshot();
    return;
  }
  if (type === "select_agent") {
    const result = world.selectAgent(String(command.agent_id ?? ""));
    sendCommandResult(websocket, type, result.ok, result.ok ? undefined : result.error, result.ok ? {
      agent_id: world.agentId,
      conversation_id: world.conversationId,
    } : undefined);
    broadcastSnapshot();
    scheduleSessionSave();
    return;
  }
  if (type === "reconnect") {
    sendCommandResult(websocket, type, true);
    await protocol.reconnect();
    broadcastSnapshot();
    return;
  }
  if (type === "voice_live_start") {
    if (!voiceService.available) {
      sendCommandResult(websocket, type, false, voiceService.capabilities().reason);
      return;
    }
    voiceClients.add(websocket);
    sendCommandResult(websocket, type, true);
    sendVoiceState(websocket, "listening", "正在听你说话");
    return;
  }
  if (type === "voice_live_stop") {
    voiceClients.delete(websocket);
    voiceInputBusy.delete(websocket);
    sendCommandResult(websocket, type, true);
    sendVoiceState(websocket, "off", "Live 已关闭");
    return;
  }
  if (type === "voice_utterance") {
    void handleVoiceUtterance(websocket, command);
    return;
  }
  if (type === "ping") {
    websocket.send(JSON.stringify({ type: "pong", at: Date.now() }));
    return;
  }
  sendCommandResult(websocket, type || "unknown", false, "未知命令");
}

function sendCommandResult(
  websocket: WebSocket,
  command: string,
  ok: boolean,
  error?: string,
  data?: Record<string, unknown>,
): void {
  if (websocket.readyState !== WebSocket.OPEN) return;
  websocket.send(JSON.stringify({ type: "command_result", command, ok, error, data }));
}

function sendSnapshot(websocket: WebSocket, snapshot = world.snapshot()): void {
  if (websocket.readyState !== WebSocket.OPEN) return;
  websocket.send(JSON.stringify({ type: "snapshot", data: snapshot }));
}

function broadcastSnapshot(snapshot = world.snapshot()): void {
  for (const client of browserClients) sendSnapshot(client, snapshot);
}

function removeBrowserClient(websocket: WebSocket): void {
  browserClients.delete(websocket);
  voiceClients.delete(websocket);
  voiceInputBusy.delete(websocket);
  world.setBrowserClients(browserClients.size);
}

async function handleVoiceUtterance(
  websocket: WebSocket,
  command: Record<string, unknown>,
): Promise<void> {
  if (!voiceClients.has(websocket) || voiceInputBusy.has(websocket)) return;
  const encoded = typeof command.audio_base64 === "string" ? command.audio_base64 : "";
  if (!encoded || encoded.length > 3_000_000) {
    sendVoiceError(websocket, "录音数据为空或过大");
    return;
  }
  const pcm = Buffer.from(encoded, "base64");
  const sampleRate = Number(command.sample_rate ?? voiceService.inputSampleRate);
  if (!Number.isFinite(sampleRate) || sampleRate < 8_000 || sampleRate > 48_000) {
    sendVoiceError(websocket, "录音采样率无效");
    return;
  }
  voiceInputBusy.add(websocket);
  sendVoiceState(websocket, "transcribing", "小米 MiMo 正在识别…");
  try {
    const transcript = await voiceService.transcribePcm(pcm, sampleRate);
    if (!voiceClients.has(websocket)) return;
    sendVoice(websocket, {
      type: "voice_transcript",
      text: transcript,
      at: Date.now(),
    });
    sendVoiceState(websocket, "waiting_agent", "已发送给 Agent，等待回复…");
    const result = await protocol.sendHumanMessage(transcript);
    if (!result.ok) throw new Error(result.error);
  } catch (error) {
    if (voiceClients.has(websocket)) {
      sendVoiceError(
        websocket,
        error instanceof Error ? error.message : String(error),
      );
      sendVoiceState(websocket, "listening", "本轮失败，请继续说");
    }
  } finally {
    voiceInputBusy.delete(websocket);
  }
}

function enqueueAssistantSpeech(sender: string, content: string): void {
  if (!content.trim() || voiceClients.size === 0 || !voiceService.available) return;
  speechQueue = speechQueue
    .then(async () => {
      if (voiceClients.size === 0 || sender !== world.agentId) return;
      broadcastVoice({
        type: "voice_audio_start",
        sample_rate: voiceService.outputSampleRate,
        text: content,
      });
      broadcastVoiceState("speaking", "Agent 正在说话…");
      const bytes = await voiceService.streamSpeech(content, (audio) => {
        broadcastVoice({
          type: "voice_audio_chunk",
          audio_base64: audio,
          sample_rate: voiceService.outputSampleRate,
        });
      });
      broadcastVoice({ type: "voice_audio_end", bytes, at: Date.now() });
    })
    .catch((error) => {
      broadcastVoice({
        type: "voice_error",
        error: error instanceof Error ? error.message : String(error),
      });
      broadcastVoiceState("listening", "语音输出失败，继续监听");
    });
}

function sendVoiceState(websocket: WebSocket, phase: string, label: string): void {
  sendVoice(websocket, { type: "voice_state", phase, label, at: Date.now() });
}

function broadcastVoiceState(phase: string, label: string): void {
  broadcastVoice({ type: "voice_state", phase, label, at: Date.now() });
}

function sendVoiceError(websocket: WebSocket, error: string): void {
  sendVoice(websocket, { type: "voice_error", error, at: Date.now() });
}

function sendVoice(websocket: WebSocket, value: Record<string, unknown>): void {
  if (websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify(value));
  }
}

function broadcastVoice(value: Record<string, unknown>): void {
  for (const client of voiceClients) sendVoice(client, value);
}

function staticPath(pathname: string): string | null {
  const relative = pathname === "/" ? "index.html" : decodeURIComponent(pathname.slice(1));
  const candidate = resolve(publicDirectory, relative);
  return candidate === publicDirectory || candidate.startsWith(`${publicDirectory}/`) ? candidate : null;
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(value));
}

function contentType(path: string): string {
  return ({
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
  } as Record<string, string>)[extname(path)] ?? "application/octet-stream";
}

function isMissingFile(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && (error as NodeJS.ErrnoException).code === "ENOENT";
}

async function shutdown(): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  clearInterval(frameTimer);
  clearInterval(connectionTimer);
  if (persistenceTimer) clearTimeout(persistenceTimer);
  await persistDialogueState();
  for (const client of browserClients) client.close(1001, "Server shutting down");
  await protocol.disconnect();
  await new Promise<void>((accept) => server.close(() => accept()));
  process.exitCode = 0;
}

function loadDialogueState(path: string): DialoguePersistenceState | undefined {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as DialoguePersistenceState;
  } catch (error) {
    if (isMissingFile(error)) return undefined;
    console.warn(`  Sessions: 无法读取 ${path}，将使用新会话`, error);
    return undefined;
  }
}

function scheduleSessionSave(): void {
  if (persistenceTimer) clearTimeout(persistenceTimer);
  persistenceTimer = setTimeout(() => {
    persistenceTimer = null;
    void persistDialogueState();
  }, 250);
  persistenceTimer.unref();
}

async function persistDialogueState(): Promise<void> {
  const value = JSON.stringify(world.exportState(), null, 2);
  persistenceWrite = persistenceWrite.then(async () => {
    await mkdir(dirname(sessionFile), { recursive: true });
    const temporary = `${sessionFile}.tmp`;
    await writeFile(temporary, value, "utf8");
    await rename(temporary, sessionFile);
  }).catch((error) => {
    console.warn(`  Sessions: 无法保存 ${sessionFile}`, error);
  });
  await persistenceWrite;
}

function readConfig(argv: string[]): DialogueConfig {
  const args = new Map<string, string | true>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key?.startsWith("--")) continue;
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      args.set(key.slice(2), next);
      index += 1;
    } else args.set(key.slice(2), true);
  }
  const envId = stringArg(args, "env-id", process.env.STAR_ENV_ID ?? "npc-dialogue-lab");
  const humanId = stringArg(args, "human-id", process.env.HUMAN_ID ?? "human_web");
  const agentId = stringArg(args, "agent-id", process.env.AGENT_ID ?? "npc_agent");
  const port = Number(stringArg(args, "port", process.env.PORT ?? "4173"));
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error("--port 必须是有效端口");
  return {
    host: stringArg(args, "host", process.env.HOST ?? "127.0.0.1"),
    port,
    hubUrl: stringArg(args, "hub-url", process.env.STAR_HUB_URL ?? "ws://127.0.0.1:8000"),
    agentId,
    conversationId: stringArg(args, "conversation-id", `${envId}:${humanId}:${agentId}`),
    scene: {
      envId,
      title: stringArg(args, "scene-title", "NPC 对话实验室"),
      background: stringArg(args, "scene-background", "傍晚的安静会客室，Human 与自主 NPC 可以持续交谈和观察彼此反应。"),
    },
    human: {
      humanId,
      displayName: stringArg(args, "human-name", "林舟"),
      role: stringArg(args, "human-role", "进入实验场景的访客"),
      background: stringArg(args, "human-background", "刚结束一天的工作，第一次来到这里，希望认识负责接待的 NPC。"),
      relationshipToAgent: stringArg(args, "relationship", "与 Agent 初次见面"),
      loginSession: stringArg(args, "login-session", `login-${randomUUID().slice(0, 12)}`),
    },
    initialMessage: optionalStringArg(args, "initial-message"),
    autoReconnect: !args.has("no-auto-reconnect"),
    monitorable: args.has("monitorable"),
  };
}

function stringArg(args: Map<string, string | true>, key: string, fallback: string): string {
  const value = args.get(key);
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function optionalStringArg(args: Map<string, string | true>, key: string): string | undefined {
  const value = args.get(key);
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function argumentValue(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(`--${name}`);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith("--") ? value : undefined;
}

function voiceLanguage(value: string | undefined): "auto" | "zh" | "en" {
  return value === "zh" || value === "en" ? value : "auto";
}

function safeFileSegment(value: string): string {
  return value.replace(/[^a-zA-Z0-9_.-]+/g, "_").slice(0, 96) || "default";
}

function printHelp(): void {
  console.log(`Agent Ling Web UI\n\nUsage:\n  npm start -- [options]\n  uv run agent-ling-console [options]\n\nOptions:\n  --hub-url URL               Star Protocol Hub (default: ws://127.0.0.1:8000)\n  --host HOST                 Web bind host (default: 127.0.0.1)\n  --port PORT                 Web port (default: 4173)\n  --env-id ID                 Environment id\n  --agent-id ID               Target Agent id\n  --human-id ID               Logged-in Human id\n  --human-name NAME           Human display name\n  --human-role ROLE           Human role in this scene\n  --human-background TEXT     Human background\n  --relationship TEXT         Human relationship to Agent\n  --scene-title TITLE         Scene title\n  --scene-background TEXT     Scene description\n  --conversation-id ID        Stable conversation id\n  --initial-message TEXT      Send after Agent joins\n  --voice NAME                MiMo TTS voice (default: 冰糖)\n  --voice-language LANG       MiMo ASR language: auto, zh, en\n  --voice-style TEXT          MiMo TTS speaking style\n  --mimo-base-url URL         MiMo API base URL\n  --monitorable               Enable Star monitoring\n  --no-auto-reconnect         Disable SDK reconnect\n  -h, --help                  Show this help`);
}
