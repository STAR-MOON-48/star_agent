import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { describe, it } from "node:test";

import { AgentClient, HubServer, type MessageActionContent } from "star-protocol";

import { DialogueWorld } from "../src/dialogue-world.js";
import { DialogueProtocol } from "../src/protocol.js";
import { listenOnAvailablePort } from "../src/port.js";
import type { DialogueConfig } from "../src/types.js";
import { MimoVoiceService, pcm16ToWav } from "../src/voice.js";

function config(hubUrl = "ws://127.0.0.1:8000"): DialogueConfig {
  return {
    hubUrl,
    host: "127.0.0.1",
    port: 4173,
    agentId: "test-agent",
    conversationId: "test-scene:test-human:test-agent",
    autoReconnect: false,
    monitorable: false,
    scene: {
      envId: "test-scene",
      title: "测试对话场景",
      background: "用于验证 Star World 场景状态。",
    },
    human: {
      humanId: "test-human",
      displayName: "测试用户",
      role: "场景观察者",
      background: "第一次进入测试场景。",
      relationshipToAgent: "初次见面",
      loginSession: "login-test",
    },
  };
}

class RecordingAgent extends AgentClient {
  readonly incomingActions: MessageActionContent[] = [];

  override async onAction(_sender: string, content: MessageActionContent): Promise<void> {
    this.incomingActions.push(content);
  }
}

describe("DialogueWorld", () => {
  it("models participants, messages and protocol pulses in Star World ECS", () => {
    let now = 1_000;
    const world = new DialogueWorld(config(), () => now);
    world.setConnection("connected", true);
    world.participantJoined("test-agent");
    world.humanMessage("你好", "message-1");
    now += 420;
    world.assistantMessage("test-agent", "你好，很高兴见到你。", { turn_id: "turn-1" });
    world.environmentAction("test-agent", "observe_social_scene", "action-1", {});
    world.environmentOutcome("test-agent", "observe_social_scene", true, { ok: true });
    world.tick(0.1);

    const snapshot = world.snapshot();
    assert.equal(snapshot.connection.hubConnected, true);
    assert.equal(snapshot.participants.find((item) => item.id === "test-agent")?.online, true);
    assert.equal(snapshot.messages.length, 2);
    assert.equal(snapshot.messages[0]?.content, "你好，很高兴见到你。");
    assert.equal(snapshot.metrics.actions, 1);
    assert.equal(snapshot.metrics.latencyMs, 420);
    assert.ok(snapshot.pulses.some((pulse) => pulse.kind === "action"));
    assert.ok(snapshot.notices.some((notice) => notice.kind === "environment_outcome"));
  });

  it("keeps speaker and scene context compatible with the Python console", () => {
    const world = new DialogueWorld(config());
    assert.deepEqual(world.humanContext(), {
      human_id: "test-human",
      display_name: "测试用户",
      role: "场景观察者",
      background: "第一次进入测试场景。",
      relationship_to_agent: "初次见面",
      authenticated: true,
      authentication_source: "web_ui_scenario",
      login_session: "login-test",
    });
    assert.equal(world.sceneContext().scene_type, "interactive_npc_dialogue_experiment");
  });

  it("automatically targets a discovered agent when the configured agent is offline", () => {
    const world = new DialogueWorld(config());
    world.setConnection("connected", true);
    world.participantJoined("test-agent-3");

    assert.equal(world.agentId, "test-agent");
    assert.equal(world.snapshot().participants.find((item) => item.id === "test-agent-3")?.role, "client");

    world.agentDiscovered("test-agent-3");

    const snapshot = world.snapshot();
    assert.equal(world.agentId, "test-agent-3");
    assert.equal(world.conversationId, "test-scene:test-human:test-agent-3");
    assert.equal(snapshot.agentId, "test-agent-3");
    assert.equal(snapshot.participants.find((item) => item.id === "test-agent-3")?.role, "agent");
    assert.equal(snapshot.participants.find((item) => item.id === "test-agent-3")?.online, true);
    assert.ok(snapshot.notices.some((notice) => notice.kind === "agent_target_switched"));
  });

  it("keeps an online configured agent as the active target", () => {
    const world = new DialogueWorld(config());
    world.participantJoined("test-agent");
    world.agentDiscovered("another-agent");

    assert.equal(world.agentId, "test-agent");
    assert.equal(world.snapshot().participants.find((item) => item.id === "another-agent")?.role, "agent");
  });

  it("switches between agent sessions and restores their saved context", () => {
    let now = 10_000;
    const world = new DialogueWorld(config(), () => now);
    world.agentDiscovered("agent-a");
    world.humanMessage("A，你好", "human-a");
    now += 250;
    world.assistantMessage("agent-a", "你好，我是 A", {
      turn_id: "turn-a",
      conversation_id: "test-scene:test-human:agent-a",
    });
    world.agentDiscovered("agent-b");

    assert.equal(world.selectAgent("agent-b").ok, true);
    assert.equal(world.snapshot().messages.length, 0);
    world.humanMessage("B，你好", "human-b");
    assert.equal(world.snapshot().messages[0]?.content, "B，你好");

    assert.equal(world.selectAgent("agent-a").ok, true);
    assert.deepEqual(world.snapshot().messages.map((message) => message.content), [
      "你好，我是 A",
      "A，你好",
    ]);

    const restored = new DialogueWorld(config(), () => now, world.exportState());
    assert.equal(restored.agentId, "agent-a");
    assert.equal(restored.conversationId, "test-scene:test-human:agent-a");
    assert.deepEqual(restored.snapshot().messages.map((message) => message.content), [
      "你好，我是 A",
      "A，你好",
    ]);
    assert.equal(restored.snapshot().sessions.length, 3);
  });
});

describe("DialogueProtocol", () => {
  it("hosts a real Star Protocol environment and serves discoverable tools", async () => {
    const hub = new HubServer({ port: 0 });
    const hubUrl = (await hub.start()).websocketUrl;
    const world = new DialogueWorld(config(hubUrl));
    const protocol = new DialogueProtocol(world);
    const agent = new RecordingAgent("test-agent-3", { autoReconnect: false });
    try {
      await protocol.connect();
      await agent.connect(hubUrl);
      await agent.joinWithRetry("test-scene", 2_000, 25);
      const tools = await agent.waitForTools(2_000);
      assert.equal(world.agentId, "test-agent-3");
      assert.deepEqual(tools.map((tool) => tool.name), [
        "observe_social_scene",
        "read_human_profile",
        "perform_social_action",
      ]);
      const result = await agent.requestAction("test-scene", "observe_social_scene", {}, { timeoutMs: 2_000 });
      assert.equal(result.success, true);
      assert.equal((result.data?.scene as Record<string, unknown>)?.environment_id, "test-scene");
      assert.ok(world.snapshot().metrics.actions >= 1);

      const sent = await protocol.sendHumanMessage("不用一问一答，你可以继续说");
      assert.equal(sent.ok, true);
      await waitFor(() => agent.incomingActions.length === 1);
      const userMessage = agent.incomingActions[0];
      assert.equal(userMessage?.name, "user_message");
      assert.match(String(userMessage?.params?.message_id), /^human-\d+-[a-f0-9]{8}$/);
      assert.equal(userMessage?.params?.conversation_id, "test-scene:test-human:test-agent-3");
    } finally {
      await agent.stop();
      await protocol.disconnect();
      await hub.stop();
    }
  });
});

describe("listenOnAvailablePort", () => {
  it("automatically advances when the preferred port is occupied", async () => {
    const occupied = createServer();
    const candidate = createServer();
    try {
      await new Promise<void>((resolve, reject) => {
        occupied.once("error", reject);
        occupied.listen(0, "127.0.0.1", resolve);
      });
      const preferredPort = (occupied.address() as AddressInfo).port;
      const selectedPort = await listenOnAvailablePort(candidate, {
        host: "127.0.0.1",
        preferredPort,
        maxAttempts: 20,
      });
      assert.ok(selectedPort > preferredPort);
      assert.ok(selectedPort <= preferredPort + 19);
    } finally {
      await Promise.all([closeServer(candidate), closeServer(occupied)]);
    }
  });
});

describe("MimoVoiceService", () => {
  it("wraps browser PCM as WAV and sends it to mimo-v2.5-asr", async () => {
    let requestBody: Record<string, unknown> | undefined;
    const service = new MimoVoiceService(
      { apiKey: "test-key", language: "zh" },
      async (_input, init) => {
        requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return new Response(JSON.stringify({
          choices: [{ message: { content: "你好，Ling" } }],
        }), { status: 200, headers: { "content-type": "application/json" } });
      },
    );
    const pcm = new Uint8Array(16_000);
    const transcript = await service.transcribePcm(pcm, 16_000);

    assert.equal(transcript, "你好，Ling");
    assert.equal(requestBody?.model, "mimo-v2.5-asr");
    assert.deepEqual(requestBody?.asr_options, { language: "zh" });
    const messages = requestBody?.messages as Array<Record<string, unknown>>;
    const parts = messages[0]?.content as Array<Record<string, unknown>>;
    const audio = parts[0]?.input_audio as Record<string, unknown>;
    assert.match(String(audio.data), /^data:audio\/wav;base64,/);
    const wav = Buffer.from(String(audio.data).split(",")[1] ?? "", "base64");
    assert.equal(wav.subarray(0, 4).toString("ascii"), "RIFF");
    assert.equal(wav.readUInt32LE(24), 16_000);
  });

  it("forwards MiMo streaming TTS PCM chunks without exposing the key", async () => {
    const first = Buffer.from([1, 2, 3, 4]).toString("base64");
    const second = Buffer.from([5, 6]).toString("base64");
    let authorization = "";
    const service = new MimoVoiceService(
      { apiKey: "secret-key", voice: "冰糖" },
      async (_input, init) => {
        authorization = String((init?.headers as Record<string, string>)?.authorization ?? "");
        const body = [
          `data: ${JSON.stringify({ choices: [{ delta: { audio: { data: first } } }] })}`,
          `data: ${JSON.stringify({ choices: [{ delta: { audio: { data: second } } }] })}`,
          "data: [DONE]",
          "",
        ].join("\n\n");
        return new Response(body, {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      },
    );
    const chunks: string[] = [];
    const bytes = await service.streamSpeech("你好", (chunk) => {
      chunks.push(chunk);
    });

    assert.equal(authorization, "Bearer secret-key");
    assert.deepEqual(chunks, [first, second]);
    assert.equal(bytes, 6);
  });

  it("reports voice as unavailable when no server-side API key exists", () => {
    const service = new MimoVoiceService({ apiKey: "" });
    assert.equal(service.capabilities().available, false);
    assert.match(service.capabilities().reason ?? "", /API_KEY/);
    assert.equal(Buffer.from(pcm16ToWav(new Uint8Array(2), 16_000)).subarray(8, 12).toString(), "WAVE");
  });
});

function closeServer(server: ReturnType<typeof createServer>): Promise<void> {
  if (!server.listening) return Promise.resolve();
  return new Promise((resolve, reject) => {
    server.close((error) => error ? reject(error) : resolve());
  });
}

async function waitFor(predicate: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.fail(`Condition was not met within ${timeoutMs}ms`);
}
