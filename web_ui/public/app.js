(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    snapshot: null,
    socket: null,
    reconnectAttempt: 0,
    reconnectTimer: null,
    filter: "all",
    toastTimer: null,
    voice: {
      available: false,
      capabilities: null,
      enabled: false,
      phase: "off",
      mediaStream: null,
      audioContext: null,
      worklet: null,
      silentGain: null,
      capture: null,
      playbackNextTime: 0,
      playbackSources: new Set(),
      resumeTimer: null,
    },
  };

  const roleMeta = {
    environment: { label: "ENV", icon: "i-star" },
    human: { label: "HUMAN", icon: "i-user" },
    agent: { label: "AGENT", icon: "i-cpu" },
    client: { label: "CLIENT", icon: "i-link" },
  };

  function connect() {
    clearTimeout(state.reconnectTimer);
    if (state.socket) state.socket.close();
    setBrowserConnection("connecting");
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/live`);
    state.socket = socket;
    socket.addEventListener("open", () => {
      state.reconnectAttempt = 0;
      setBrowserConnection("open");
      if (state.voice.enabled) {
        socket.send(JSON.stringify({ type: "voice_live_start" }));
        setVoicePhase("starting", "正在恢复 Live…");
      }
    });
    socket.addEventListener("message", (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === "snapshot") {
        state.snapshot = message.data;
        render();
      } else if (message.type?.startsWith("voice_")) {
        handleVoiceMessage(message);
      } else if (message.type === "command_result") {
        if (!message.ok) {
          toast(message.error || "命令执行失败", "error");
          if (message.command === "voice_live_start") void stopVoiceLive(false);
        } else if (message.command === "select_agent") {
          toast(`已切换到 ${message.data?.agent_id || "Agent"}，会话上下文已恢复`);
        }
      }
    });
    socket.addEventListener("close", () => {
      if (state.socket !== socket) return;
      setBrowserConnection("closed");
      const wait = Math.min(10_000, 800 * 2 ** state.reconnectAttempt++);
      state.reconnectTimer = setTimeout(connect, wait);
    });
    socket.addEventListener("error", () => socket.close());
  }

  function render() {
    const snapshot = state.snapshot;
    if (!snapshot) return;
    const participants = Array.isArray(snapshot.participants) ? snapshot.participants : [];
    const agent = participants.find((item) => item.id === snapshot.agentId);
    const agentOnline = Boolean(agent?.online && snapshot.connection?.hubConnected);

    setText("scene-title", snapshot.scene?.title);
    setText("scene-id", `ENV / ${snapshot.scene?.envId || "—"}`);
    setText("scene-background", snapshot.scene?.background);
    setText("hub-url", snapshot.connection?.hubUrl);
    setText("conversation-id", snapshot.conversationId);
    setText("human-name", snapshot.human?.displayName);
    setText("human-id", snapshot.human?.humanId);
    setText("human-role", snapshot.human?.role);
    setText("human-background", snapshot.human?.background);
    setText("human-relationship", snapshot.human?.relationshipToAgent);
    setText("human-monogram", monogram(snapshot.human?.displayName));
    setText("participant-count", participants.length);
    setText("metric-online", pad(snapshot.metrics?.online));
    setText("metric-messages", pad(snapshot.metrics?.messages));
    setText("metric-actions", pad(snapshot.metrics?.actions));
    setText("metric-latency", snapshot.metrics?.latencyMs == null ? "—" : formatLatency(snapshot.metrics.latencyMs));
    setText("agent-availability", agentOnline ? `${agent.label} 在线` : agent ? `${agent.label} 离线` : "等待 Agent");
    setText("browser-count", `${snapshot.connection?.browserClients || 0} 个观察端`);
    setText("revision-label", `SNAPSHOT / ${String(snapshot.revision || 0).padStart(6, "0")}`);
    setText("agent-activity", agent ? `${agent.label} · ${agent.activity}` : "等待 Agent 加入");
    setText("conversation-agent-label", agent ? `HUMAN ↔ ${agent.label}` : "HUMAN ↔ NPC AGENT");

    renderHubConnection(snapshot.connection);
    renderAgentSelector(participants, snapshot);
    renderParticipants(participants, snapshot);
    renderScene(participants, snapshot.pulses || [], snapshot);
    renderMessages(snapshot.messages || [], agent);
    renderNotices(snapshot.notices || []);
    setComposer(agentOnline, agent);
    renderVoiceButton(agentOnline);
  }

  function renderHubConnection(connection) {
    const pill = $("connection-pill");
    pill.className = "connection-pill";
    if (connection?.state === "connected") {
      pill.classList.add("is-online");
      setText("connection-label", "STAR HUB ONLINE");
    } else if (connection?.state === "connecting") {
      pill.classList.add("is-connecting");
      setText("connection-label", "CONNECTING HUB");
    } else {
      pill.classList.add("is-offline");
      setText("connection-label", connection?.state === "error" ? "HUB ERROR" : "HUB OFFLINE");
    }
  }

  function renderAgentSelector(participants, snapshot) {
    const selector = $("agent-selector");
    const agents = participants.filter((participant) => participant.role === "agent");
    const signature = agents.map((agent) => `${agent.id}:${agent.online}`).join("|");
    if (selector.dataset.signature !== signature) {
      selector.replaceChildren();
      agents.forEach((agent) => {
        const option = make("option", "", `${agent.label} · ${agent.online ? "在线" : "离线"}`);
        option.value = agent.id;
        selector.append(option);
      });
      selector.dataset.signature = signature;
    }
    selector.disabled = agents.length === 0;
    if (agents.some((agent) => agent.id === snapshot.agentId)) selector.value = snapshot.agentId;
  }

  function renderParticipants(participants, snapshot) {
    const list = $("participant-list");
    list.replaceChildren();
    const sessions = new Map((snapshot.sessions || []).map((session) => [session.agentId, session]));
    participants.forEach((participant) => {
      const selectable = participant.role === "agent";
      const active = participant.id === snapshot.agentId;
      const row = make(selectable ? "button" : "article", `participant-row role-${participant.role}${participant.online ? " is-online" : ""}${active ? " is-active" : ""}`);
      if (selectable) {
        row.type = "button";
        row.dataset.agentId = participant.id;
        row.setAttribute("aria-pressed", String(active));
        row.title = active ? "当前对话 Agent" : `切换到 ${participant.label} 的会话`;
      }
      const mark = make("span", "participant-mark");
      mark.append(icon(roleMeta[participant.role]?.icon || "i-link"));
      const copy = make("div", "participant-copy");
      copy.append(make("strong", "", participant.label), make("small", "", participant.id));
      const count = sessions.get(participant.id)?.messageCount || 0;
      const statusText = active
        ? `ACTIVE${count ? ` · ${count}` : ""}`
        : `${participant.online ? participant.state : "offline"}${count ? ` · ${count}` : ""}`;
      const status = make("span", "participant-status", statusText);
      row.append(mark, copy, status);
      list.append(row);
    });
  }

  function renderScene(participants, pulses, snapshot) {
    const layer = $("participant-layer");
    const svg = $("connection-layer");
    layer.replaceChildren();
    svg.replaceChildren();

    const environment = participants.find((item) => item.role === "environment");
    participants.forEach((participant) => {
      const selectable = participant.role === "agent";
      const active = participant.id === snapshot.agentId;
      const node = make(selectable ? "button" : "article", `scene-node role-${participant.role}${participant.online ? " is-online" : " is-offline"}${active ? " is-active" : ""}`);
      if (selectable) {
        node.type = "button";
        node.dataset.agentId = participant.id;
        node.setAttribute("aria-pressed", String(active));
        node.title = active ? "当前对话 Agent" : `激活 ${participant.label} 的会话`;
      }
      node.style.left = `${participant.x * 100}%`;
      node.style.top = `${participant.y * 100}%`;
      node.dataset.id = participant.id;
      const orb = make("span", "node-orb");
      orb.append(icon(roleMeta[participant.role]?.icon || "i-link"));
      const copy = make("div", "node-copy");
      copy.append(make("span", "node-role", roleMeta[participant.role]?.label || "CLIENT"));
      copy.append(make("strong", "", participant.label));
      copy.append(make("small", "", participant.activity));
      node.append(orb, copy);
      layer.append(node);

      if (environment && participant.id !== environment.id) {
        svg.append(connectionLine(environment, participant, participant.online ? "active" : "muted"));
      }
    });

    pulses.slice(0, 12).reverse().forEach((pulse, index) => {
      const from = participants.find((item) => item.id === pulse.from);
      const to = participants.find((item) => item.id === pulse.to);
      if (!from || !to) return;
      const line = connectionLine(from, to, `pulse kind-${pulse.kind}`);
      line.style.animationDelay = `${index * -0.18}s`;
      svg.append(line);
    });
  }

  function connectionLine(from, to, className) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", `${from.x * 100}%`);
    line.setAttribute("y1", `${from.y * 100}%`);
    line.setAttribute("x2", `${to.x * 100}%`);
    line.setAttribute("y2", `${to.y * 100}%`);
    line.setAttribute("class", className);
    return line;
  }

  function renderMessages(messages, agent) {
    const list = $("message-list");
    list.replaceChildren();
    if (!messages.length) {
      const empty = make("div", "empty-state");
      empty.append(icon("i-message"), make("p", "", agent
        ? `这是与 ${agent.label} 的独立会话。发送消息后，切换回来仍会保留上下文。`
        : "选择一个 Agent 后即可开始独立会话。"));
      list.append(empty);
      return;
    }
    [...messages].reverse().forEach((message) => {
      const item = make("article", `message role-${message.role}`);
      const avatar = make("span", "message-avatar", monogram(message.speakerLabel));
      const body = make("div", "message-body");
      const meta = make("div", "message-meta");
      meta.append(make("strong", "", message.speakerLabel), make("time", "", formatTime(message.at)));
      body.append(meta, make("p", "", message.content));
      item.append(avatar, body);
      list.append(item);
    });
    requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
  }

  function renderNotices(notices) {
    const list = $("protocol-list");
    list.replaceChildren();
    const filtered = notices.filter((notice) => {
      if (state.filter === "error") return notice.level === "error";
      if (state.filter === "action") return notice.kind.includes("action") || notice.kind.includes("outcome") || notice.kind.includes("discover");
      return true;
    });
    if (!filtered.length) {
      list.append(make("p", "protocol-empty", "当前筛选下还没有协议事件。"));
      return;
    }
    filtered.forEach((notice) => {
      const details = make("details", `protocol-event level-${notice.level}`);
      const summary = make("summary", "");
      const index = make("span", "event-index", String(notice.kind || "event").slice(0, 2).toUpperCase());
      const copy = make("span", "event-copy");
      copy.append(make("strong", "", notice.summary), make("small", "", `${notice.kind} · ${formatTime(notice.at)}`));
      summary.append(index, copy);
      const pre = make("pre", "", JSON.stringify(notice.payload || {}, null, 2));
      details.append(summary, pre);
      list.append(details);
    });
  }

  function setComposer(enabled, agent) {
    const input = $("message-input");
    const button = $("send-button");
    input.disabled = !enabled;
    button.disabled = !enabled || !input.value.trim();
    input.placeholder = enabled
      ? `对 ${agent?.label || "Agent"} 说话…`
      : agent ? `${agent.label} 当前离线，可切换查看其他会话…` : "等待 Agent 加入场景…";
    $("agent-activity").classList.toggle("is-online", enabled);
  }

  function sendMessage() {
    const input = $("message-input");
    const content = input.value.trim();
    if (!content || state.socket?.readyState !== WebSocket.OPEN) return;
    state.socket.send(JSON.stringify({ type: "send_message", content }));
    input.value = "";
    updateComposerCount();
    setComposer(false);
  }

  function setBrowserConnection(status) {
    if (status === "closed") {
      if (state.voice.enabled) setVoicePhase("starting", "Live 通道重连中…");
      toast("浏览器实时通道断开，正在重连…", "warning");
    }
  }

  function updateComposerCount() {
    const input = $("message-input");
    setText("character-count", `${input.value.length} / 4000`);
    const agent = state.snapshot?.participants?.find((item) => item.id === state.snapshot.agentId);
    $("send-button").disabled = !(agent?.online && input.value.trim());
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
  }

  function handleVoiceMessage(message) {
    if (message.type === "voice_capabilities") {
      state.voice.capabilities = message.data || {};
      state.voice.available = Boolean(message.data?.available);
      const agent = state.snapshot?.participants?.find((item) => item.id === state.snapshot?.agentId);
      renderVoiceButton(Boolean(agent?.online && state.snapshot?.connection?.hubConnected));
      return;
    }
    if (message.type === "voice_state") {
      if (state.voice.enabled || message.phase === "off") {
        setVoicePhase(message.phase || "off", message.label || "");
      }
      return;
    }
    if (message.type === "voice_transcript") {
      setText("voice-transcript", `你说：${message.text}`);
      return;
    }
    if (message.type === "voice_audio_start") {
      beginVoicePlayback(message.text || "");
      return;
    }
    if (message.type === "voice_audio_chunk") {
      playVoiceChunk(message.audio_base64, Number(message.sample_rate) || 24_000);
      return;
    }
    if (message.type === "voice_audio_end") {
      finishVoicePlayback();
      return;
    }
    if (message.type === "voice_error") {
      toast(message.error || "Live 语音处理失败", "error");
      if (state.voice.enabled) {
        state.voice.capture = freshVoiceCapture();
        setVoicePhase("listening", "本轮失败，请继续说");
      }
    }
  }

  function renderVoiceButton(agentOnline) {
    const button = $("voice-live-button");
    const supported = Boolean(navigator.mediaDevices?.getUserMedia && window.AudioWorkletNode);
    button.disabled = !state.voice.enabled && (!state.voice.available || !agentOnline || !supported);
    button.classList.toggle("is-active", state.voice.enabled);
    button.setAttribute("aria-pressed", String(state.voice.enabled));
    const reason = !supported
      ? "浏览器不支持"
      : !state.voice.available
        ? state.voice.capabilities?.reason || "未配置 Key"
        : !agentOnline
          ? "等待 Agent"
          : "开启 LIVE";
    if (!state.voice.enabled) setText("voice-live-label", reason);
    button.title = reason;
    $("voice-live-bar").hidden = !state.voice.enabled;
  }

  async function startVoiceLive() {
    if (state.voice.enabled) return;
    if (!state.voice.available) {
      toast(state.voice.capabilities?.reason || "服务端未配置小米 MiMo API Key", "error");
      return;
    }
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      await audioContext.audioWorklet.addModule("/pcm-processor.js");
      await audioContext.resume();
      const source = audioContext.createMediaStreamSource(mediaStream);
      const worklet = new AudioWorkletNode(audioContext, "pcm-capture");
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      source.connect(worklet);
      worklet.connect(silentGain);
      silentGain.connect(audioContext.destination);
      worklet.port.onmessage = (event) => handlePcmChunk(new Int16Array(event.data));

      state.voice.enabled = true;
      state.voice.mediaStream = mediaStream;
      state.voice.audioContext = audioContext;
      state.voice.worklet = worklet;
      state.voice.silentGain = silentGain;
      state.voice.capture = freshVoiceCapture();
      state.voice.playbackNextTime = audioContext.currentTime;
      setVoicePhase("starting", "正在开启小米语音 Live…");
      renderVoiceButton(true);
      if (state.socket?.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({ type: "voice_live_start" }));
      }
    } catch (error) {
      await stopVoiceLive(false);
      toast(`无法开启麦克风：${error instanceof Error ? error.message : String(error)}`, "error");
    }
  }

  async function stopVoiceLive(notifyServer = true) {
    const voice = state.voice;
    voice.enabled = false;
    clearTimeout(voice.resumeTimer);
    for (const source of voice.playbackSources) {
      try { source.stop(); } catch { /* already stopped */ }
    }
    voice.playbackSources.clear();
    voice.worklet?.disconnect();
    voice.silentGain?.disconnect();
    for (const track of voice.mediaStream?.getTracks?.() || []) track.stop();
    if (voice.audioContext && voice.audioContext.state !== "closed") {
      await voice.audioContext.close();
    }
    voice.mediaStream = null;
    voice.audioContext = null;
    voice.worklet = null;
    voice.silentGain = null;
    voice.capture = null;
    setVoicePhase("off", "Live 已关闭");
    if (notifyServer && state.socket?.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({ type: "voice_live_stop" }));
    }
    const agent = state.snapshot?.participants?.find((item) => item.id === state.snapshot?.agentId);
    renderVoiceButton(Boolean(agent?.online && state.snapshot?.connection?.hubConnected));
  }

  function freshVoiceCapture() {
    return {
      preRoll: [],
      captured: [],
      speaking: false,
      activeStreak: 0,
      silenceChunks: 0,
    };
  }

  function handlePcmChunk(pcm) {
    if (!state.voice.enabled || !["listening", "hearing"].includes(state.voice.phase)) return;
    const capture = state.voice.capture || (state.voice.capture = freshVoiceCapture());
    const chunk = new Int16Array(pcm);
    const active = pcmRms(chunk) >= 500;
    if (!capture.speaking) {
      capture.preRoll.push(chunk);
      if (capture.preRoll.length > 10) capture.preRoll.shift();
      capture.activeStreak = active ? capture.activeStreak + 1 : 0;
      if (capture.activeStreak < 3) return;
      capture.speaking = true;
      capture.captured.push(...capture.preRoll);
      capture.preRoll = [];
      capture.silenceChunks = 0;
      setVoicePhase("hearing", "检测到语音，继续说…");
      return;
    }
    capture.captured.push(chunk);
    capture.silenceChunks = active ? 0 : capture.silenceChunks + 1;
    const duration = capture.captured.length * 0.03;
    const reachedSilence = duration >= 0.35 && capture.silenceChunks >= 27;
    const reachedLimit = duration >= 20;
    if (!reachedSilence && !reachedLimit) return;
    const trim = Math.max(0, capture.silenceChunks - 5);
    if (trim) capture.captured.splice(-trim, trim);
    const encoded = pcmChunksToBase64(capture.captured);
    state.voice.capture = freshVoiceCapture();
    setVoicePhase("transcribing", "小米 MiMo 正在识别…");
    if (state.socket?.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({
        type: "voice_utterance",
        audio_base64: encoded,
        sample_rate: 16_000,
      }));
    } else {
      setVoicePhase("starting", "Live 通道重连中…");
    }
  }

  function pcmRms(pcm) {
    if (!pcm.length) return 0;
    let sum = 0;
    for (let index = 0; index < pcm.length; index += 1) sum += pcm[index] * pcm[index];
    return Math.sqrt(sum / pcm.length);
  }

  function pcmChunksToBase64(chunks) {
    const samples = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const bytes = new Uint8Array(samples * 2);
    let offset = 0;
    for (const chunk of chunks) {
      const source = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
      bytes.set(source, offset);
      offset += source.length;
    }
    let binary = "";
    for (let index = 0; index < bytes.length; index += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    }
    return btoa(binary);
  }

  function beginVoicePlayback(text) {
    if (!state.voice.enabled || !state.voice.audioContext) return;
    clearTimeout(state.voice.resumeTimer);
    state.voice.capture = freshVoiceCapture();
    state.voice.playbackNextTime = Math.max(
      state.voice.audioContext.currentTime + 0.04,
      state.voice.playbackNextTime,
    );
    setVoicePhase("speaking", "Agent 正在说话…");
    setText("voice-transcript", text ? `Ling：${text}` : "MiMo TTS 流式播放中");
  }

  function playVoiceChunk(encoded, sampleRate) {
    const context = state.voice.audioContext;
    if (!state.voice.enabled || !context || !encoded) return;
    const binary = atob(encoded);
    const pcm = new Int16Array(Math.floor(binary.length / 2));
    for (let index = 0; index < pcm.length; index += 1) {
      const low = binary.charCodeAt(index * 2);
      const high = binary.charCodeAt(index * 2 + 1);
      pcm[index] = (high << 8) | low;
    }
    const buffer = context.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) channel[index] = pcm[index] / 0x8000;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(context.currentTime + 0.025, state.voice.playbackNextTime);
    source.start(startAt);
    state.voice.playbackNextTime = startAt + buffer.duration;
    state.voice.playbackSources.add(source);
    source.onended = () => state.voice.playbackSources.delete(source);
  }

  function finishVoicePlayback() {
    const context = state.voice.audioContext;
    if (!state.voice.enabled || !context) return;
    const delay = Math.max(0, (state.voice.playbackNextTime - context.currentTime) * 1000) + 120;
    clearTimeout(state.voice.resumeTimer);
    state.voice.resumeTimer = setTimeout(() => {
      if (!state.voice.enabled) return;
      state.voice.capture = freshVoiceCapture();
      setVoicePhase("listening", "正在听你说话");
      setText("voice-transcript", "MiMo ASR · Agent · MiMo TTS");
    }, delay);
  }

  function setVoicePhase(phase, label) {
    state.voice.phase = phase;
    const bar = $("voice-live-bar");
    bar.className = `voice-live-bar phase-${phase}`;
    bar.hidden = !state.voice.enabled;
    setText("voice-phase-label", label || phase);
    setText("voice-live-label", state.voice.enabled ? label || phase : "开启 LIVE");
  }

  function toast(message, kind = "info") {
    const node = $("toast");
    clearTimeout(state.toastTimer);
    node.textContent = message;
    node.className = `toast is-${kind}`;
    node.hidden = false;
    state.toastTimer = setTimeout(() => { node.hidden = true; }, 3600);
  }

  function make(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  }

  function icon(id) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${id}`);
    svg.append(use);
    return svg;
  }

  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value == null || value === "" ? "—" : String(value);
  }

  function monogram(value) {
    const text = String(value || "?").trim();
    return text ? [...text][0].toUpperCase() : "?";
  }

  function pad(value) { return String(Number(value) || 0).padStart(2, "0"); }
  function formatTime(value) { return new Date(Number(value) || Date.now()).toLocaleTimeString("zh-CN", { hour12: false }); }
  function formatLatency(value) { return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`; }

  $("composer").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(); });
  $("message-input").addEventListener("input", updateComposerCount);
  $("message-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); }
  });
  $("reconnect-button").addEventListener("click", () => {
    if (state.socket?.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({ type: "reconnect" }));
      toast("正在重连 Star Hub…");
    } else connect();
  });
  function selectAgentFromEvent(event) {
    const target = event.target.closest("[data-agent-id]");
    if (!target || !target.dataset.agentId || state.socket?.readyState !== WebSocket.OPEN) return;
    if (target.dataset.agentId === state.snapshot?.agentId) return;
    state.socket.send(JSON.stringify({ type: "select_agent", agent_id: target.dataset.agentId }));
  }
  $("participant-list").addEventListener("click", selectAgentFromEvent);
  $("participant-layer").addEventListener("click", selectAgentFromEvent);
  $("agent-selector").addEventListener("change", (event) => {
    const agentId = event.target.value;
    if (!agentId || agentId === state.snapshot?.agentId || state.socket?.readyState !== WebSocket.OPEN) return;
    state.socket.send(JSON.stringify({ type: "select_agent", agent_id: agentId }));
  });
  $("voice-live-button").addEventListener("click", () => {
    if (state.voice.enabled) void stopVoiceLive();
    else void startVoiceLive();
  });
  $("voice-stop-button").addEventListener("click", () => void stopVoiceLive());
  $("protocol-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    $("protocol-tabs").querySelectorAll("button").forEach((node) => node.classList.toggle("is-active", node === button));
    if (state.snapshot) renderNotices(state.snapshot.notices || []);
  });
  setInterval(() => setText("clock", new Date().toLocaleTimeString("zh-CN", { hour12: false })), 1000);
  setText("clock", new Date().toLocaleTimeString("zh-CN", { hour12: false }));
  connect();
})();
