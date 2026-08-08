"use strict";

// Browser interpreter client.
//
// Audio in and audio out both run through Web Audio. The server owns the turn
// state machine, so this file sends microphone blocks and control messages and
// renders what comes back; it makes no decision about when a turn starts or ends.

const STAGES = ["capture", "asr", "verify", "llm", "tts"];
const STAGE_LABELS = {
  capture: "capture",
  asr: "transcribe",
  verify: "verify",
  llm: "translate",
  tts: "speak",
};
const STAGE_MARKS = {
  pending: "·",
  running: "…",
  ok: "✓",
  empty: "∅",
  skipped: "–",
  failed: "✗",
};
// One WebSocket frame per render quantum would be ~375 frames a second. Batching
// to about 32 ms matches the segmenter's window and keeps the socket quiet.
const BLOCKS_PER_MESSAGE = 4;
const MAXIMUM_LOG_LINES = 500;
const MONITORING_REFRESH_MILLISECONDS = 5000;
const CHART_COLORS = ["#0969da", "#1a7f37", "#bf8700", "#8250df", "#cf222e", "#1b7c83", "#bc4c00", "#57606a"];

const state = {
  socket: null,
  audioContext: null,
  captureNode: null,
  mediaStream: null,
  playbackContext: null,
  playbackCursor: 0,
  playedSamples: 0,
  playbackRate: 24000,
  pendingBlocks: [],
  talking: false,
  mode: "push_to_talk",
  voiceMode: "push_to_talk",
  session: null,
  budget: {},
  latestTurn: null,
  services: {},
  logsPaused: false,
  configuration: null,
  operations: null,
  operationTimer: null,
  monitoring: null,
  monitoringTimer: null,
  monitoringResizeTimer: null,
  historyOffset: 0,
  historyLimit: 200,
  historyTotal: 0,
  recentHistoryLimit: 20,
  messages: {},
};

const view = {};

function element(id) {
  return document.getElementById(id);
}

function setStatus(text, kind) {
  view.status.textContent = text;
  view.status.dataset.kind = kind || "";
}

function translate(message) {
  return state.messages[message] || message;
}

async function loadInterface() {
  const result = await fetchJson("/api/interface");
  state.messages = result.messages || {};
  document.documentElement.lang = result.locale || "en";
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const source = node.nodeValue.trim();
    if (source && state.messages[source]) {
      node.nodeValue = node.nodeValue.replace(source, state.messages[source]);
    }
  }
  document.querySelectorAll("[placeholder], [aria-label], [title]").forEach((node) => {
    ["placeholder", "aria-label", "title"].forEach((attribute) => {
      const source = node.getAttribute(attribute);
      if (source && state.messages[source]) node.setAttribute(attribute, state.messages[source]);
    });
  });
}

// -- transport --------------------------------------------------------------

function send(message) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message));
  }
}

function connect() {
  if (state.socket && [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState)) {
    return;
  }
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws`);
  socket.binaryType = "arraybuffer";
  state.socket = socket;

  socket.onopen = () => {
    setStatus("connected", "ok");
    view.connection.textContent = translate("Disconnect");
  };
  socket.onclose = () => {
    setStatus("disconnected", "error");
    stopCapture();
    view.connection.textContent = translate("Connect");
  };
  socket.onerror = () => setStatus("connection error", "error");
  socket.onmessage = (message) => {
    if (typeof message.data === "string") {
      handleControl(JSON.parse(message.data));
    } else {
      playChunk(new Float32Array(message.data));
    }
  };
}

function handleControl(message) {
  switch (message.type) {
    case "session":
      state.session = message.session;
      state.playbackRate = message.playback_rate;
      state.mode = message.mode;
      if (state.mode !== "text") state.voiceMode = state.mode;
      state.budget = message.budget_ms || {};
      state.recentHistoryLimit = message.history_turns ?? 20;
      view.session.textContent = message.session;
      view.routing.textContent = message.routing || "—";
      view.performanceMode.textContent = message.perf_mode || "—";
      view.privacyWarning.hidden = !message.audio_leaves_device;
      fillLanguages(message.languages, message.target);
      updateRecentTurnsVisibility();
      updateModeView();
      setStatus("session ready", "ok");
      break;
    case "event":
      handleEvent(message.kind, message.payload || {});
      break;
    case "log":
      appendLog(message.line);
      break;
    case "playback_begin":
      state.playbackRate = message.rate || state.playbackRate;
      resetPlayback();
      break;
    case "playback_end":
      break;
    case "playback_flush":
      resetPlayback();
      break;
    case "settings_reload":
      setStatus("settings applied; reconnecting", "ok");
      state.socket.close();
      window.setTimeout(connect, 250);
      break;
    case "error":
      setStatus(message.message, "error");
      break;
    default:
      break;
  }
}

// -- interpreter events -----------------------------------------------------

function handleEvent(kind, payload) {
  switch (kind) {
    case "state":
      view.state.textContent = payload.state;
      view.state.dataset.state = payload.state;
      // The server shuts its own gate while speaking; muting here as well stops
      // the browser from sending audio that would only be discarded.
      setCaptureEnabled(payload.state !== "SPEAKING" && state.mode !== "text");
      if (payload.state === "IDLE") {
        state.talking = false;
        updateTalkButton();
      }
      if (payload.state === "LISTENING") {
        view.source.textContent = "";
        view.translation.textContent = "";
        resetStages();
        setStage("capture", "running");
      }
      break;
    case "level":
      view.level.value = Math.min(1, (payload.rms || 0) * 12);
      break;
    case "lang":
      view.language.textContent = `${payload.lang || "—"}${
        payload.source && payload.source !== "lid" ? ` (${payload.source})` : ""
      }`;
      break;
    case "eou":
      resetStages();
      setStage("capture", "ok", `${payload.seconds}s ${payload.ended_by}`);
      setStage("asr", "running");
      break;
    case "text_submitted":
      resetStages();
      ["capture", "asr", "verify"].forEach((stage) => setStage(stage, "skipped", "text input"));
      setStage("llm", "running");
      break;
    case "asr":
      if (payload.empty) {
        setStage("asr", "empty");
        view.source.textContent = "(silence)";
      } else {
        setStage("asr", "ok", `n_best=${(payload.n_best || []).length}`);
        setStage("llm", "running");
        view.source.textContent = payload.text || "";
      }
      break;
    case "verify":
      if (payload.state === "done") {
        setStage("verify", "ok", `cer=${payload.cer}`);
      } else if (payload.state === "failed") {
        setStage("verify", "failed", payload.message || "");
      } else if (payload.state === "running") {
        setStage("verify", "running", payload.reason || "");
      }
      break;
    case "translation_delta":
      view.translation.textContent = payload.text || "";
      break;
    case "translation":
      if (payload.timeout) {
        setStage("llm", "failed", "timeout");
      } else if (!payload.text) {
        setStage("llm", "empty");
      } else {
        setStage("llm", "ok");
        setStage("tts", "running");
        view.translation.textContent = payload.text;
      }
      break;
    case "first_audio":
      setStage("tts", "ok", `${payload.ms}ms`);
      break;
    case "history":
      appendHistory(payload);
      break;
    case "turn":
      finishStages();
      state.latestTurn = payload;
      renderTurnDiagnostics(payload);
      break;
    case "budget":
      if (state.latestTurn) {
        state.latestTurn.over_budget_ms = payload.over || {};
        renderTurnDiagnostics(state.latestTurn);
      }
      break;
    case "service":
      state.services[payload.name] = payload;
      renderServiceStatus();
      break;
    case "placement":
      if (state.services[payload.role]) {
        state.services[payload.role].side = payload.side;
        state.services[payload.role].degraded = true;
        renderServiceStatus();
      }
      setStatus(`${payload.role} to ${payload.side}: ${payload.reason}`, "error");
      break;
    case "privacy":
      view.privacyWarning.hidden = !payload.audio_leaves_device;
      break;
    case "error":
      setStage(payload.where, "failed", payload.message || "");
      setStatus(`${payload.where}: ${payload.message}`, "error");
      break;
    default:
      break;
  }
}

// -- stage panel ------------------------------------------------------------

function resetStages() {
  STAGES.forEach((stage) => setStage(stage, "pending", ""));
}

function setStage(stage, status, detail) {
  const row = view.stages[stage];
  if (!row) {
    return;
  }
  row.dataset.status = status;
  row.querySelector(".stage-mark").textContent = STAGE_MARKS[status] || "·";
  row.querySelector(".stage-status").textContent = status;
  row.querySelector(".stage-detail").textContent = detail || "";
}

function finishStages() {
  STAGES.forEach((stage) => {
    const row = view.stages[stage];
    if (row && (row.dataset.status === "pending" || row.dataset.status === "running")) {
      setStage(stage, "skipped", "");
    }
  });
}

// -- capture ----------------------------------------------------------------

async function startCapture() {
  if (state.audioContext) {
    return;
  }
  const selectedInput = element("input-device").value;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      ...(selectedInput ? { deviceId: { exact: selectedInput } } : {}),
    },
  });
  // Ask for the working rate. A browser may refuse, so the rate actually used is
  // reported to the server, which resamples rather than mis-reading the pitch.
  const context = new AudioContext({ sampleRate: 16000 });
  await context.audioWorklet.addModule("/static/capture-worklet.js");
  const source = context.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(context, "kotonoha-capture");

  worklet.port.onmessage = (message) => {
    state.pendingBlocks.push(message.data);
    if (state.pendingBlocks.length < BLOCKS_PER_MESSAGE) {
      return;
    }
    const total = state.pendingBlocks.reduce((sum, block) => sum + block.length, 0);
    const batch = new Float32Array(total);
    let offset = 0;
    state.pendingBlocks.forEach((block) => {
      batch.set(block, offset);
      offset += block.length;
    });
    state.pendingBlocks.length = 0;
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(batch.buffer);
    }
  };

  source.connect(worklet);
  // Terminating in a zero-gain node keeps the graph alive without echoing the
  // microphone into the speaker.
  const sink = context.createGain();
  sink.gain.value = 0;
  worklet.connect(sink).connect(context.destination);

  state.audioContext = context;
  state.captureNode = worklet;
  state.mediaStream = stream;
  send({ type: "hello", capture_rate: context.sampleRate });
  view.micRate.textContent = `${context.sampleRate} Hz`;
  setCaptureEnabled(state.mode !== "text");
  await refreshAudioDevices();
}

function setCaptureEnabled(enabled) {
  const active = Boolean(state.captureNode) && enabled;
  view.mic.textContent = active ? "OPEN" : "SHUT";
  view.mic.dataset.open = String(active);
  if (state.captureNode) {
    state.captureNode.port.postMessage({ enabled });
  }
}

function stopCapture() {
  if (state.captureNode) {
    state.captureNode.port.postMessage({ enabled: false });
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
  }
  if (state.audioContext) {
    state.audioContext.close();
  }
  state.audioContext = null;
  state.captureNode = null;
  state.mediaStream = null;
}

// -- playback ---------------------------------------------------------------

function playbackContext() {
  if (!state.playbackContext || state.playbackContext.sampleRate !== state.playbackRate) {
    if (state.playbackContext) {
      state.playbackContext.close();
    }
    state.playbackContext = new AudioContext({ sampleRate: state.playbackRate });
    applyOutputDevice(state.playbackContext);
  }
  return state.playbackContext;
}

function applyOutputDevice(context) {
  const selectedOutput = element("output-device").value;
  if (!selectedOutput || typeof context.setSinkId !== "function") {
    return;
  }
  context.setSinkId(selectedOutput).catch((error) => {
    setStatus(`output device unavailable: ${error.message}`, "error");
  });
}

async function refreshAudioDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    setStatus("audio device enumeration is unavailable", "error");
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  fillAudioDevices(element("input-device"), devices, "audioinput", "Microphone");
  fillAudioDevices(element("output-device"), devices, "audiooutput", "Speaker");
}

function fillAudioDevices(select, devices, kind, fallbackLabel) {
  const selected = select.value;
  select.replaceChildren();
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "System default";
  select.appendChild(defaultOption);
  devices
    .filter((device) => device.kind === kind)
    .forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `${fallbackLabel} ${index + 1}`;
      option.selected = device.deviceId === selected;
      select.appendChild(option);
    });
}

async function testAudioDevices() {
  try {
    await startCapture();
    const context = playbackContext();
    await context.resume();
    applyOutputDevice(context);
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    gain.gain.value = 0.04;
    oscillator.frequency.value = 440;
    oscillator.connect(gain).connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.25);
    setStatus("audio test active; verify the meter and tone", "ok");
  } catch (error) {
    setStatus(`audio test failed: ${error.message}`, "error");
  }
}

function resetPlayback() {
  state.playbackCursor = 0;
  state.playedSamples = 0;
}

function playChunk(samples) {
  if (samples.length === 0) {
    return;
  }
  const context = playbackContext();
  const buffer = context.createBuffer(1, samples.length, state.playbackRate);
  buffer.copyToChannel(samples, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);

  // Schedule back to back. Starting every chunk at "now" would overlap them.
  const startAt = Math.max(context.currentTime, state.playbackCursor);
  source.start(startAt);
  state.playbackCursor = startAt + buffer.duration;

  // The server reopens the microphone only once it believes playback drained, so
  // this acknowledgement is what keeps half-duplex correct rather than optional.
  source.onended = () => {
    state.playedSamples += samples.length;
    send({ type: "played", samples: state.playedSamples });
  };
}

// -- controls ---------------------------------------------------------------

function updateTalkButton() {
  view.talk.textContent = state.talking ? translate("Stop (space)") : translate("Talk (space)");
  view.talk.dataset.active = String(state.talking);
}

function updateModeView() {
  view.mode.textContent = state.mode;
  view.textRow.hidden = state.mode !== "text";
  setCaptureEnabled(state.mode !== "text");
}

function toggleTalk() {
  if (state.mode !== "push_to_talk") {
    return;
  }
  state.talking = !state.talking;
  send({ type: "ptt", down: state.talking });
  updateTalkButton();
}

function cycleMode() {
  const order = ["push_to_talk", "auto", "text"];
  state.mode = order[(order.indexOf(state.mode) + 1) % order.length];
  if (state.mode !== "text") state.voiceMode = state.mode;
  send({ type: "mode", mode: state.mode });
  updateModeView();
}

function toggleTextMode() {
  state.mode = state.mode === "text" ? state.voiceMode : "text";
  send({ type: "mode", mode: state.mode });
  updateModeView();
  if (state.mode === "text") view.textInput.focus();
  else view.textInput.value = "";
}

function cycleTargetLanguage() {
  if (view.target.options.length === 0) return;
  view.target.selectedIndex = (view.target.selectedIndex + 1) % view.target.options.length;
  send({ type: "target", language: view.target.value });
}

function clearCurrentTurn() {
  view.source.textContent = "";
  view.translation.textContent = "";
}

function updateRecentTurnsVisibility() {
  const visible = element("recent-turns-visible").checked && state.recentHistoryLimit > 0;
  element("recent-turns").hidden = !visible;
  element("recent-turns-visible").disabled = state.recentHistoryLimit === 0;
}

function toggleConnection() {
  if (state.socket && [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState)) {
    state.socket.close(1000, "operator disconnected");
  } else {
    connect();
  }
}

function fillLanguages(languages, target) {
  view.target.innerHTML = "";
  (languages || []).forEach((language) => {
    const option = document.createElement("option");
    option.value = language;
    option.textContent = language;
    option.selected = language === target;
    view.target.appendChild(option);
  });
}

function appendHistory(payload) {
  const row = document.createElement("div");
  row.className = "history-row";
  row.textContent = `${payload.src_lang || "?"}→${payload.tgt_lang || "?"}  ${
    payload.source_text || ""
  }  ⇒  ${payload.translation || ""}`;
  view.history.prepend(row);
  while (view.history.childElementCount > state.recentHistoryLimit) {
    view.history.lastElementChild.remove();
  }
  view.history.hidden = state.recentHistoryLimit === 0;
}

function renderTurnDiagnostics(record) {
  const stages = record.stages_ms || {};
  const limits = state.budget;
  const rows = [
    [translate("ASR (+verify)"), stages.asr, (limits.asr || 0) + (limits.verify || 0)],
    [translate("LLM first clause"), stages.llm_first_clause, limits.llm_first_clause],
    [translate("TTS first packet"), stages.tts_first_packet, limits.tts_first_packet],
    [translate("EOU to audio"), stages.total_to_first_audio, (limits.total || 0) - (limits.silence || 0)],
  ];
  const container = element("turn-latency");
  container.replaceChildren();
  rows.forEach(([label, measured, limit]) => {
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = Number.isFinite(measured) ? `${Math.round(measured)} / ${limit} ms` : "—";
    value.dataset.overBudget = String(Number.isFinite(measured) && measured > limit);
    container.append(name, value);
  });
  Object.entries(record.over_budget_ms || {}).forEach(([stage, duration]) => {
    const name = document.createElement("span");
    name.textContent = stage;
    const value = document.createElement("strong");
    value.textContent = `+${Math.round(duration)} ms`;
    value.dataset.overBudget = "true";
    container.append(name, value);
  });
}

function renderServiceStatus() {
  const container = element("service-status");
  const roles = ["asr", "asr-verify", "llm", "tts"];
  const rows = roles.map((role) => {
    const service = state.services[role];
    const row = document.createElement("div");
    row.className = "service-status-row";
    const name = document.createElement("strong");
    name.textContent = role;
    const status = document.createElement("span");
    if (!service) {
      status.textContent = "?";
      status.dataset.status = "unknown";
    } else {
      const detail = service.detail || {};
      const tag = detail.backend || detail.error || "";
      status.textContent = `${service.ok ? "UP" : "DOWN"} · ${service.side || "local"}${
        service.degraded ? " · degraded" : ""
      }${tag ? ` · ${String(tag).slice(0, 40)}` : ""}`;
      status.dataset.status = service.ok ? "ready" : "failed";
    }
    row.append(name, status);
    return row;
  });
  container.replaceChildren(...rows);
}

// -- logs -------------------------------------------------------------------

function appendLog(line) {
  if (state.logsPaused) {
    return;
  }
  const filter = view.logFilter.value.trim().toLowerCase();
  if (filter && !line.toLowerCase().includes(filter)) {
    return;
  }
  const row = document.createElement("div");
  row.className = "log-line";
  const level = (line.match(/\]\s+(\w+)/) || [])[1];
  if (level) {
    row.dataset.level = level.toLowerCase();
  }
  row.textContent = line;
  view.logs.appendChild(row);
  while (view.logs.childElementCount > MAXIMUM_LOG_LINES) {
    view.logs.firstElementChild.remove();
  }
  if (view.logFollow.checked) {
    view.logs.scrollTop = view.logs.scrollHeight;
  }
}

// -- application pages -----------------------------------------------------

function showPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => {
    const active = page.id === pageId;
    page.hidden = !active;
    page.dataset.active = String(active);
  });
  document.querySelectorAll("#navigation button").forEach((button) => {
    button.dataset.active = String(button.dataset.page === pageId);
  });
  if (pageId === "monitoring") {
    startMonitoring();
  } else {
    stopMonitoring();
  }
  if (pageId === "configuration") {
    loadConfiguration();
  } else if (pageId === "history-page") {
    loadHistoryArchive();
  } else if (pageId === "operations") {
    loadOperations();
  } else if (pageId === "license") {
    loadLicense();
  }
}

function setTheme(theme) {
  const selected = ["system", "light", "dark"].includes(theme) ? theme : "system";
  document.documentElement.dataset.theme = selected;
  localStorage.setItem("kotonoha-theme", selected);
  element("theme-select").value = selected;
  if (state.monitoring) {
    renderMonitoringCharts(state.monitoring);
  }
}

async function fetchJson(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...(options || {}),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

// -- monitoring ------------------------------------------------------------

function startMonitoring() {
  loadMonitoring();
  if (state.monitoringTimer === null) {
    state.monitoringTimer = window.setInterval(loadMonitoring, MONITORING_REFRESH_MILLISECONDS);
  }
}

function stopMonitoring() {
  if (state.monitoringTimer !== null) {
    window.clearInterval(state.monitoringTimer);
    state.monitoringTimer = null;
  }
}

async function loadMonitoring() {
  const windowSeconds = element("monitor-window").value;
  try {
    const monitoring = await fetchJson(`/api/monitoring?window_seconds=${windowSeconds}`);
    state.monitoring = monitoring;
    renderMonitoring(monitoring);
  } catch (error) {
    element("monitor-status").textContent = `Metrics unavailable: ${error.message}`;
    element("monitor-status").dataset.kind = "error";
  }
}

function renderMonitoring(monitoring) {
  const summary = monitoring.summary;
  element("monitor-services-ready").textContent = `${summary.services_ready}/${summary.services_total}`;
  element("monitor-turns").textContent = formatNumber(summary.turns_total);
  element("monitor-first-audio").textContent = formatDuration(summary.first_audio_p95_ms);
  element("monitor-over-budget").textContent = formatNumber(summary.over_budget_turns_total);
  element("monitor-failovers").textContent = formatNumber(summary.failovers_total);
  const generatedAt = new Date(monitoring.generated_at * 1000).toLocaleTimeString();
  const sampleCount = monitoring.series.length;
  element("monitor-status").textContent = monitoring.last_error
    ? `Last update ${generatedAt}; collector error: ${monitoring.last_error}`
    : `Last update ${generatedAt}; ${sampleCount} samples at ${monitoring.sample_interval_seconds}s intervals`;
  element("monitor-status").dataset.kind = monitoring.last_error ? "error" : "ok";
  renderMonitoringServices(monitoring.services);
  renderMonitoringCharts(monitoring);
}

function renderMonitoringServices(services) {
  const grid = element("monitor-service-grid");
  const cards = services.map((service) => {
    const card = document.createElement("article");
    card.className = "monitor-service-card";
    const heading = document.createElement("div");
    heading.className = "monitor-service-heading";
    const title = document.createElement("h3");
    title.textContent = `${service.role} · ${service.source}`;
    const status = document.createElement("span");
    status.className = "monitor-service-status";
    status.dataset.status = service.ready === true ? "ready" : service.scrape_up ? "failed" : "unreachable";
    status.textContent = service.ready === true ? "Ready" : service.scrape_up ? "Not ready" : "Unreachable";
    heading.append(title, status);
    card.appendChild(heading);

    const details = document.createElement("dl");
    details.className = "monitor-service-details";
    appendDefinition(details, "Host", `${service.operating_system} · ${service.machine}`);
    appendDefinition(details, "Kernel", service.kernel);
    appendDefinition(details, "Accelerator", `${service.accelerator_backend} · ${service.memory_architecture}`);
    appendDefinition(details, "CPU load", formatPercent(service.cpu_load_ratio, true));
    appendDefinition(
      details,
      "System memory",
      formatMemoryUsage(service.system_memory_total_bytes, service.system_memory_available_bytes, true),
    );
    appendDefinition(
      details,
      "Accelerator memory",
      formatMemoryUsage(service.accelerator_memory_total_bytes, service.accelerator_memory_free_bytes, false),
    );
    appendDefinition(details, "Disk", formatDiskUsage(service.disk_total_bytes, service.disk_used_bytes));
    card.appendChild(details);

    const meterLabel = document.createElement("label");
    meterLabel.className = "monitor-memory-meter";
    const meterText = document.createElement("span");
    meterText.textContent = `Effective memory ${formatPercent(service.memory_percent)}`;
    const meter = document.createElement("progress");
    meter.max = 100;
    meter.value = service.memory_percent ?? 0;
    meterLabel.append(meterText, meter);
    card.appendChild(meterLabel);
    return card;
  });
  grid.replaceChildren(...cards);
}

function appendDefinition(list, termText, valueText) {
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = termText;
  description.textContent = valueText;
  list.append(term, description);
}

function formatNumber(value) {
  return Number.isFinite(value) ? new Intl.NumberFormat().format(value) : "—";
}

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "—";
  return milliseconds >= 1000 ? `${(milliseconds / 1000).toFixed(2)} s` : `${milliseconds.toFixed(0)} ms`;
}

function formatPercent(value, ratio = false) {
  if (!Number.isFinite(value)) return "—";
  const percent = ratio ? value * 100 : value;
  return `${percent.toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit < 2 ? 0 : 1)} ${units[unit]}`;
}

function formatMemoryUsage(total, freeOrAvailable, availableMeansFree) {
  if (!Number.isFinite(total)) return "—";
  if (!Number.isFinite(freeOrAvailable)) return formatBytes(total);
  const used = Math.max(0, total - freeOrAvailable);
  const qualifier = availableMeansFree ? "used" : "allocated";
  return `${formatBytes(used)} / ${formatBytes(total)} ${qualifier}`;
}

function formatDiskUsage(total, used) {
  if (!Number.isFinite(total)) return "—";
  return Number.isFinite(used) ? `${formatBytes(used)} / ${formatBytes(total)} used` : formatBytes(total);
}

function renderMonitoringCharts(monitoring) {
  const series = monitoring.series || [];
  drawLineChart(element("monitor-availability-chart"), {
    datasets: [{ name: "Ready", values: series.map((point) => point.services_total ? 100 * point.services_ready / point.services_total : null) }],
    timestamps: series.map((point) => point.timestamp),
    minimum: 0,
    maximum: 100,
    formatValue: (value) => `${value.toFixed(0)}%`,
  });

  const memoryKeys = Array.from(new Set(series.flatMap((point) => Object.keys(point.memory_percent || {})))).sort();
  drawLineChart(element("monitor-memory-chart"), {
    datasets: memoryKeys.map((key) => ({ name: key, values: series.map((point) => point.memory_percent?.[key] ?? null) })),
    timestamps: series.map((point) => point.timestamp),
    minimum: 0,
    maximum: 100,
    formatValue: (value) => `${value.toFixed(0)}%`,
  });

  drawLineChart(element("monitor-latency-chart"), {
    datasets: [
      { name: "p95", values: series.map((point) => point.first_audio_p95_ms) },
      { name: "Budget", values: series.map(() => monitoring.summary.first_audio_budget_ms), dashed: true },
    ],
    timestamps: series.map((point) => point.timestamp),
    minimum: 0,
    formatValue: (value) => formatDuration(value),
  });

  const requestRates = series.map((point, index) => {
    if (index === 0) return null;
    const previous = series[index - 1];
    const seconds = point.timestamp - previous.timestamp;
    return seconds > 0 ? Math.max(0, point.requests_total - previous.requests_total) / seconds : null;
  });
  drawLineChart(element("monitor-traffic-chart"), {
    datasets: [{ name: "Requests/s", values: requestRates }],
    timestamps: series.map((point) => point.timestamp),
    minimum: 0,
    formatValue: (value) => value.toFixed(2),
  });
}

function drawLineChart(canvas, options) {
  const width = Math.max(280, canvas.clientWidth || 600);
  const height = Math.max(210, canvas.clientHeight || 240);
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * pixelRatio);
  canvas.height = Math.floor(height * pixelRatio);
  const context = canvas.getContext("2d");
  context.scale(pixelRatio, pixelRatio);
  const styles = getComputedStyle(document.documentElement);
  const muted = styles.getPropertyValue("--muted").trim();
  const line = styles.getPropertyValue("--line").trim();
  context.font = "12px system-ui, sans-serif";

  const datasets = options.datasets.filter((dataset) => dataset.values.some(Number.isFinite));
  renderChartLegend(canvas, datasets);
  if (datasets.length === 0 || options.timestamps.length === 0) {
    context.fillStyle = muted;
    context.textAlign = "center";
    context.fillText("Waiting for samples", width / 2, height / 2);
    return;
  }

  const padding = { left: 48, right: 16, top: 30, bottom: 28 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = datasets.flatMap((dataset) => dataset.values.filter(Number.isFinite));
  let minimum = Number.isFinite(options.minimum) ? options.minimum : Math.min(...values);
  let maximum = Number.isFinite(options.maximum) ? options.maximum : Math.max(...values);
  if (maximum <= minimum) maximum = minimum + 1;

  context.strokeStyle = line;
  context.fillStyle = muted;
  context.lineWidth = 1;
  context.textAlign = "right";
  for (let index = 0; index <= 4; index += 1) {
    const ratio = index / 4;
    const y = padding.top + plotHeight * ratio;
    const value = maximum - (maximum - minimum) * ratio;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(options.formatValue(value), padding.left - 6, y + 4);
  }

  datasets.forEach((dataset, datasetIndex) => {
    context.strokeStyle = CHART_COLORS[datasetIndex % CHART_COLORS.length];
    context.fillStyle = context.strokeStyle;
    context.lineWidth = 2;
    context.setLineDash(dataset.dashed ? [5, 4] : []);
    context.beginPath();
    let drawing = false;
    dataset.values.forEach((value, index) => {
      if (!Number.isFinite(value)) {
        drawing = false;
        return;
      }
      const x = padding.left + plotWidth * (options.timestamps.length === 1 ? 0.5 : index / (options.timestamps.length - 1));
      const y = padding.top + plotHeight * (maximum - value) / (maximum - minimum);
      if (drawing) context.lineTo(x, y);
      else context.moveTo(x, y);
      drawing = true;
    });
    context.stroke();
    context.setLineDash([]);
  });

  const firstTimestamp = options.timestamps[0];
  const lastTimestamp = options.timestamps[options.timestamps.length - 1];
  context.fillStyle = muted;
  context.textAlign = "left";
  context.fillText(new Date(firstTimestamp * 1000).toLocaleTimeString(), padding.left, height - 7);
  context.textAlign = "right";
  context.fillText(new Date(lastTimestamp * 1000).toLocaleTimeString(), width - padding.right, height - 7);
}

function renderChartLegend(canvas, datasets) {
  let legend = canvas.parentElement.querySelector(".monitor-chart-legend");
  if (legend === null) {
    legend = document.createElement("div");
    legend.className = "monitor-chart-legend";
    canvas.before(legend);
  }
  const items = datasets.map((dataset, index) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = CHART_COLORS[index % CHART_COLORS.length];
    item.append(swatch, document.createTextNode(dataset.name));
    return item;
  });
  legend.replaceChildren(...items);
}

// -- configuration ---------------------------------------------------------

async function loadConfiguration() {
  try {
    const target = element("config-target").value;
    const path = target === "remote" ? "/api/config/remote" : "/api/config";
    state.configuration = await fetchJson(path);
    renderConfiguration(state.configuration);
    element("config-status").textContent = "Configuration loaded";
  } catch (error) {
    element("config-status").textContent = error.message;
  }
}

function renderConfiguration(configuration) {
  element("config-path").textContent = configuration.path;
  const sectionNavigation = element("config-sections");
  const fieldContainer = element("config-fields");
  sectionNavigation.replaceChildren();
  fieldContainer.replaceChildren();

  configuration.sections.forEach((section, sectionIndex) => {
    const panel = document.createElement("section");
    panel.className = "settings-section";
    panel.id = `config-section-${section.name}`;
    panel.dataset.section = section.name;
    panel.hidden = sectionIndex !== 0;
    const heading = document.createElement("h2");
    heading.textContent = section.label;
    heading.tabIndex = -1;
    panel.appendChild(heading);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = section.label;
    button.dataset.active = String(sectionIndex === 0);
    button.setAttribute("aria-pressed", String(sectionIndex === 0));
    button.setAttribute("aria-controls", `config-section-${section.name}`);
    button.addEventListener("click", () => {
      sectionNavigation.querySelectorAll("button").forEach((item) => {
        item.dataset.active = String(item === button);
        item.setAttribute("aria-pressed", String(item === button));
      });
      fieldContainer.querySelectorAll(".settings-section").forEach((panel) => {
        panel.hidden = panel.dataset.section !== section.name;
      });
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      button.scrollIntoView({
        behavior: "auto",
        block: "nearest",
        inline: "center",
      });
      panel.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: "start",
      });
      heading.focus({ preventScroll: true });
    });
    sectionNavigation.appendChild(button);

    configuration.fields
      .filter((field) => field.section === section.name)
      .forEach((field) => panel.appendChild(configurationField(field)));
    fieldContainer.appendChild(panel);
  });
}

function configurationField(field) {
  const row = document.createElement("label");
  row.className = "configuration-field";
  const title = document.createElement("span");
  title.className = "configuration-path";
  title.textContent = `${field.path}${field.modified ? "  [modified]" : ""}`;
  row.appendChild(title);

  let input;
  if (field.kind === "select") {
    input = document.createElement("select");
    field.choices.forEach((choice) => {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice;
      option.selected = choice === field.value;
      input.appendChild(option);
    });
  } else if (field.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(field.value);
  } else if (field.value_kind === "collection" || field.kind === "placement") {
    input = document.createElement("textarea");
    input.rows = 4;
    input.value = JSON.stringify(field.value, null, 2);
  } else {
    input = document.createElement("input");
    input.type = field.secret ? "password" : field.value_kind === "number" ? "number" : "text";
    input.value = field.value === null ? "" : String(field.value);
    input.placeholder = field.secret ? "Leave empty to preserve the current value" : "";
  }
  input.dataset.path = field.path;
  input.dataset.kind = field.kind;
  input.dataset.valueKind = field.value_kind;
  input.dataset.optional = String(field.optional);
  input.dataset.secret = String(field.secret);
  input.dataset.original = JSON.stringify(field.value);
  row.appendChild(input);
  const description = document.createElement("p");
  description.className = "configuration-description";
  description.textContent = field.description || "";
  row.appendChild(description);
  return row;
}

function configurationValue(input) {
  if (input.dataset.kind === "bool") {
    return input.checked;
  }
  const raw = input.value.trim();
  if (input.dataset.secret === "true" && raw === "") {
    return undefined;
  }
  if (raw === "" && input.dataset.optional === "true") {
    return null;
  }
  if (input.dataset.valueKind === "number") {
    const number = Number(raw);
    if (!Number.isFinite(number)) {
      throw new Error(`${input.dataset.path} must be numeric`);
    }
    return number;
  }
  if (input.dataset.valueKind === "collection" || input.dataset.kind === "placement") {
    return JSON.parse(raw);
  }
  return raw;
}

async function saveConfiguration() {
  const changes = {};
  try {
    document.querySelectorAll("#config-fields [data-path]").forEach((input) => {
      const value = configurationValue(input);
      if (value !== undefined && JSON.stringify(value) !== input.dataset.original) {
        changes[input.dataset.path] = value;
      }
    });
    if (Object.keys(changes).length === 0) {
      element("config-status").textContent = "No changes";
      return;
    }
    element("config-save").disabled = true;
    const result = await fetchJson("/api/config", {
      method: "PUT",
      body: JSON.stringify({ target: element("config-target").value, changes }),
    });
    const roles = result.reloading_roles.length ? `; reloading ${result.reloading_roles.join(", ")}` : "";
    element("config-status").textContent = `Applied ${result.changed.length} settings${roles}`;
    state.configuration = result;
    renderConfiguration(result);
  } catch (error) {
    element("config-status").textContent = error.message;
  } finally {
    element("config-save").disabled = false;
  }
}

// -- history ---------------------------------------------------------------

async function loadHistoryArchive() {
  const parameters = new URLSearchParams();
  const search = element("history-search").value.trim();
  const language = element("history-language").value;
  const outcome = element("history-outcome").value;
  if (search) parameters.set("query", search);
  if (language) parameters.set("source_language", language);
  if (outcome) parameters.set("outcome", outcome);
  parameters.set("offset", String(state.historyOffset));
  parameters.set("limit", String(state.historyLimit));
  try {
    const result = await fetchJson(`/api/history?${parameters}`);
    state.historyTotal = result.total;
    fillFilter(element("history-language"), result.languages, language, "All languages");
    fillFilter(element("history-outcome"), result.outcomes, outcome, "All outcomes");
    renderHistoryArchive(result.entries);
    const first = result.total ? state.historyOffset + 1 : 0;
    const last = Math.min(state.historyOffset + result.entries.length, result.total);
    element("history-status").textContent = `${first}-${last} of ${result.total} turns`;
    element("history-previous").disabled = state.historyOffset === 0;
    element("history-next").disabled = state.historyOffset + state.historyLimit >= result.total;
  } catch (error) {
    element("history-status").textContent = error.message;
  }
}

function renderHistoryArchive(entries) {
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const heading = document.createElement("tr");
  ["Time", "Direction", "Source", "Translation", "Outcome"].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    heading.appendChild(cell);
  });
  head.appendChild(heading);
  table.appendChild(head);
  const body = document.createElement("tbody");
  entries.forEach((entry) => {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    [
      entry.time,
      `${entry.src_lang || "?"}→${entry.tgt_lang || "?"}`,
      entry.source_text || "",
      entry.translation || "",
      entry.outcome,
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    const show = () => showHistoryDetail(entry);
    row.addEventListener("click", show);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        show();
      }
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  element("history-results").replaceChildren(table);
  element("history-detail").hidden = true;
}

function showHistoryDetail(entry) {
  const diagnostics = [
    ["lang_source", entry.lang_source],
    ["lid_confidence", entry.lid_confidence],
    ["asr_avg_logprob", entry.asr_avg_logprob],
    ["cross_verified", entry.cross_verified],
    ["audio_seconds", entry.audio_seconds],
    ["session", entry.session_id],
    ["turn_id", entry.turn_id],
  ].filter((item) => item[1] !== null && item[1] !== undefined);
  element("history-detail-content").textContent = [
    `${entry.time}  ${entry.src_lang || "?"} → ${entry.tgt_lang || "?"}  [${entry.outcome}]`,
    "",
    "Source",
    entry.source_text || "",
    "",
    "Translation",
    entry.translation || "",
    "",
    diagnostics.map(([key, value]) => `${key}=${value}`).join("  "),
  ].join("\n");
  element("history-detail").hidden = false;
}

function historyParameters() {
  const parameters = new URLSearchParams();
  const search = element("history-search").value.trim();
  const language = element("history-language").value;
  const outcome = element("history-outcome").value;
  if (search) parameters.set("query", search);
  if (language) parameters.set("source_language", language);
  if (outcome) parameters.set("outcome", outcome);
  return parameters;
}

function exportHistoryArchive() {
  window.location.assign(`/api/history/export?${historyParameters()}`);
}

function fillFilter(select, values, selected, allLabel) {
  select.replaceChildren();
  [[allLabel, ""], ...(values || []).map((value) => [value, value])].forEach(([label, value]) => {
    const option = document.createElement("option");
    option.textContent = label;
    option.value = value;
    option.selected = value === selected;
    select.appendChild(option);
  });
}

async function clearHistoryArchive() {
  if (!window.confirm("Delete all recorded interpretation turns?")) {
    return;
  }
  const result = await fetchJson("/api/history", { method: "DELETE" });
  element("history-status").textContent = `Cleared ${result.removed} turns`;
  await loadHistoryArchive();
}

// -- operations and license ------------------------------------------------

async function loadOperations() {
  const result = await fetchJson("/api/operations");
  state.operations = result.operations;
  const select = element("operation-select");
  const current = select.value;
  select.replaceChildren();
  result.operations.forEach((operation) => {
    const option = document.createElement("option");
    option.value = operation.name;
    option.textContent = operation.label || operation.name;
    option.selected = operation.name === current;
    select.appendChild(option);
  });
  renderOperation();
  renderOperationJob(result.job);
}

function renderOperation() {
  const operation = (state.operations || []).find((item) => item.name === element("operation-select").value);
  if (!operation) return;
  element("operation-description").textContent = operation.description;
  const fields = element("operation-fields");
  fields.replaceChildren();
  const defaults = { "replay-seconds": "30", host: "127.0.0.1", samples: "10", "netcheck-seconds": "6", service: "asr" };
  operation.fields.forEach((name) => {
    const label = document.createElement("label");
    label.textContent = name;
    const input = document.createElement("input");
    input.type = ["wav", "glossary-path"].includes(name) ? "file" : "text";
    input.dataset.operationField = name;
    if (input.type !== "file") input.value = defaults[name] || "";
    label.appendChild(input);
    fields.appendChild(label);
  });
}

async function runOperation() {
  const values = {};
  try {
    for (const input of document.querySelectorAll("[data-operation-field]")) {
      if (input.type === "file") {
        if (!input.files.length) {
          values[input.dataset.operationField] = "";
          continue;
        }
        const form = new FormData();
        form.append("upload", input.files[0]);
        const response = await fetch("/api/operations/upload", { method: "POST", body: form });
        const uploaded = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(uploaded.detail || "file upload failed");
        values[input.dataset.operationField] = uploaded.path;
      } else {
        values[input.dataset.operationField] = input.value;
      }
    }
    const job = await fetchJson("/api/operations", {
      method: "POST",
      body: JSON.stringify({ operation: element("operation-select").value, values }),
    });
    renderOperationJob(job);
    window.clearInterval(state.operationTimer);
    state.operationTimer = window.setInterval(refreshOperation, 500);
  } catch (error) {
    element("operation-state").textContent = error.message;
  }
}

async function refreshOperation() {
  const result = await fetchJson("/api/operations");
  renderOperationJob(result.job);
  if (!result.job.running) {
    window.clearInterval(state.operationTimer);
  }
}

function renderOperationJob(job) {
  element("operation-state").textContent = job.running ? "Running" : job.return_code === null ? "Ready" : `Exit ${job.return_code}`;
  element("operation-output").textContent = (job.lines || []).join("\n");
  element("operation-run").disabled = job.running;
  element("operation-stop").disabled = !job.running;
}

async function loadLicense() {
  try {
    const result = await fetchJson("/api/license");
    element("license-version").textContent = `Version ${result.version}`;
    element("license-text").textContent = result.license || "License text unavailable";
    renderTable(
      element("dependency-licenses"),
      ["Package", "Version", "Declared license"],
      result.dependencies.map((dependency) => [dependency.name, dependency.version, dependency.license]),
    );
  } catch (error) {
    element("license-text").textContent = error.message;
  }
}

function renderTable(container, headers, rows) {
  const table = document.createElement("table");
  const heading = document.createElement("tr");
  headers.forEach((header) => {
    const cell = document.createElement("th");
    cell.textContent = header;
    heading.appendChild(cell);
  });
  table.appendChild(heading);
  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    row.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value === null ? "" : String(value);
      tableRow.appendChild(cell);
    });
    table.appendChild(tableRow);
  });
  container.replaceChildren(table);
}

// -- wiring -----------------------------------------------------------------

async function main() {
  view.status = element("status");
  view.state = element("turn-state");
  view.session = element("session-id");
  view.mic = element("mic");
  view.micRate = element("mic-rate");
  view.level = element("level");
  view.language = element("language");
  view.mode = element("mode");
  view.routing = element("routing");
  view.performanceMode = element("performance-mode");
  view.privacyWarning = element("privacy-warning");
  view.talk = element("talk");
  view.connection = element("connection");
  view.target = element("target");
  view.source = element("source");
  view.translation = element("translation");
  view.history = element("history");
  view.logs = element("logs");
  view.logFilter = element("log-filter");
  view.logFollow = element("log-follow");
  view.textRow = element("text-row");
  view.textInput = element("text-input");
  view.stages = {};
  STAGES.forEach((stage) => {
    view.stages[stage] = element(`stage-${stage}`);
  });

  view.talk.addEventListener("click", toggleTalk);
  element("mode-button").addEventListener("click", cycleMode);
  element("clear-turn").addEventListener("click", clearCurrentTurn);
  element("recent-turns-visible").addEventListener("change", updateRecentTurnsVisibility);
  view.connection.addEventListener("click", toggleConnection);
  element("enable-audio").addEventListener("click", async () => {
    try {
      await startCapture();
      element("enable-audio").disabled = true;
    } catch (error) {
      setStatus(`microphone denied: ${error.message}`, "error");
    }
  });
  element("refresh-audio").addEventListener("click", async () => {
    try {
      await refreshAudioDevices();
    } catch (error) {
      setStatus(`device refresh failed: ${error.message}`, "error");
    }
  });
  element("test-audio").addEventListener("click", testAudioDevices);
  element("input-device").addEventListener("change", () => {
    if (state.audioContext) {
      stopCapture();
      element("enable-audio").disabled = false;
      setStatus("input device changed; enable the microphone again", "ok");
    }
  });
  element("output-device").addEventListener("change", () => {
    if (state.playbackContext) {
      applyOutputDevice(state.playbackContext);
    }
  });
  view.target.addEventListener("change", () => {
    send({ type: "target", language: view.target.value });
  });
  element("text-send").addEventListener("click", () => {
    const text = view.textInput.value.trim();
    if (text) {
      send({ type: "text", text });
      view.textInput.value = "";
    }
  });
  element("log-pause").addEventListener("click", (event) => {
    state.logsPaused = !state.logsPaused;
    event.target.textContent = state.logsPaused ? "Resume" : "Pause";
  });
  element("log-clear").addEventListener("click", () => {
    view.logs.innerHTML = "";
  });
  document.querySelectorAll("#navigation button").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  element("theme-select").addEventListener("change", (event) => setTheme(event.target.value));
  element("monitor-refresh").addEventListener("click", loadMonitoring);
  element("monitor-window").addEventListener("change", loadMonitoring);
  window.addEventListener("resize", () => {
    window.clearTimeout(state.monitoringResizeTimer);
    state.monitoringResizeTimer = window.setTimeout(() => {
      if (state.monitoring && !element("monitoring").hidden) {
        renderMonitoringCharts(state.monitoring);
      }
    }, 100);
  });
  element("config-save").addEventListener("click", saveConfiguration);
  element("config-reload").addEventListener("click", loadConfiguration);
  element("config-target").addEventListener("change", loadConfiguration);
  element("history-reload").addEventListener("click", loadHistoryArchive);
  element("history-search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      state.historyOffset = 0;
      loadHistoryArchive();
    }
  });
  element("history-language").addEventListener("change", () => {
    state.historyOffset = 0;
    loadHistoryArchive();
  });
  element("history-outcome").addEventListener("change", () => {
    state.historyOffset = 0;
    loadHistoryArchive();
  });
  element("history-previous").addEventListener("click", () => {
    state.historyOffset = Math.max(0, state.historyOffset - state.historyLimit);
    loadHistoryArchive();
  });
  element("history-next").addEventListener("click", () => {
    if (state.historyOffset + state.historyLimit < state.historyTotal) {
      state.historyOffset += state.historyLimit;
      loadHistoryArchive();
    }
  });
  element("history-export").addEventListener("click", exportHistoryArchive);
  element("history-clear").addEventListener("click", clearHistoryArchive);
  element("operation-select").addEventListener("change", renderOperation);
  element("operation-run").addEventListener("click", runOperation);
  element("operation-stop").addEventListener("click", async () => {
    renderOperationJob(await fetchJson("/api/operations", { method: "DELETE" }));
  });
  element("operation-clear").addEventListener("click", () => {
    element("operation-output").textContent = "";
  });

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest("input, select, textarea, button")) {
      if (event.key === "Escape" && state.mode === "text") {
        event.target.blur();
        toggleTextMode();
      }
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      toggleTalk();
    } else if (event.key === "a") {
      cycleMode();
    } else if (event.key === "r") {
      cycleTargetLanguage();
    } else if (event.key === "c") {
      clearCurrentTurn();
    } else if (event.key === "h") {
      element("recent-turns-visible").checked = !element("recent-turns-visible").checked;
      updateRecentTurnsVisibility();
    } else if (event.key === "t") {
      toggleTextMode();
    } else if (event.key === "q") {
      toggleConnection();
    }
  });

  resetStages();
  renderTurnDiagnostics({});
  renderServiceStatus();
  updateTalkButton();
  setTheme(localStorage.getItem("kotonoha-theme") || "system");
  try {
    await loadInterface();
  } catch (error) {
    setStatus(`localization unavailable: ${error.message}`, "error");
  }
  refreshAudioDevices().catch(() => {});
  connect();
}

document.addEventListener("DOMContentLoaded", main);
