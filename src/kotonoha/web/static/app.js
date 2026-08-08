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
  session: null,
  logsPaused: false,
  configuration: null,
  operations: null,
  operationTimer: null,
};

const view = {};

function element(id) {
  return document.getElementById(id);
}

function setStatus(text, kind) {
  view.status.textContent = text;
  view.status.dataset.kind = kind || "";
}

// -- transport --------------------------------------------------------------

function send(message) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message));
  }
}

function connect() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws`);
  socket.binaryType = "arraybuffer";
  state.socket = socket;

  socket.onopen = () => setStatus("connected", "ok");
  socket.onclose = () => {
    setStatus("disconnected", "error");
    stopCapture();
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
      view.session.textContent = message.session;
      fillLanguages(message.languages, message.target);
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
  view.mic.textContent = enabled ? "OPEN" : "SHUT";
  view.mic.dataset.open = String(enabled);
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
  view.talk.textContent = state.talking ? "Stop (space)" : "Talk (space)";
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
  send({ type: "mode", mode: state.mode });
  updateModeView();
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
  while (view.history.childElementCount > 50) {
    view.history.lastElementChild.remove();
  }
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

// -- configuration ---------------------------------------------------------

async function loadConfiguration() {
  try {
    state.configuration = await fetchJson("/api/config");
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
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = section.label;
    button.dataset.active = String(sectionIndex === 0);
    button.addEventListener("click", () => {
      sectionNavigation.querySelectorAll("button").forEach((item) => {
        item.dataset.active = String(item === button);
      });
      fieldContainer.querySelectorAll(".settings-section").forEach((panel) => {
        panel.hidden = panel.dataset.section !== section.name;
      });
    });
    sectionNavigation.appendChild(button);

    const panel = document.createElement("section");
    panel.className = "settings-section";
    panel.dataset.section = section.name;
    panel.hidden = sectionIndex !== 0;
    const heading = document.createElement("h2");
    heading.textContent = section.label;
    panel.appendChild(heading);
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
      body: JSON.stringify({ changes }),
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
  try {
    const result = await fetchJson(`/api/history?${parameters}`);
    fillFilter(element("history-language"), result.languages, language, "All languages");
    fillFilter(element("history-outcome"), result.outcomes, outcome, "All outcomes");
    renderTable(
      element("history-results"),
      ["Time", "Direction", "Source", "Translation", "Outcome"],
      result.entries.map((entry) => [
        entry.time,
        `${entry.src_lang || "?"}→${entry.tgt_lang || "?"}`,
        entry.source_text || "",
        entry.translation || "",
        entry.outcome,
      ]),
    );
    element("history-status").textContent = `${result.total} turns`;
  } catch (error) {
    element("history-status").textContent = error.message;
  }
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
    option.textContent = operation.name;
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
    input.type = "text";
    input.dataset.operationField = name;
    input.value = defaults[name] || "";
    label.appendChild(input);
    fields.appendChild(label);
  });
}

async function runOperation() {
  const values = {};
  document.querySelectorAll("[data-operation-field]").forEach((input) => {
    values[input.dataset.operationField] = input.value;
  });
  try {
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

function main() {
  view.status = element("status");
  view.state = element("turn-state");
  view.session = element("session-id");
  view.mic = element("mic");
  view.micRate = element("mic-rate");
  view.level = element("level");
  view.language = element("language");
  view.mode = element("mode");
  view.talk = element("talk");
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
  element("config-save").addEventListener("click", saveConfiguration);
  element("history-reload").addEventListener("click", loadHistoryArchive);
  element("history-search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadHistoryArchive();
  });
  element("history-clear").addEventListener("click", clearHistoryArchive);
  element("operation-select").addEventListener("change", renderOperation);
  element("operation-run").addEventListener("click", runOperation);
  element("operation-stop").addEventListener("click", async () => {
    renderOperationJob(await fetchJson("/api/operations", { method: "DELETE" }));
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.closest("input, select, textarea, button")) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      toggleTalk();
    } else if (event.key === "a") {
      cycleMode();
    }
  });

  resetStages();
  updateTalkButton();
  setTheme(localStorage.getItem("kotonoha-theme") || "system");
  refreshAudioDevices().catch(() => {});
  connect();
}

document.addEventListener("DOMContentLoaded", main);
