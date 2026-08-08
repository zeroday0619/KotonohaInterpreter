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
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
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
  }
  return state.playbackContext;
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

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") {
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
  connect();
}

document.addEventListener("DOMContentLoaded", main);
