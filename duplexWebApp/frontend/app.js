// Frontend duplex client.
//
// Capture: AudioContext @ 16kHz, AudioWorklet emits int16 PCM frames over WebSocket.
// Playback: AudioContext schedules incoming int16 PCM into a sample-accurate queue;
//           a "stop_playback" message flushes anything not yet played so barge-in is instant.

const $ = (id) => document.getElementById(id);
const langSelect = $("lang");
const startBtn = $("start");
const stopBtn = $("stop");
const resetBtn = $("reset");
const stateEl = $("state");
const dotEl = $("dot");
const meterEl = $("meter");
const logEl = $("log");
const modeRadios = Array.from(document.querySelectorAll('input[name="talkMode"]'));
const holdBtn = $("holdTalk");

let ws = null;
let captureCtx = null;
let captureNode = null;
let mediaStream = null;
let playCtx = null;
let playHead = 0; // scheduling head, in playCtx time
let scheduled = []; // active BufferSourceNodes so we can stop them
let botSampleRate = 16000;
let talkMode = "push";
let pushHeld = false;
let silenceTailFrames = 0;

const SILENCE_TAIL_FRAMES = 30; // ~960ms @ 512 samples / 16kHz, enough for server VAD.

function setState(s, cls) {
  stateEl.textContent = s;
  dotEl.className = "dot" + (cls ? " " + cls : "");
}

function connected() {
  return ws && ws.readyState === WebSocket.OPEN;
}

function isMicOpen() {
  return connected() && (talkMode === "always" || pushHeld);
}

function setTalkMode(mode) {
  const wasOpen = isMicOpen();
  talkMode = mode;
  if (talkMode === "always") {
    pushHeld = false;
    silenceTailFrames = 0;
  } else if (wasOpen) {
    silenceTailFrames = SILENCE_TAIL_FRAMES;
  }
  updateMicUi();
}

function updateMicUi() {
  const open = isMicOpen();
  document.body.classList.toggle("mic-open", open);
  holdBtn.classList.toggle("active", pushHeld);
  holdBtn.disabled = !connected() || talkMode !== "push";
  holdBtn.setAttribute("aria-pressed", String(pushHeld));
  holdBtn.title = open
    ? "Mic open"
    : !connected()
      ? "Start the call"
      : talkMode === "push"
      ? "Hold to talk"
      : "Mic paused";
  holdBtn.setAttribute("aria-label", holdBtn.title);
}

function openPushToTalk() {
  if (talkMode !== "push" || !connected() || pushHeld) return;
  pushHeld = true;
  silenceTailFrames = 0;
  updateMicUi();
}

function closePushToTalk() {
  if (talkMode !== "push" || !pushHeld) return;
  pushHeld = false;
  // Feed the server a brief silence tail so it can close the utterance without
  // needing the browser to keep streaming live mic input.
  silenceTailFrames = SILENCE_TAIL_FRAMES;
  updateMicUi();
}

function silentFrame(byteLength) {
  return new ArrayBuffer(byteLength);
}

function log(type, extra) {
  const d = document.createElement("div");
  d.className = "ev-" + type;
  const t = new Date().toLocaleTimeString();
  d.textContent = `[${t}] ${type}` + (extra ? "  " + extra : "");
  logEl.prepend(d);
  while (logEl.childElementCount > 200) logEl.removeChild(logEl.lastChild);
}

async function start() {
  startBtn.disabled = true;
  try {
    // Prefer 16kHz so capture matches the model's expected rate.
    captureCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });
    if (captureCtx.sampleRate !== 16000) {
      log(
        "warn",
        `AudioContext sample rate ${captureCtx.sampleRate}Hz, expected 16000Hz`,
      );
    }
    await captureCtx.audioWorklet.addModule("/worklet.js");

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const source = captureCtx.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(captureCtx, "capture");
    source.connect(captureNode);
    // Do NOT connect captureNode to destination — we don't want to hear ourselves.

    playCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });
    playHead = playCtx.currentTime;

    captureNode.port.onmessage = (e) => {
      const { pcm, peak } = e.data;
      if (typeof peak === "number") {
        meterEl.style.width = isMicOpen()
          ? Math.min(100, Math.round(peak * 140)) + "%"
          : "0%";
      }
      if (!connected()) return;
      if (isMicOpen()) {
        ws.send(pcm);
      } else if (silenceTailFrames > 0) {
        ws.send(silentFrame(pcm.byteLength));
        silenceTailFrames -= 1;
      }
    };

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const lang = langSelect.value;
    ws = new WebSocket(`${proto}://${location.host}/ws?lang=${lang}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setState("connecting…", "thinking");
      log("ws_open");
      stopBtn.disabled = false;
      resetBtn.disabled = false;
      updateMicUi();
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          handleControl(JSON.parse(ev.data));
        } catch (err) {
          log("error", "bad json: " + err.message);
        }
      } else {
        playPcm16(new Int16Array(ev.data));
      }
    };
    ws.onclose = () => {
      log("ws_close");
      setState("disconnected", "");
      stopAll();
    };
    ws.onerror = () => log("error", "ws error");
  } catch (err) {
    log("error", err.message || String(err));
    setState("error", "error");
    startBtn.disabled = false;
  }
}

function handleControl(m) {
  log(m.type || "msg", m.duration ? `(${m.duration.toFixed(2)}s)` : "");
  switch (m.type) {
    case "ready":
      setState(talkMode === "push" ? "ready — hold to talk" : "listening", "listening");
      break;
    case "user_speech_start":
      setState("you: speaking…", "you");
      // Cut bot playback the moment server-side detects we started speaking.
      flushPlayback();
      break;
    case "user_speech_end":
      if (m.too_short) setState("too short — listening", "listening");
      else setState("thinking…", "thinking");
      break;
    case "thinking":
      setState("thinking…", "thinking");
      break;
    case "speaking_start":
      botSampleRate = m.sample_rate || 16000;
      // Reset scheduling head so the response starts immediately, not after stale time.
      playHead = playCtx.currentTime + 0.04;
      setState("bot: speaking…", "bot");
      break;
    case "speaking_text":
      if (m.text) log("text", '"' + m.text + '"');
      break;
    case "speaking_end":
      setState(talkMode === "push" ? "ready — hold to talk" : "listening", "listening");
      break;
    case "stop_playback":
    case "speaking_cancelled":
      flushPlayback();
      setState(talkMode === "push" ? "ready — hold to talk" : "listening", "listening");
      break;
    case "error":
      log("error", m.message || "");
      setState("error", "error");
      break;
  }
}

function playPcm16(int16) {
  if (!playCtx) return;
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
  const buf = playCtx.createBuffer(1, f32.length, botSampleRate);
  buf.copyToChannel(f32, 0);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);
  const startAt = Math.max(playHead, playCtx.currentTime + 0.01);
  src.start(startAt);
  playHead = startAt + buf.duration;
  scheduled.push(src);
  src.onended = () => {
    const i = scheduled.indexOf(src);
    if (i >= 0) scheduled.splice(i, 1);
  };
}

function flushPlayback() {
  for (const s of scheduled) {
    try {
      s.stop();
    } catch (e) {
      /* already stopped */
    }
  }
  scheduled = [];
  if (playCtx) playHead = playCtx.currentTime;
}

function stopAll() {
  flushPlayback();
  if (captureNode) {
    try {
      captureNode.disconnect();
    } catch (e) {}
    captureNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (captureCtx) {
    captureCtx.close().catch(() => {});
    captureCtx = null;
  }
  if (playCtx) {
    playCtx.close().catch(() => {});
    playCtx = null;
  }
  if (ws && ws.readyState <= 1) {
    try {
      ws.close();
    } catch (e) {}
  }
  ws = null;
  pushHeld = false;
  silenceTailFrames = 0;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  resetBtn.disabled = true;
  meterEl.style.width = "0%";
  updateMicUi();
}

langSelect.addEventListener("change", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_lang", lang: langSelect.value }));
  }
});

startBtn.onclick = start;
stopBtn.onclick = () => {
  setState("stopping", "");
  stopAll();
  setState("idle", "");
};
resetBtn.onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "reset" }));
  }
  flushPlayback();
};

modeRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) setTalkMode(radio.value);
  });
});

holdBtn.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  holdBtn.setPointerCapture?.(ev.pointerId);
  openPushToTalk();
});
holdBtn.addEventListener("pointerup", closePushToTalk);
holdBtn.addEventListener("pointercancel", closePushToTalk);
holdBtn.addEventListener("lostpointercapture", closePushToTalk);
holdBtn.addEventListener("contextmenu", (ev) => ev.preventDefault());

window.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey && ev.key.toLowerCase() === "d") {
    ev.preventDefault();
    openPushToTalk();
  }
});

window.addEventListener("keyup", (ev) => {
  if (ev.key.toLowerCase() === "d") {
    ev.preventDefault();
    closePushToTalk();
  }
});

window.addEventListener("blur", closePushToTalk);
updateMicUi();
