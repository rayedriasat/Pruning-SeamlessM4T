from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .boundary_vad import BoundaryVAD
from .model import SeamlessTranslator

log = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    sample_rate: int = 16000
    # Boundary VAD
    silence_threshold: float = 0.2          # prob below this is silence (matches stream.py)
    silence_patience_sec: float = 0.8       # trailing silence before utterance is "done"
    # Speech-start detection — sustained recent voicing required, NOT "any speech in buffer".
    # The longer "while bot is speaking" value protects against AEC residuals causing self-barge-in.
    speech_start_sec: float = 0.32          # ~4 frames @ 80ms; latency for fresh user speech
    speech_start_sec_during_bot: float = 0.64  # ~8 frames; stricter when bot is on
    # Buffering / cadence
    buffer_duration_sec: float = 30.0       # rolling cap on the accumulator
    min_chunk_sec: float = 0.5              # minimum utterance length to dispatch
    check_interval_sec: float = 0.20        # how often the boundary task evaluates
    min_new_audio_sec: float = 0.15         # require this much fresh audio between checks
    # Output streaming
    output_chunk_ms: int = 100              # bot audio chunk size sent to the browser


class Session:
    """Per-WebSocket conversation state machine using the trained boundary adapter.

    Wire format
    -----------
    Client -> Server:
      - Binary: int16 LE PCM @ 16kHz, any size; the server accumulates into a rolling buffer.
      - Text JSON: {"type":"reset"} clears the current utterance and any in-flight generation.

    Server -> Client:
      - Text JSON events:
          ready
          user_speech_start
          user_speech_end {duration, too_short}
          thinking
          speaking_start  {sample_rate}
          speaking_text   {text}
          speaking_end
          speaking_cancelled
          stop_playback
          error           {message}
      - Binary: int16 LE PCM @ 16kHz, only between speaking_start and speaking_end.
    """

    def __init__(self, ws: WebSocket, translator: SeamlessTranslator,
                 vad: BoundaryVAD, cfg: SessionConfig | None = None,
                 tgt_lang: str = "ben"):
        self.ws = ws
        self.translator = translator
        self.vad = vad
        self.cfg = cfg or SessionConfig()
        self.tgt_lang = tgt_lang
        sr = self.cfg.sample_rate

        # Rolling audio buffer as a list of int16 numpy arrays — appending is O(1)
        # and concatenation only happens once per boundary check.
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._last_check_samples = 0

        self._max_buffer_samples = int(self.cfg.buffer_duration_sec * sr)
        self._min_chunk_samples = int(self.cfg.min_chunk_sec * sr)
        self._min_new_samples = int(self.cfg.min_new_audio_sec * sr)

        # State
        self._speech_detected = False        # any speech seen in the current buffer
        self._bot_speaking = False           # we are currently streaming output back
        self._gen_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    # ----- Lifecycle -------------------------------------------------------

    async def run(self) -> None:
        await self._send_json({"type": "ready"})
        check_task = asyncio.create_task(self._boundary_loop(), name="boundary_loop")
        try:
            while True:
                msg = await self.ws.receive()
                t = msg.get("type")
                if t == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    self._on_audio_bytes(msg["bytes"])
                elif msg.get("text") is not None:
                    await self._on_text(msg["text"])
        except WebSocketDisconnect:
            return
        finally:
            check_task.cancel()
            try:
                await check_task
            except (asyncio.CancelledError, Exception):
                pass

    async def cleanup(self) -> None:
        await self._cancel_generation(notify=False)
        self._chunks.clear()
        self._total_samples = 0
        self._last_check_samples = 0

    # ----- Inbound ---------------------------------------------------------

    def _on_audio_bytes(self, data: bytes) -> None:
        if not data:
            return
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self._chunks.append(arr)
        self._total_samples += len(arr)
        # Trim from the front only when no speech is in flight, so we don't
        # truncate a long utterance the user is still speaking.
        if not self._speech_detected and self._total_samples > self._max_buffer_samples:
            self._trim_front(self._total_samples - self._max_buffer_samples)

    async def _on_text(self, text: str) -> None:
        try:
            obj = json.loads(text)
        except Exception:
            return
        if obj.get("type") == "reset":
            self._chunks.clear()
            self._total_samples = 0
            self._last_check_samples = 0
            self._speech_detected = False
            await self._cancel_generation(notify=False)
        elif obj.get("type") == "set_lang":
            new_lang = obj.get("lang")
            if new_lang:
                self.tgt_lang = new_lang

    def _trim_front(self, drop: int) -> None:
        original_drop = drop
        while drop > 0 and self._chunks:
            head = self._chunks[0]
            if len(head) <= drop:
                self._chunks.pop(0)
                self._total_samples -= len(head)
                drop -= len(head)
            else:
                self._chunks[0] = head[drop:]
                self._total_samples -= drop
                drop = 0
        # Adjust last-check pointer so we don't accidentally re-skip work.
        self._last_check_samples = max(0, self._last_check_samples - original_drop)

    def _snapshot(self) -> np.ndarray:
        """Concatenate the current buffer; safe to hand off to a worker thread."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0].copy()
        return np.concatenate(self._chunks)

    # ----- Boundary detection loop ----------------------------------------

    async def _boundary_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.check_interval_sec)
            try:
                await self._boundary_step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("boundary loop error")

    async def _boundary_step(self) -> None:
        n = self._total_samples
        if n < self._min_chunk_samples:
            return
        # Skip if too little new audio has arrived since last check (saves encoder calls).
        if n - self._last_check_samples < self._min_new_samples:
            return

        audio = self._snapshot()
        n = len(audio)
        self._last_check_samples = n

        loop = asyncio.get_running_loop()
        probs = await loop.run_in_executor(None, self.vad.probs, audio)

        T = int(probs.shape[0])
        if T == 0:
            return

        is_silence = probs < self.cfg.silence_threshold

        sr = self.cfg.sample_rate
        frames_per_sec = T / (n / sr)

        # Trailing non-silence streak — sustained recent voicing.
        ns_streak = 0
        for i in range(T - 1, -1, -1):
            if not is_silence[i]:
                ns_streak += 1
            else:
                break

        # Trailing silence streak.
        sil_streak = 0
        for i in range(T - 1, -1, -1):
            if is_silence[i]:
                sil_streak += 1
            else:
                break

        # Speech-start: require trailing run of non-silence. While the bot is speaking,
        # use a stricter threshold so AEC residuals don't cause self-barge-in.
        start_sec = (
            self.cfg.speech_start_sec_during_bot if self._bot_speaking
            else self.cfg.speech_start_sec
        )
        start_frames = max(1, int(round(start_sec * frames_per_sec)))

        if not self._speech_detected and ns_streak >= start_frames:
            self._speech_detected = True
            if self._bot_speaking:
                await self._cancel_generation(notify=True)
            await self._send_json({"type": "user_speech_start"})

        if not self._speech_detected:
            return

        patience_frames = max(1, int(round(self.cfg.silence_patience_sec * frames_per_sec)))

        if sil_streak < patience_frames:
            return

        # Utterance ended. Slice off the speech portion (everything before the trailing silence).
        boundary_frame = T - sil_streak
        boundary_sample = max(0, int(round((boundary_frame / T) * n)))

        if boundary_sample <= self._min_chunk_samples:
            # Too short: discard whatever's in the buffer and wait for fresh speech.
            self._chunks.clear()
            self._total_samples = 0
            self._last_check_samples = 0
            self._speech_detected = False
            await self._send_json({
                "type": "user_speech_end",
                "duration": boundary_sample / sr,
                "too_short": True,
            })
            return

        utterance = audio[:boundary_sample].copy()
        # Drop the dispatched portion AND any trailing silence we just measured. Anything
        # that arrived since the snapshot (between executor return and now) stays in the
        # buffer for the next utterance — which would only ever be very recent samples.
        consumed = min(self._total_samples, n)
        self._trim_front(consumed)
        self._speech_detected = False
        self._last_check_samples = self._total_samples

        await self._send_json({
            "type": "user_speech_end",
            "duration": len(utterance) / sr,
            "too_short": False,
        })

        await self._cancel_generation(notify=False)
        self._gen_task = asyncio.create_task(self._generate_and_stream(utterance))

    # ----- Generation ------------------------------------------------------

    async def _generate_and_stream(self, audio: np.ndarray) -> None:
        result = None
        wav = None
        try:
            await self._send_json({"type": "thinking"})
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.translator.translate, audio, self.tgt_lang)
            # The utterance has been copied/processed by the worker at this point.
            # Drop the per-turn input so long replies do not keep old mic audio alive.
            del audio

            wav = result.audio
            if wav is None or len(wav) == 0:
                if result.text:
                    await self._send_json({"type": "speaking_text", "text": result.text})
                await self._send_json({"type": "speaking_end"})
                return

            sr = result.sample_rate
            # Clear any audio captured during the "thinking" gap so the boundary detector
            # only ever sees mic frames captured *after* playback starts. Without this,
            # stale silence + AEC residuals sit in the buffer and can spuriously trip
            # the speech-start check, causing the bot to barge in on itself.
            self._chunks.clear()
            self._total_samples = 0
            self._last_check_samples = 0
            self._speech_detected = False
            await self._send_json({"type": "speaking_start", "sample_rate": sr})
            if result.text:
                await self._send_json({"type": "speaking_text", "text": result.text})
            self._bot_speaking = True

            chunk = max(1, int(self.cfg.output_chunk_ms / 1000 * sr))
            for i in range(0, len(wav), chunk):
                seg = wav[i:i + chunk]
                pcm = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                await self._send_bytes(pcm)
                # Pace below real-time so a barge-in interrupts before everything's on the wire.
                await asyncio.sleep((len(seg) / sr) * 0.5)

            await self._send_json({"type": "speaking_end"})
        except asyncio.CancelledError:
            try:
                await self._send_json({"type": "speaking_cancelled"})
            except Exception:
                pass
            raise
        except Exception as e:
            log.exception("generation failed")
            try:
                await self._send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            self._bot_speaking = False
            del wav, result

    async def _cancel_generation(self, notify: bool) -> None:
        task = self._gen_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._gen_task = None
        if notify:
            await self._send_json({"type": "stop_playback"})

    # ----- Outbound (serialized) ------------------------------------------

    async def _send_json(self, obj) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(obj))
            except Exception:
                pass

    async def _send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_bytes(data)
            except Exception:
                pass
