# Duplex Conversation — SeamlessM4T (pruned, DoRA-merged) + boundary adapter

Full-duplex voice pipeline:

- **Browser** captures the mic at 16 kHz via `AudioWorklet`, streams int16 PCM over a WebSocket.
- **FastAPI backend** runs the **trained CIF boundary adapter** (`models2/boundary_adapter.pt`) on top of SeamlessM4T's speech encoder to detect when you've stopped speaking. A frame is treated as silence when the adapter's sigmoid output drops below `silence_threshold` (0.2 by default), matching `stream.py`'s semantics.
- On utterance end, it runs the **pruned SeamlessM4T-v2 speech-to-speech** model (`models2/phase7_final_merged`, eng → ben) and decodes both the waveform and intermediate text (using the saved `_vocab_remap_to_old`).
- Translated audio streams back in 100 ms chunks. If you start speaking again while the bot is talking, the server cancels the in-flight generation and tells the browser to **flush playback** for instant barge-in.

The speech encoder is **shared** between the boundary adapter and the translator — no double-load.

## Layout

```
duplex_nihal/
├── backend/
│   ├── app.py            # FastAPI + WebSocket route + lifespan
│   ├── session.py        # Per-connection state machine (rolling buffer + boundary loop)
│   ├── boundary_vad.py   # CIFBoundaryDetector head over SeamlessM4T speech encoder
│   └── model.py          # SeamlessM4T-v2 wrapper (audio + intermediate text)
├── frontend/
│   ├── index.html
│   ├── app.js            # WS client + scheduled playback queue
│   └── worklet.js        # Mic → 512-sample int16 frames
├── models2/
│   ├── boundary_adapter.pt          # trained CIF boundary head
│   └── phase7_final_merged/       # pruned/merged SeamlessM4T-v2
├── requirements.txt
└── run.sh
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-matched PyTorch wheel first (see https://pytorch.org/get-started/locally/) — it'll be picked up automatically.

## Run

```bash
./run.sh
# then open http://localhost:8000
```

Click **Start**, allow the mic, and speak. The status pill shows the pipeline state and the log shows decoded Bengali text per turn.

Override paths via env:
```
MODEL_PATH=/path/to/phase7_final_merged ADAPTER_PATH=/path/to/boundary_adapter.pt ./run.sh
```

## Tuning

Edit `SessionConfig` at the top of [backend/session.py](backend/session.py):

| Field | Meaning |
| --- | --- |
| `silence_threshold`     | Boundary-head probability below which a frame is silence. From `stream.py`: `0.2`. |
| `silence_patience_sec`  | Trailing silence required to end an utterance. Lower = snappier, higher = more patient. |
| `min_chunk_sec`         | Drop utterances shorter than this. |
| `buffer_duration_sec`   | Cap on the rolling accumulator while idle. |
| `check_interval_sec`    | Cadence of the boundary detector loop. |
| `min_new_audio_sec`     | Min fresh audio between encoder calls (saves compute). |
| `output_chunk_ms`       | Bot audio chunk size streamed to the browser. |

## Wire protocol

Client → server:
- Binary frames: int16 LE PCM @ 16 kHz, any size — server accumulates into a rolling buffer.
- Text JSON: `{"type":"reset"}` clears the current utterance and any in-flight generation.

Server → client:
- Text JSON events: `ready`, `user_speech_start`, `user_speech_end` (`duration`/`too_short`), `thinking`, `speaking_start` (`sample_rate`), `speaking_text` (`text`), `speaking_end`, `speaking_cancelled`, `stop_playback`, `error` (`message`).
- Binary frames: int16 LE PCM @ 16 kHz, only between `speaking_start` and `speaking_end`.

## Notes

- Use **headphones**. Browser AEC suppresses most loopback, but a loud speaker can still trip the boundary head into thinking the bot's voice is yours and barge in on itself.
- The SeamlessM4T speech encoder runs once per boundary check and the result is reused — no duplicate passes when an utterance ends and translation kicks off (transformers internally re-encodes during `generate()`; if you want to feed encoder outputs directly into `generate(encoder_outputs=…)` to skip that, see `SeamlessM4Tv2ForSpeechToSpeech.generate`).
- Translation runs in a thread-pool executor so the WebSocket loop stays responsive and cancellation works.
