from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def _cuda_cache_stats(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


def _maybe_compact_cuda_cache(device: torch.device, *, reason: str) -> None:
    """Release large stale CUDA cache blocks while keeping model weights resident.

    `empty_cache()` does not free live tensors, but calling it after every small
    encoder pass can add allocator churn. This only runs after full translation
    calls and only when the reserved-but-unused cache has grown enough to matter.
    """
    allocated, reserved = _cuda_cache_stats(device)
    cached = reserved - allocated
    if cached <= 512 * 1024 * 1024:
        return
    if reserved <= max(allocated * 2, allocated + 1024 * 1024 * 1024):
        return
    log.debug(
        "compacting CUDA cache after %s (allocated=%.1fMB reserved=%.1fMB)",
        reason,
        allocated / 1024 / 1024,
        reserved / 1024 / 1024,
    )
    torch.cuda.empty_cache()


@dataclass
class TranslationResult:
    audio: np.ndarray            # 16kHz mono float32
    sample_rate: int             # always 16000 for SeamlessM4T-v2 vocoder
    text: str | None             # intermediate text (target language)


def _remap_ids_for_decode(model, ids: torch.Tensor) -> torch.Tensor:
    """Map ids back to the original tokenizer's vocab if the model was vocab-pruned."""
    remap = getattr(model, "_vocab_remap_to_old", None)
    if remap is None:
        return ids
    ids = ids.clone()
    mask = (ids >= 0) & (ids < len(remap))
    ids[mask] = remap[ids[mask]]
    return ids


class SeamlessTranslator:
    """Wraps the pruned SeamlessM4T-v2 speech-to-speech model.

    Input:  16kHz mono float32 PCM (English speech).
    Output: 16kHz mono float32 PCM (Hindi speech) + intermediate text.
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_path: str | Path, device: str | None = None,
                 tgt_lang: str = "hin", speaker_id: int = 0):
        from transformers import SeamlessM4TProcessor, SeamlessM4Tv2ForSpeechToSpeech

        self.model_path = Path(model_path)
        self.tgt_lang = tgt_lang
        self.speaker_id = speaker_id

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._inference_lock = threading.RLock()

        log.info("loading processor from %s", self.model_path)
        self.processor = SeamlessM4TProcessor.from_pretrained(str(self.model_path))

        log.info("loading model on %s (dtype=%s)", self.device, self.dtype)
        self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype
        ).to(self.device).eval()
        self.model._duplex_inference_lock = self._inference_lock

        # Restore vocab remap saved during pruning, used to map intermediate ids
        # back to the tokenizer's original ids before decoding.
        custom_path = self.model_path / "_custom_state.pt"
        if custom_path.exists():
            try:
                cs = torch.load(str(custom_path), map_location="cpu", weights_only=True)
                if isinstance(cs, dict) and "_vocab_remap_to_old" in cs:
                    self.model._vocab_remap_to_old = cs["_vocab_remap_to_old"]
                    log.info("attached vocab remap (size=%d)", len(cs["_vocab_remap_to_old"]))
            except Exception as e:
                log.warning("could not load _custom_state.pt: %s", e)

    def translate(self, audio_f32: np.ndarray, tgt_lang: str | None = None) -> TranslationResult:
        if audio_f32.ndim != 1:
            audio_f32 = audio_f32.reshape(-1)
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)

        generate_lang = tgt_lang if tgt_lang else self.tgt_lang
        wav_np = np.zeros(0, dtype=np.float32)
        text: str | None = None
        inputs = None
        out = None
        wav = None
        seq = None
        ids = None

        try:
            with self._inference_lock, torch.inference_mode():
                inputs = self.processor(
                    audio=audio_f32,
                    sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                if self.dtype == torch.float16 and "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].half()

                out = self.model.generate(
                    **inputs,
                    tgt_lang=generate_lang,
                    speaker_id=self.speaker_id,
                    return_intermediate_token_ids=True,
                )

                # Audio
                wav = getattr(out, "waveform", None)
                if wav is None and isinstance(out, (tuple, list)):
                    wav = out[0]
                if wav is not None:
                    wav_np = wav.detach().to(torch.float32).cpu().numpy().squeeze()
                    if wav_np.ndim == 0:
                        wav_np = wav_np.reshape(1)
                    wav_np = wav_np.astype(np.float32, copy=False)

                # Intermediate text (best-effort)
                seq = getattr(out, "sequences", None)
                if seq is not None:
                    try:
                        ids = _remap_ids_for_decode(self.model, seq.detach().cpu())
                        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
                    except Exception as e:
                        log.debug("text decode failed: %s", e)
        finally:
            del ids, seq, wav, out, inputs
            _maybe_compact_cuda_cache(self.device, reason="translation")

        return TranslationResult(audio=wav_np, sample_rate=self.SAMPLE_RATE, text=text)

    def warmup(self) -> None:
        log.info("warming up translator")
        try:
            _ = self.translate(np.zeros(self.SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            log.warning("warmup failed (non-fatal): %s", e)
