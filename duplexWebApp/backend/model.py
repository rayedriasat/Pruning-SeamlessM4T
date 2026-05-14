from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


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

        log.info("loading processor from %s", self.model_path)
        self.processor = SeamlessM4TProcessor.from_pretrained(str(self.model_path))

        log.info("loading model on %s (dtype=%s)", self.device, self.dtype)
        self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype
        ).to(self.device).eval()

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

    @torch.no_grad()
    def translate(self, audio_f32: np.ndarray, tgt_lang: str | None = None) -> TranslationResult:
        if audio_f32.ndim != 1:
            audio_f32 = audio_f32.reshape(-1)
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)

        inputs = self.processor(
            audio=audio_f32,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == torch.float16 and "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].half()

        generate_lang = tgt_lang if tgt_lang else self.tgt_lang

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
        if wav is None:
            wav_np = np.zeros(0, dtype=np.float32)
        else:
            wav_np = wav.detach().to(torch.float32).cpu().numpy().squeeze()
            if wav_np.ndim == 0:
                wav_np = wav_np.reshape(1)
            wav_np = wav_np.astype(np.float32, copy=False)

        # Intermediate text (best-effort)
        text: str | None = None
        seq = getattr(out, "sequences", None)
        if seq is not None:
            try:
                ids = _remap_ids_for_decode(self.model, seq.detach().cpu())
                text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            except Exception as e:
                log.debug("text decode failed: %s", e)

        return TranslationResult(audio=wav_np, sample_rate=self.SAMPLE_RATE, text=text)

    def warmup(self) -> None:
        log.info("warming up translator")
        try:
            _ = self.translate(np.zeros(self.SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            log.warning("warmup failed (non-fatal): %s", e)
