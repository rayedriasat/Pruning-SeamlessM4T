"""
voice_cloning_module.py
=======================
Standalone zero-shot voice cloning module extracted from the S2S
Knowledge Distillation notebook (v4).

Works with ANY speech-to-speech translation model — including a pruned
SeamlessM4T — as long as that model can produce a translated waveform.

QUICK-START (with a pruned SeamlessM4T):
-----------------------------------------
    from voice_cloning_module import VoiceCloner, SpeakerEncoder

    cloner = VoiceCloner(cosyvoice_model_dir='pretrained_models/CosyVoice-300M')

    # en_audio  : np.ndarray (float32, 16 kHz) — the speaker's input
    # zh_audio  : np.ndarray — your pruned model's translated output (any SR)
    cloned = cloner.clone(
        reference_audio=en_audio,          # source of voice identity
        translated_audio=zh_audio,         # content to speak (currently unused by CosyVoice zero-shot; see NOTE below)
        ref_sr=16_000,
    )
    import soundfile as sf
    sf.write('cloned_output.wav', cloned, 16_000)

NOTE on cosine similarity = 1.0
---------------------------------
In the original notebook, `_synthesize()` calls CosyVoice with
`tts_text=''` and only passes the reference audio waveform.
This means CosyVoice reconstructs the reference speaker's voice
directly — it does NOT consume the translated token sequence at all.
That is why you see cosine similarity ≈ 1.0: it is essentially an
identity clone of the source speaker, not a true cross-lingual
voice transfer.

To do real cross-lingual voice cloning you need one of:
  A) Feed the translated TEXT (from an ASR/TTS pipeline) to CosyVoice
     instead of an empty string.
  B) Use CosyVoice's `inference_cross_lingual` method, which accepts
     a reference audio + a target-language text prompt.

Both options are shown as `clone_with_text()` and
`clone_cross_lingual()` below.

DEPENDENCIES
-------------
    pip install speechbrain librosa torch torchaudio soundfile modelscope

    # CosyVoice (cloned from GitHub, NOT the PyPI stub):
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    # then add CosyVoice/ to sys.path (done automatically below if you
    # set COSYVOICE_DIR)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Optional

import numpy as np
import torch
import librosa

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SPEAKER ENCODER  (ECAPA-TDNN via SpeechBrain)
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEncoder:
    """
    Extracts a 192-dim d-vector from raw audio using ECAPA-TDNN.

    Args:
        model_source : HuggingFace / SpeechBrain model ID or local path.
                       Default: 'speechbrain/spkrec-ecapa-voxceleb'
        device       : torch.device or 'cuda' / 'cpu' string.
                       SpeechBrain requires the 'cuda:0' form, not bare 'cuda'.
        save_dir     : where SpeechBrain caches model weights.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[torch.device | str] = None,
        save_dir: str = "/tmp/spkrec_ecapa",
    ):
        from speechbrain.inference.speaker import EncoderClassifier

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # SpeechBrain needs the explicit 'cuda:N' form
        dev_str = str(self.device)
        if dev_str == "cuda":
            dev_str = "cuda:0"

        print(f"[SpeakerEncoder] Loading {model_source} on {dev_str} ...")
        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=save_dir,
            run_opts={"device": dev_str},
        )
        print("[SpeakerEncoder] Ready.")

    @torch.no_grad()
    def extract(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        """
        Returns a (192,) float32 speaker embedding.

        Args:
            audio : raw waveform, any sampling rate
            sr    : sampling rate of `audio`
        """
        if sr != 16_000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav).squeeze().cpu().float().numpy()
        return emb

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two speaker embeddings."""
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    def free(self):
        del self.model
        torch.cuda.empty_cache()
        print("[SpeakerEncoder] Freed from GPU.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COSYVOICE LOADER  (handles path / import complexity)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_cosyvoice_importable(cosyvoice_repo_dir: Optional[str] = None):
    """
    Makes `cosyvoice` importable from the cloned GitHub repo.
    Removes any conflicting PyPI stub named 'cosyvoice' first.

    If cosyvoice_repo_dir is None the function checks whether
    `cosyvoice` is already on sys.path.
    """
    import subprocess

    # Remove the wrong PyPI package (it shadows the real repo)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "cosyvoice"],
        capture_output=True,
    )

    if cosyvoice_repo_dir is not None:
        cosy_dir = cosyvoice_repo_dir
        matcha_dir = os.path.join(cosy_dir, "third_party", "Matcha-TTS")

        if not os.path.exists(os.path.join(cosy_dir, "cosyvoice", "cli", "cosyvoice.py")):
            raise RuntimeError(
                f"CosyVoice repo not found at {cosy_dir}.\n"
                "Run:  git clone --recursive "
                "https://github.com/FunAudioLLM/CosyVoice.git"
            )

        # Clear stale imports
        for key in list(sys.modules.keys()):
            if "cosyvoice" in key or "matcha" in key:
                del sys.modules[key]

        # Remove stale entries, then re-add at front
        sys.path = [p for p in sys.path if "cosyvoice" not in p.lower() or p == cosy_dir]
        for p in [matcha_dir, cosy_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VOICE CLONER  (main API)
# ─────────────────────────────────────────────────────────────────────────────

class VoiceCloner:
    """
    Zero-shot voice cloning using CosyVoice-300M.

    Designed to be a drop-in post-processing step after any
    speech-to-speech translation model (original SeamlessM4T,
    a pruned variant, or your own student).

    Args:
        cosyvoice_model_dir  : path to the downloaded CosyVoice-300M weights
                               (e.g. 'pretrained_models/CosyVoice-300M').
                               Download with:
                                   from modelscope import snapshot_download
                                   snapshot_download('iic/CosyVoice-300M',
                                                     local_dir=cosyvoice_model_dir)
        cosyvoice_repo_dir   : path to the cloned CosyVoice GitHub repo.
                               Only needed if the repo is not already on sys.path.
    """

    def __init__(
        self,
        cosyvoice_model_dir: str = "pretrained_models/CosyVoice-300M",
        cosyvoice_repo_dir: Optional[str] = None,
    ):
        _ensure_cosyvoice_importable(cosyvoice_repo_dir)
        from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore

        print(f"[VoiceCloner] Loading CosyVoice from {cosyvoice_model_dir} ...")
        self._model = CosyVoice(cosyvoice_model_dir)
        print("[VoiceCloner] Ready.")

    # ------------------------------------------------------------------
    # A) ZERO-SHOT CLONE  (replicates original notebook behaviour)
    # ------------------------------------------------------------------
    def clone(
        self,
        reference_audio: np.ndarray,
        translated_audio: Optional[np.ndarray] = None,  # currently informational only
        ref_sr: int = 16_000,
    ) -> np.ndarray:
        """
        Clones the voice from `reference_audio` using CosyVoice zero-shot mode.

        ⚠️  WARNING: As in the original notebook, this method calls
        `inference_zero_shot` with `tts_text=''`, so the `translated_audio`
        argument is accepted for API compatibility but is NOT consumed by
        CosyVoice. The output will reproduce the reference speaker's voice
        only. Use `clone_with_text()` for actual cross-lingual synthesis.

        Args:
            reference_audio : source speaker waveform (float32, 16 kHz)
            translated_audio: the translated waveform from your S2S model
                              (unused here, kept for signature compatibility)
            ref_sr          : sampling rate of reference_audio

        Returns:
            Cloned waveform as float32 numpy array at 16 kHz.
        """
        if ref_sr != 16_000:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=ref_sr, target_sr=16_000
            )

        ref_wav = torch.tensor(reference_audio, dtype=torch.float32).unsqueeze(0)
        t0 = time.time()
        out = self._model.inference_zero_shot(
            tts_text="",
            prompt_speech_16k=ref_wav,
            stream=False,
        )
        print(f"[VoiceCloner.clone] Done in {time.time()-t0:.2f}s")
        return out["tts_speech"].squeeze().numpy()

    # ------------------------------------------------------------------
    # B) CLONE WITH TARGET TEXT  (recommended for real cross-lingual use)
    # ------------------------------------------------------------------
    def clone_with_text(
        self,
        reference_audio: np.ndarray,
        target_text: str,
        prompt_text: str = "",
        ref_sr: int = 16_000,
    ) -> np.ndarray:
        """
        Zero-shot voice cloning with actual target text content.

        This is what you SHOULD use if you have an ASR model that can
        transcribe the translated audio, or if your S2S model produces text.

        Args:
            reference_audio : source speaker waveform (float32, 16 kHz)
            target_text     : text to be spoken in the cloned voice
                              (e.g. the Chinese transcription of the
                               translated audio from SeamlessM4T)
            prompt_text     : optional transcription of the reference audio
                              (improves prosody when provided)
            ref_sr          : sampling rate of reference_audio

        Returns:
            Synthesised waveform as float32 numpy array at 16 kHz.
        """
        if ref_sr != 16_000:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=ref_sr, target_sr=16_000
            )

        ref_wav = torch.tensor(reference_audio, dtype=torch.float32).unsqueeze(0)
        t0 = time.time()
        out = self._model.inference_zero_shot(
            tts_text=target_text,
            prompt_text=prompt_text,
            prompt_speech_16k=ref_wav,
            stream=False,
        )
        print(f"[VoiceCloner.clone_with_text] Done in {time.time()-t0:.2f}s")
        return out["tts_speech"].squeeze().numpy()

    # ------------------------------------------------------------------
    # C) CROSS-LINGUAL CLONE  (best for EN→ZH with CosyVoice-300M)
    # ------------------------------------------------------------------
    def clone_cross_lingual(
        self,
        reference_audio: np.ndarray,
        target_text: str,
        ref_sr: int = 16_000,
    ) -> np.ndarray:
        """
        Cross-lingual voice cloning: reference audio from one language,
        target text in another language.

        Requires CosyVoice-300M (supports cross-lingual synthesis).

        Args:
            reference_audio : English source speaker waveform
            target_text     : Chinese (or other target language) text
            ref_sr          : sampling rate of reference_audio

        Returns:
            Synthesised waveform as float32 numpy array at 22050 Hz or 16 kHz
            (depends on CosyVoice model).
        """
        if ref_sr != 16_000:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=ref_sr, target_sr=16_000
            )

        ref_wav = torch.tensor(reference_audio, dtype=torch.float32).unsqueeze(0)
        t0 = time.time()
        out = self._model.inference_cross_lingual(
            tts_text=target_text,
            prompt_speech_16k=ref_wav,
            stream=False,
        )
        print(f"[VoiceCloner.clone_cross_lingual] Done in {time.time()-t0:.2f}s")
        return out["tts_speech"].squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEAMLESSM4T ADAPTER  (bridges pruned model → VoiceCloner)
# ─────────────────────────────────────────────────────────────────────────────

class SeamlessVoiceCloningPipeline:
    """
    End-to-end pipeline: pruned SeamlessM4T  →  CosyVoice voice cloning.

    Usage:
        pipeline = SeamlessVoiceCloningPipeline(
            seamless_model=your_pruned_model,       # HuggingFace model
            seamless_processor=your_processor,       # AutoProcessor
            cosyvoice_model_dir='pretrained_models/CosyVoice-300M',
            cosyvoice_repo_dir='/path/to/CosyVoice',
        )
        cloned_zh = pipeline.translate_and_clone(en_audio_np, sr=16_000)

    The pipeline:
      1. Runs your pruned SeamlessM4T to get a Chinese waveform.
      2. Passes the original English audio + Chinese waveform to VoiceCloner.
      3. Returns a Chinese waveform in the original speaker's voice.
    """

    def __init__(
        self,
        seamless_model,
        seamless_processor,
        cosyvoice_model_dir: str = "pretrained_models/CosyVoice-300M",
        cosyvoice_repo_dir: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        tgt_lang: str = "cmn",
    ):
        self.model     = seamless_model
        self.processor = seamless_processor
        self.tgt_lang  = tgt_lang
        self.device    = torch.device(device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ))
        self.model.to(self.device).eval()

        self.cloner = VoiceCloner(
            cosyvoice_model_dir=cosyvoice_model_dir,
            cosyvoice_repo_dir=cosyvoice_repo_dir,
        )
        self.spk_enc = SpeakerEncoder(device=self.device)

    @torch.no_grad()
    def _run_seamless(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        """Runs the (pruned) SeamlessM4T model on one utterance."""
        inp = self.processor(
            audio=audio, src_lang="eng",
            sampling_rate=sr, return_tensors="pt"
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model.generate(**inp, tgt_lang=self.tgt_lang)

        if isinstance(out, (tuple, list)):
            wav = out[0]
        elif hasattr(out, "waveform"):
            wav = out.waveform
        else:
            wav = out
        return wav.squeeze().cpu().float().numpy()

    def translate_and_clone(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        mode: str = "zero_shot",
        target_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Translates English audio to Chinese using the pruned SeamlessM4T
        and then clones the original speaker's voice.

        Args:
            audio       : English waveform (float32)
            sr          : sampling rate
            mode        : one of
                          'zero_shot'     — clone voice, empty text (notebook default)
                          'with_text'     — clone voice + provide target_text
                          'cross_lingual' — cross-lingual mode (needs target_text)
            target_text : Chinese text for 'with_text' / 'cross_lingual' modes.
                          Obtain this by running an ASR model on the SeamlessM4T output.

        Returns:
            Cloned Chinese waveform (float32, 16 kHz).
        """
        print("[Pipeline] Running SeamlessM4T translation ...")
        zh_audio = self._run_seamless(audio, sr)

        if mode == "zero_shot":
            return self.cloner.clone(
                reference_audio=audio,
                translated_audio=zh_audio,
                ref_sr=sr,
            )
        elif mode == "with_text":
            if not target_text:
                raise ValueError("target_text is required for mode='with_text'")
            return self.cloner.clone_with_text(
                reference_audio=audio,
                target_text=target_text,
                ref_sr=sr,
            )
        elif mode == "cross_lingual":
            if not target_text:
                raise ValueError("target_text is required for mode='cross_lingual'")
            return self.cloner.clone_cross_lingual(
                reference_audio=audio,
                target_text=target_text,
                ref_sr=sr,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}. "
                             "Choose 'zero_shot', 'with_text', or 'cross_lingual'.")

    def speaker_similarity(
        self, ref_audio: np.ndarray, syn_audio: np.ndarray, sr: int = 16_000
    ) -> float:
        """Computes cosine similarity between source and cloned speaker embeddings."""
        emb_ref = self.spk_enc.extract(ref_audio, sr)
        emb_syn = self.spk_enc.extract(syn_audio, sr)
        sim = self.spk_enc.cosine_similarity(emb_ref, emb_syn)
        print(f"[Pipeline] Speaker cosine similarity: {sim:.4f}")
        return sim


# ─────────────────────────────────────────────────────────────────────────────
# 5.  QUICK-TEST  (run this file directly to verify imports)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, soundfile as sf

    parser = argparse.ArgumentParser(
        description="Voice cloning smoke-test: clone a WAV file."
    )
    parser.add_argument("--input",  required=True, help="Input English WAV file")
    parser.add_argument("--output", default="cloned_output.wav")
    parser.add_argument("--cosyvoice_dir",  default="pretrained_models/CosyVoice-300M")
    parser.add_argument("--cosyvoice_repo", default=None,
                        help="Path to cloned CosyVoice GitHub repo")
    args = parser.parse_args()

    audio, sr = sf.read(args.input)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    cloner = VoiceCloner(
        cosyvoice_model_dir=args.cosyvoice_dir,
        cosyvoice_repo_dir=args.cosyvoice_repo,
    )
    result = cloner.clone(reference_audio=audio, ref_sr=sr)
    sf.write(args.output, result, 16_000)
    print(f"Saved: {args.output}")

    # Optional: measure speaker similarity
    enc = SpeakerEncoder()
    sim = enc.cosine_similarity(enc.extract(audio, sr), enc.extract(result, 16_000))
    print(f"Speaker cosine similarity (original vs clone): {sim:.4f}")
