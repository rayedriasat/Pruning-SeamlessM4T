"""CIF Boundary Detector adapter — voice-activity head on SeamlessM4T's encoder.

Mirrors the head architecture trained alongside the pruned model (see stream.py):
a 3-layer MLP over per-frame 1024-dim encoder hidden states with a sigmoid output.

A frame is treated as **silence** when the sigmoid output is below
`silence_threshold` (0.2 in the reference implementation).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CIFBoundaryDetector(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BoundaryVAD:
    """Runs the speech encoder + boundary head and returns per-frame probs."""

    def __init__(self, base_model, processor, adapter_path: str | Path,
                 device: str | torch.device = "cpu"):
        self.base_model = base_model
        self.processor = processor
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._inference_lock = getattr(base_model, "_duplex_inference_lock", threading.RLock())

        self.adapter = CIFBoundaryDetector().to(self.device).eval()
        state = torch.load(str(adapter_path), map_location=self.device, weights_only=True)
        self.adapter.load_state_dict(state)
        for p in self.adapter.parameters():
            p.requires_grad = False
        log.info("loaded boundary adapter from %s", adapter_path)

    def probs(self, audio_f32: np.ndarray) -> np.ndarray:
        """Return per-encoder-frame VAD probabilities for `audio_f32` (16kHz mono)."""
        if audio_f32.ndim != 1:
            audio_f32 = audio_f32.reshape(-1)
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)

        inputs = None
        encoder_out = None
        hidden = None
        probs = None
        out = np.zeros(0, dtype=np.float32)
        try:
            with self._inference_lock, torch.inference_mode():
                inputs = self.processor(
                    audio=audio_f32,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                if getattr(self.base_model, "dtype", torch.float32) == torch.float16 and "input_features" in inputs:
                    inputs["input_features"] = inputs["input_features"].half()

                encoder_out = self.base_model.speech_encoder(
                    **inputs,
                    return_dict=True,
                )
                hidden = encoder_out.last_hidden_state.float()  # (1, T, 1024)
                probs = self.adapter(hidden).squeeze(-1)        # (1, T)
                out = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)
        finally:
            del probs, hidden, encoder_out, inputs
        return out
