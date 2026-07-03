"""faster-whisper wrapper. Model loads once at startup (takes a few seconds)."""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


def _register_cuda_dlls():
    """ctranslate2 needs cublas/cudnn DLLs; pip's nvidia-* wheels ship them
    but Windows won't find them without add_dll_directory."""
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn"):
        bin_dir = base / sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 uses plain LoadLibrary, which searches PATH not dll dirs
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()


class Transcriber:
    def __init__(self, whisper_cfg: dict):
        self.cfg = dict(whisper_cfg)
        device = self.cfg.get("device", "auto")
        compute = self.cfg.get("compute_type", "auto")
        model = self.cfg.get("model", "base")
        log.info("loading whisper model %s (device=%s)...", model, device)
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute)
            # warm-up: CUDA problems (missing DLLs) only surface at inference time
            list(self.model.transcribe(np.zeros(16000, dtype=np.float32))[0])
        except (RuntimeError, ValueError, OSError) as e:
            if device in ("auto", "cuda"):
                # ponytail: CUDA can fail on missing cuBLAS/cuDNN; CPU fallback keeps the app usable
                log.warning("GPU path failed (%s); falling back to CPU int8", e)
                self.model = WhisperModel(model, device="cpu", compute_type="int8")
            else:
                raise
        log.info("whisper model ready")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < 1600:  # <0.1s of audio, nothing to do
            return ""
        no_speech_max = self.cfg.get("no_speech_threshold", 0.6)
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.get("language") or None,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": self.cfg.get("vad_min_silence_ms", 400),
                "speech_pad_ms": self.cfg.get("vad_speech_pad_ms", 300),
            },
            beam_size=self.cfg.get("beam_size", 5),
            initial_prompt=self.cfg.get("initial_prompt") or None,
            temperature=0.0,  # deterministic decoding; whisper's temperature-fallback ladder is where garbled words/punctuation come from
            compression_ratio_threshold=2.4,  # reject repetitive/garbled segments (whisper's own default, was previously unset)
            log_prob_threshold=-1.0,  # reject low-confidence segments (whisper's own default, was previously unset)
            condition_on_previous_text=True,  # use cross-segment context for coherence on multi-sentence dictation
            hallucination_silence_threshold=self.cfg.get("hallucination_silence_threshold", 2.0),  # skip long silent gaps instead of hallucinating into them (replaces condition_on_previous_text=False as the loop-prevention mechanism)
            no_speech_threshold=no_speech_max,
        )
        parts = []
        for seg in segments:
            text = seg.text.strip()
            if not text or seg.no_speech_prob > no_speech_max:
                continue
            if not any(ch.isalnum() for ch in text):  # e.g. "....." or "???" hallucinated on silence
                continue
            parts.append(text)
        return " ".join(parts).strip()
