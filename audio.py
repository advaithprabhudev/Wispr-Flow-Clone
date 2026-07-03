"""Microphone capture: start() while hotkey held, stop() returns float32 mono @16kHz."""
import logging
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # what Whisper expects
DEBUG_WAV_PATH = Path(__file__).parent / "last_recording.wav"


def _high_pass(samples: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
    """One-pole high-pass (no scipy dep) to cut mic rumble/hum before whisper sees it."""
    if samples.size < 2:
        return samples
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / SAMPLE_RATE
    alpha = rc / (rc + dt)
    diff = np.empty_like(samples)
    diff[0] = 0.0
    diff[1:] = alpha * (samples[1:] - samples[:-1])
    y = np.frompyfunc(lambda prev, d: alpha * prev + d, 2, 1).accumulate(diff, dtype=object)
    return y.astype(np.float32)


class Recorder:
    def __init__(self, save_debug_wav: bool = False):
        self._stream = None
        self._chunks = []
        self.save_debug_wav = save_debug_wav

    def start(self):
        if self._stream is not None:
            return
        self._chunks = []

        def callback(indata, frames, time_info, status):
            if status:
                log.warning("audio status: %s", status)
            self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        samples = np.concatenate(self._chunks).flatten()
        samples = samples - samples.mean()  # remove DC offset before filtering/gain
        samples = _high_pass(samples)  # cut <80Hz rumble/hum that whisper mishears as speech content
        peak = np.abs(samples).max()
        if 0 < peak < 0.5:  # ponytail: quiet mics hurt whisper accuracy; cap gain so noise floor isn't blown up
            samples = samples * min(0.9 / peak, 10.0)
        if self.save_debug_wav:
            self._write_wav(samples)
        return samples

    @staticmethod
    def _write_wav(samples: np.ndarray):
        pcm16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(DEBUG_WAV_PATH), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm16.tobytes())
        log.info("saved debug recording to %s", DEBUG_WAV_PATH)


def mic_available() -> bool:
    try:
        return any(d["max_input_channels"] > 0 for d in sd.query_devices())
    except Exception as e:
        log.error("audio device query failed: %s", e)
        return False
