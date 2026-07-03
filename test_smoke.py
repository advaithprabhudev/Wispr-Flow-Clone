"""Runnable check for the logic that can break silently: config merge and
SendInput unicode event building. Runs without mic, GPU, or Ollama.
Usage: python test_smoke.py"""
import wave

import numpy as np

import audio
import cleanup
import config
from inject import KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, build_unicode_events


def test_config_defaults():
    cfg = config.load()
    assert cfg["whisper"]["model"]
    assert cfg["whisper"]["vad_min_silence_ms"]
    assert cfg["whisper"]["hallucination_silence_threshold"]
    assert cfg["ollama"]["url"].startswith("http")
    assert isinstance(cfg["hotkey"], list) and cfg["hotkey"]


def test_cleanup_rejects_drifted_output():
    assert cleanup._is_reasonable("hello world", "hello world")
    assert not cleanup._is_reasonable("", "hello world")
    assert not cleanup._is_reasonable("hi", "this is a much longer sentence than that")
    assert not cleanup._is_reasonable("a very long rewritten answer instead of a cleanup", "hi")


def test_high_pass_removes_dc_offset_and_preserves_shape():
    samples = (np.ones(1600, dtype=np.float32) * 0.3
               + np.sin(np.linspace(0, 40 * np.pi, 1600)).astype(np.float32) * 0.2)
    filtered = audio._high_pass(samples)
    assert filtered.shape == samples.shape
    assert filtered.dtype == np.float32
    assert abs(filtered[-200:].mean()) < abs(samples.mean())  # DC component attenuated


def test_debug_wav_roundtrip():
    samples = (np.sin(np.linspace(0, 6.28, 1600)) * 0.5).astype(np.float32)
    audio.Recorder._write_wav(samples)
    with wave.open(str(audio.DEBUG_WAV_PATH), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 16000
        assert f.getnframes() == len(samples)
    audio.DEBUG_WAV_PATH.unlink()


def test_unicode_events_bmp():
    events = build_unicode_events("hi")
    assert events == [
        (ord("h"), KEYEVENTF_UNICODE),
        (ord("h"), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
        (ord("i"), KEYEVENTF_UNICODE),
        (ord("i"), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
    ]


def test_unicode_events_surrogate_pair():
    events = build_unicode_events("\U0001F600")  # emoji -> 2 UTF-16 units -> 4 events
    assert len(events) == 4
    assert events[0][0] == 0xD83D and events[2][0] == 0xDE00


def test_newline_becomes_carriage_return():
    events = build_unicode_events("a\nb")
    assert events[2][0] == 0x0D  # \r


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("all smoke tests passed")
