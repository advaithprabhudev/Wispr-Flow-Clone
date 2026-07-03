"""Load config.yaml with defaults."""
import copy
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

DEFAULTS = {
    "hotkey": ["ctrl", "alt", "space"],
    "whisper": {
        "model": "large-v3",
        "device": "auto",  # cpu | cuda | auto
        "compute_type": "auto",
        "language": "en",  # None = auto-detect (unreliable on short clips), or e.g. "en"
        "beam_size": 5,
        "initial_prompt": None,  # e.g. domain vocab/names to bias decoding
        "vad_min_silence_ms": 400,  # trailing/leading silence trimmed before it hallucinates text
        "vad_speech_pad_ms": 300,  # padding kept around detected speech so word edges aren't clipped
        "no_speech_threshold": 0.45,  # segments above this silence-confidence are dropped (kills "....." / "???"); 0.6 is whisper's stock default and didn't filter anything extra
        "hallucination_silence_threshold": 2.0,  # skip >2s silent gaps instead of hallucinating text into them
    },
    "debug": {
        "save_last_recording": False,  # writes last_recording.wav next to the app for listening back
    },
    "ollama": {
        "enabled": True,
        "url": "http://localhost:11434",
        "model": "qwen2.5:7b-instruct",
    },
}


def load() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        user = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg
