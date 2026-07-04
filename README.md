# Wispr Flow Clone

Local, private push-to-talk dictation for Windows. Hold a hotkey, speak, release — your words are transcribed and typed into whatever app has focus. Still working om it, works well but repeats the same thing 2 times.


Hotkey → record → [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription → optional [Ollama](https://ollama.com) cleanup → text injected via Windows API.

Runs entirely on your machine. No audio or text leaves your PC.

## Requirements

- Windows
- Python 3.10+
- [Ollama](https://ollama.com) running locally (optional, for transcript cleanup)
- NVIDIA GPU recommended for faster transcription (CPU works too)

## Setup

```bash
pip install -r requirements.txt
copy config.example.yaml config.yaml
python app.py
```

Edit `config.yaml` to change the hotkey, Whisper model, or Ollama settings.

## Usage

Hold the configured hotkey (default `ctrl+alt+space`), speak, release. The cleaned-up transcript is typed into the currently focused window. A tray icon shows status (idle / recording / transcribing) and lets you reload config or quit.

## Configuration

See `config.example.yaml` for all options, including:

- **Whisper**: model size, device, language, VAD tuning, hallucination filtering
- **Ollama**: enable/disable cleanup, model choice
- **Debug**: save last recording to `last_recording.wav` for troubleshooting

## Tests

```bash
python -m pytest test_smoke.py
```
