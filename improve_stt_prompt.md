# Prompt: upgrade voice-to-text quality in wispr_flow_clone

Paste everything below into a Claude Code (Fable 5) session opened in
`C:\Users\prabh\wispr_flow_clone`.

---

My local voice-dictation app (this repo) has poor transcription accuracy and I
want you to fix it. Context you should verify, not assume:

- Pipeline: hold hotkey → `audio.py` records float32 mono 16 kHz via
  sounddevice → transcription via faster-whisper → `cleanup.py` (Ollama) →
  `inject.py` types the text into the focused app.
- Current model is `base` (see `config.py` DEFAULTS → `whisper.model`), which
  is the main accuracy bottleneck.
- Hardware: NVIDIA RTX 5080 (16 GB VRAM), CUDA already works — the app
  transcribes 3 s of audio in ~0.4 s on the current small model, so there is
  plenty of headroom for a much larger model.

Do the following, in order, testing after each step:

1. **Upgrade the model.** Switch the default to `large-v3` with
   `compute_type: float16` on CUDA. If first-token latency after releasing the
   hotkey feels too slow in practice, fall back to `distil-large-v3` (near
   large-v3 accuracy, several times faster) and note the tradeoff. Keep the
   CPU fallback path working — the fallback model should be a config option
   (e.g. `small` with `int8`), not hardcoded.

2. **Tune faster-whisper decoding for dictation.** In the transcribe call, set
   and expose via `config.yaml`:
   - `beam_size=5` (accuracy over greedy decoding)
   - `language: en` pinned in config (auto-detect on short clips is a common
     source of garbage transcripts — keep it configurable)
   - `vad_filter=True` with `min_silence_duration_ms` around 300–500 so
     leading/trailing silence doesn't produce hallucinated text
   - `condition_on_previous_text=False` (short dictation clips; prevents
     hallucination loops)
   - an `initial_prompt` config option so I can bias vocabulary toward my
     domain (software terms, names I use often). Document it in
     config.example.yaml with an example value.

3. **Check audio quality into the model.** Verify the recorded audio isn't the
   problem: add a debug flag that saves the last recording to a WAV file so I
   can listen to it. If the level is very low or clipped, normalize peak
   amplitude before transcription.

4. **Verify end-to-end.** Keep the existing model warm-up at startup. Time a
   realistic 5–10 s dictation and report: model load time, transcription
   latency, and before/after transcripts of the same test audio so I can see
   the accuracy difference. Update `config.example.yaml` and the README to
   document every new option.

Constraints: fully local (no cloud STT), keep the existing module structure,
keep the CUDA→CPU fallback intact, and don't break the running tray app —
tell me the restart command when done.
