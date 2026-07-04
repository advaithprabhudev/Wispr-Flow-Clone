
import logging
import sys
import threading

import pystray
from PIL import Image, ImageDraw

import audio
import cleanup
import config
import hotkey
import inject
import transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATE_COLORS = {"idle": (90, 90, 90), "recording": (220, 60, 60), "transcribing": (60, 120, 220)}


def _icon_image(state: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=STATE_COLORS[state])
    return img


class App:
    def __init__(self):
        self.cfg = config.load()
        self.recorder = audio.Recorder(save_debug_wav=self.cfg["debug"]["save_last_recording"])
        self.transcriber = None  # loaded in run() after startup checks
        self.listener = None
        self.tray = None
        self.target_hwnd = 0
        self._busy = threading.Lock()

    # --- hotkey callbacks (run on keyboard hook thread) ---

    def on_press(self):
        if self._busy.locked():
            return
        self.target_hwnd = inject.snapshot_foreground()
        self.recorder.start()
        self.set_state("recording")

    def on_release(self):
        clip = self.recorder.stop()
        self.set_state("transcribing")
        threading.Thread(target=self._process, args=(clip,), daemon=True).start()

    def _process(self, clip):
        with self._busy:
            try:
                text = self.transcriber.transcribe(clip)
                log.info("transcript: %r", text)
                if text and self.cfg["ollama"]["enabled"]:
                    text = cleanup.clean(text, self.cfg["ollama"])
                    log.info("cleaned:    %r", text)
                if text:
                    inject.inject_text(text, self.target_hwnd)
            except Exception:
                log.exception("dictation pipeline failed")
            finally:
                self.set_state("idle")

    # --- tray ---

    def set_state(self, state: str):
        if self.tray:
            self.tray.icon = _icon_image(state)
            self.tray.title = f"Dictation: {state}"

    def reload_config(self, *_):
        old_whisper = self.cfg["whisper"]
        self.cfg = config.load()
        self.listener.stop()
        self.listener = hotkey.HotkeyListener(self.cfg["hotkey"], self.on_press, self.on_release)
        self.listener.start()
        if self.cfg["whisper"] != old_whisper:
            log.info("whisper settings changed; reloading model")
            self.transcriber = transcribe.Transcriber(self.cfg["whisper"])
        log.info("config reloaded")

    def quit(self, *_):
        self.listener.stop()
        self.tray.stop()

    # --- startup ---

    def run(self):
        errors = []
        if not audio.mic_available():
            errors.append("No microphone found. Plug one in or check Windows privacy settings.")
        if self.cfg["ollama"]["enabled"]:
            ok, msg = cleanup.ollama_ready(self.cfg["ollama"])
            if not ok:
                errors.append(msg + "\n(Or set ollama.enabled: false in config.yaml to paste raw transcripts.)")
        if errors:
            for e in errors:
                log.error(e)
            sys.exit(1)

        self.transcriber = transcribe.Transcriber(self.cfg["whisper"])
        self.listener = hotkey.HotkeyListener(self.cfg["hotkey"], self.on_press, self.on_release)
        self.listener.start()
        log.info("ready — hold %s to dictate", "+".join(self.cfg["hotkey"]))

        self.tray = pystray.Icon(
            "dictation", _icon_image("idle"), "Dictation: idle",
            menu=pystray.Menu(
                pystray.MenuItem("Reload config", self.reload_config),
                pystray.MenuItem("Quit", self.quit),
            ),
        )
        self.tray.run()  # blocks until quit


if __name__ == "__main__":
    App().run()
