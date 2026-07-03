"""Global push-to-talk hotkey: fires on_press when the full combo goes down,
on_release when any combo key comes back up."""
import logging

import keyboard

log = logging.getLogger(__name__)

# keyboard lib reports left/right variants; fold them into the generic name
_ALIASES = {
    "left ctrl": "ctrl", "right ctrl": "ctrl",
    "left alt": "alt", "right alt": "alt", "alt gr": "alt",
    "left shift": "shift", "right shift": "shift",
    "left windows": "windows", "right windows": "windows",
}


def _normalize(name: str) -> str:
    return _ALIASES.get(name, name)


class HotkeyListener:
    def __init__(self, combo: list[str], on_press, on_release):
        self.combo = {_normalize(k.lower()) for k in combo}
        self.on_press = on_press
        self.on_release = on_release
        self._pressed: set[str] = set()
        self._active = False
        self._hook = None

    def start(self):
        self._hook = keyboard.hook(self._handle)

    def stop(self):
        if self._hook:
            keyboard.unhook(self._hook)
            self._hook = None
        self._pressed.clear()
        self._active = False

    def _handle(self, event):
        name = _normalize((event.name or "").lower())
        if name not in self.combo:
            return
        if event.event_type == "down":
            self._pressed.add(name)
            if not self._active and self._pressed == self.combo:
                self._active = True
                self._safe(self.on_press)
        else:  # up
            self._pressed.discard(name)
            if self._active:
                self._active = False
                self._safe(self.on_release)

    @staticmethod
    def _safe(fn):
        try:
            fn()
        except Exception:
            log.exception("hotkey callback failed")
