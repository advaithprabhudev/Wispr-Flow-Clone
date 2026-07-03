"""Text injection: clipboard copy (manual-paste fallback) + SendInput KEYEVENTF_UNICODE
typing into the window that had focus when the hotkey was pressed. ctypes only, no pywin32."""
import ctypes
import ctypes.wintypes as wt
import logging
import time

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
        ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", _INPUTUNION)]


def snapshot_foreground() -> int:
    return user32.GetForegroundWindow()


def build_unicode_events(text: str) -> list[tuple[int, int]]:
    """(wScan, flags) pairs. UTF-16 code units handle surrogate pairs (emoji etc.);
    \n becomes \r so it registers as Enter in most apps."""
    events = []
    units = text.replace("\r\n", "\n").replace("\n", "\r").encode("utf-16-le")
    for i in range(0, len(units), 2):
        scan = int.from_bytes(units[i:i + 2], "little")
        events.append((scan, KEYEVENTF_UNICODE))
        events.append((scan, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    return events


def _send_unicode(text: str):
    events = build_unicode_events(text)
    arr = (INPUT * len(events))()
    for inp, (scan, flags) in zip(arr, events):
        inp.type = INPUT_KEYBOARD
        inp.union.ki = KEYBDINPUT(0, scan, flags, 0, None)
    sent = user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))
    if sent != len(arr):
        raise OSError(f"SendInput sent {sent}/{len(arr)} events")


def copy_to_clipboard(text: str):
    data = text.encode("utf-16-le") + b"\x00\x00"
    for _ in range(5):  # clipboard can be briefly locked by another app
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        raise OSError("could not open clipboard")
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = kernel32.GlobalLock(handle)
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        user32.SetClipboardData(CF_UNICODETEXT, handle)
    finally:
        user32.CloseClipboard()


def inject_text(text: str, target_hwnd: int):
    if not text:
        return
    try:
        copy_to_clipboard(text)  # fallback: user can Ctrl+V manually if injection misses
    except OSError as e:
        log.warning("clipboard copy failed: %s", e)
    if target_hwnd and user32.GetForegroundWindow() != target_hwnd:
        user32.SetForegroundWindow(target_hwnd)
        time.sleep(0.05)
    _send_unicode(text)
