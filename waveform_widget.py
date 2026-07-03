"""Wispr Flow-style desktop waveform widget.

Frameless always-on-top beige pill; smooth black sinusoidal waves whose
amplitude follows live mic level. Frames are drawn with Pillow at 2x
supersampling then downscaled, because tkinter's canvas cannot antialias.
Drag to move, Esc or right-click to close. Run: python waveform_widget.py
"""
import math
import tkinter as tk

import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFilter, ImageTk

SAMPLE_RATE = 16000  # match audio.py
W, H = 520, 68
SS = 2               # supersampling factor (the whole antialiasing trick)
RADIUS = 22
BG = "#0B0D12"       # transparent key color only, never visible
PILL = "#EDE4D3"     # beige
EDGE = "#D8CCB6"
# black waves; lighter warm grays stand in for lower opacity
LAYERS = [  # (spatial freq, speed, amplitude share, color, stroke width @1x)
    (1.3, 1.00, 1.00, "#111111", 2.6),
    (2.1, -0.62, 0.62, "#55503F", 1.8),
    (2.9, 0.83, 0.38, "#9A907A", 1.4),
]
# spectral4()-derived hues from the Siri wave shader: magenta/yellow/green/cyan,
# each phase-shifted (radians) to fan out into a chromatic glow band.
ABERRATION_BANDS = [
    ((255, 0, 170), -0.30),
    ((255, 220, 0), -0.10),
    ((0, 220, 120), 0.10),
    ((0, 180, 255), 0.30),
]


class WaveWidget:
    def __init__(self):
        self.level = 0.0    # smoothed 0..1
        self._target = 0.0  # written from audio thread
        self.t = 0.0
        self.mode = "wave"  # "wave" | "dots" -- toggle with 'd'

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", BG)
        x = (self.root.winfo_screenwidth() - W) // 2
        y = self.root.winfo_screenheight() - H - 80
        self.root.geometry(f"{W}x{H}+{x}+{y}")

        self.label = tk.Label(self.root, bg=BG, bd=0)
        self.label.pack()
        self._photo = None  # keep a ref so tk doesn't GC the frame

        self.label.bind("<Button-1>", self._drag_start)
        self.label.bind("<B1-Motion>", self._drag_move)
        self.label.bind("<Button-3>", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<KeyPress-d>", self._toggle_mode)
        self.root.focus_force()

        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                     dtype="float32", callback=self._audio_cb)
        self.stream.start()

    def _audio_cb(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata**2)))
        self._target = min(1.0, rms * 9.0)

    def _drag_start(self, e):
        self._dx, self._dy = e.x, e.y

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _toggle_mode(self, e):
        self.mode = "dots" if self.mode == "wave" else "wave"

    def _render(self) -> Image.Image:
        w, h = W * SS, H * SS
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([SS, SS, w - SS, h - SS], RADIUS * SS,
                            fill=PILL, outline=EDGE, width=SS)

        mid = h / 2

        if self.mode == "dots":
            return self._render_dots(img, w, h, mid).resize((W, H), Image.LANCZOS)

        max_amp = h * 0.36

        # Voice drives both amplitude and drift speed, so waves visibly pick
        # up motion as you speak instead of just growing taller.
        drift = self.t * (0.5 + 1.9 * self.level)
        # low/mid breathing terms (ported from the Siri wave shader) add
        # organic variation to aberration spread even at constant level.
        low = 0.45 + 0.45 * math.sin(self.t * 0.8) * math.sin(self.t * 0.37 + 1.0)
        mid_b = 0.40 + 0.40 * math.sin(self.t * 1.7 + 2.0) * math.sin(self.t * 0.53)

        # Siri-wave style chromatic aberration: four spectral bands sharing
        # the main wave's shape but phase-shifted, blurred into a soft glow.
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        aber_spread = 1.0 + 0.3 * mid_b
        for color, ab in ABERRATION_BANDS:
            pts = []
            for x in range(0, w + 1, 2 * SS):
                p = x / w
                env = math.sin(p * math.pi) ** 1.3
                y = mid + math.sin(p * math.pi * 2 * 1.3 + drift * 2.4 + ab * aber_spread) \
                    * max_amp * self.level * env
                pts.append((x, y))
            alpha = round(70 * self.level + 20)
            gd.line(pts, fill=color + (alpha,), width=round(3.2 * SS), joint="curve")
        glow = glow.filter(ImageFilter.GaussianBlur(SS * 1.6))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
        d = ImageDraw.Draw(img)

        for freq, speed, share, color, lw in LAYERS:
            pts = []
            for x in range(0, w + 1, 2 * SS):
                p = x / w
                env = math.sin(p * math.pi) ** 1.3  # taper at pill edges
                amp = max_amp * share * (1.0 + 0.08 * low)
                y = mid + math.sin(p * math.pi * 2 * freq + drift * speed * 2.4) \
                    * amp * self.level * env
                pts.append((x, y))
            d.line(pts, fill=color, width=round(lw * SS), joint="curve")

        return img.resize((W, H), Image.LANCZOS)

    def _render_dots(self, img: Image.Image, w: int, h: int, mid: float) -> Image.Image:
        # fluid-dots shader, ported as blurred merging circles rather than a
        # true per-pixel smin metaball field (too slow in pure Python).
        # Confined to a short rounded rectangle (not the full pill width) so
        # dots stay compact instead of spreading edge to edge.
        box_w = min(w, h * 1.15)
        box_h = h * 0.62
        box_y0, box_y1 = mid - box_h / 2, mid + box_h / 2
        box_x0, box_x1 = (w - box_w) / 2, (w + box_w) / 2
        d0 = ImageDraw.Draw(img)
        d0.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], RADIUS * SS * 0.35,
                             outline=EDGE, width=SS)

        n = 4
        colors = [c for c, _ in ABERRATION_BANDS]
        orbit_r = box_w * 0.16 * (1.0 - 0.55 * self.level)  # dots pull together as you speak
        base_r = h * 0.07 + h * 0.06 * self.level
        spin = self.t * (0.6 + 0.8 * self.level)

        def positions():
            for i in range(n):
                ang = i / n * 2 * math.pi + spin
                cx = w / 2 + math.cos(ang) * orbit_r
                cy = mid + math.sin(ang * 1.3 + i) * box_h * 0.22 * (1.0 - 0.5 * self.level)
                r = base_r * (1.0 + 0.15 * math.sin(self.t * 1.7 + i * 1.9))
                yield colors[i], cx, cy, r

        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        glow_alpha = round(120 * max(self.level, 0.15) + 40)
        for color, cx, cy, r in positions():
            gd.ellipse([cx - r * 1.6, cy - r * 1.6, cx + r * 1.6, cy + r * 1.6],
                       fill=color + (glow_alpha,))
        glow = glow.filter(ImageFilter.GaussianBlur(SS * 2.2))
        img = Image.alpha_composite(img.convert("RGBA"), glow)

        d = ImageDraw.Draw(img)
        for color, cx, cy, r in positions():
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (230,))

        return img.convert("RGB")

    def _tick(self):
        self.t += 0.016
        idle = 0.14 + 0.06 * math.sin(self.t * 0.9)
        target = max(idle * 0.6, self._target)
        gain = 0.3 if target > self.level else 0.06  # fast attack, slow release
        self.level += (target - self.level) * gain

        self._photo = ImageTk.PhotoImage(self._render())
        self.label.configure(image=self._photo)
        self.root.after(16, self._tick)

    def run(self):
        self._tick()
        try:
            self.root.mainloop()
        finally:
            self.stream.stop()
            self.stream.close()


if __name__ == "__main__":
    WaveWidget().run()
