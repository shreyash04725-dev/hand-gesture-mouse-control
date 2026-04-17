"""
virtual_keyboard.py
-------------------
Draws a QWERTY virtual keyboard on the OpenCV frame and detects taps.
"""

import cv2
import numpy as np
import time


# ── Keyboard layout ───────────────────────────────────────────────────────────
ROWS = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M","BACK"],
    ["SPACE","ENTER"],
]

# Special key widths (multiplier of key_w)
WIDE_KEYS = {"SPACE": 4, "ENTER": 2, "BACK": 2}


class Key:
    def __init__(self, label, x, y, w, h):
        self.label = label
        self.x = x      # top-left x
        self.y = y      # top-left y
        self.w = w
        self.h = h
        self.active = False
        self.active_until = 0

    @property
    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def contains(self, px, py):
        return self.x < px < self.x + self.w and self.y < py < self.y + self.h


class VirtualKeyboard:
    def __init__(self, start_x=60, start_y=400, key_w=65, key_h=58, gap=6):
        self.keys: list[Key] = []
        self._build(start_x, start_y, key_w, key_h, gap)
        self._active_key = None
        self._active_until = 0

    def _build(self, sx, sy, kw, kh, gap):
        for row_idx, row in enumerate(ROWS):
            x = sx + row_idx * (kw // 3)   # slight indent per row
            y = sy + row_idx * (kh + gap)
            for label in row:
                w = kw * WIDE_KEYS.get(label, 1) + gap * (WIDE_KEYS.get(label, 1) - 1)
                self.keys.append(Key(label, x, y, w, kh))
                x += w + gap

    def set_active_key(self, label):
        self._active_key  = label
        self._active_until = time.time() + 0.35   # highlight for 350 ms

    def check_tap(self, index_tip, thumb_tip, threshold):
        """
        Returns key label if index fingertip is over a key
        AND thumb-index distance is below threshold (pinch gesture).
        """
        import numpy as np
        dist = np.hypot(index_tip[0] - thumb_tip[0],
                        index_tip[1] - thumb_tip[1])
        if dist >= threshold:
            return None
        for key in self.keys:
            if key.contains(*index_tip):
                return key.label
        return None

    def draw(self, frame):
        now = time.time()
        for key in self.keys:
            x1, y1, x2, y2 = key.rect

            # Decide colours
            if key.label == self._active_key and now < self._active_until:
                bg_color    = (50, 220, 120)   # green flash on press
                text_color  = (0, 0, 0)
                border_color = (0, 180, 80)
            else:
                bg_color    = (30, 30, 30)     # dark glass look
                text_color  = (230, 230, 230)
                border_color = (80, 80, 80)

            # Background (semi-transparent overlay trick)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

            # Label
            label = key.label
            font_scale = 0.55 if len(label) == 1 else 0.38
            thickness  = 2    if len(label) == 1 else 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
            tx = x1 + (key.w - tw) // 2
            ty = y1 + (key.h + th) // 2
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        text_color, thickness, cv2.LINE_AA)
