"""Moving QR queue drawn into the 2x2 crosshair "archive" slots of the template.

All state is guarded by a lock: push() is called from the capture_flow worker
thread, render() from the main loop. Images are pre-resized in push().
Animation is time-based (evaluated per render), no threads or sleeps.
"""
import threading
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

from constant import (
    QR_SLOT_RECTS_FRAC,
    QR_SLOT_ORDER,
    QR_SLOT_PADDING_PX,
    QR_QUEUE_RESET_S,
    QR_ANIM_STEPS,
    QR_ANIM_DELAY_MS,
)


class _Item:
    __slots__ = ("img", "slot", "prev_slot", "t0")

    def __init__(self, img, slot, prev_slot, t0):
        self.img = img
        self.slot = slot            # logical position (0 = newest)
        self.prev_slot = prev_slot  # position before the last shift (None = new)
        self.t0 = t0


class QRQueue:
    def __init__(self, width: int, height: int, clock: Callable[[], float] = time.monotonic):
        self.w = int(width)
        self.h = int(height)
        self._clock = clock
        self._lock = threading.Lock()
        self._items: List[_Item] = []   # index 0 = newest
        self._last_push: Optional[float] = None
        self._anim_s = max(0.0, QR_ANIM_STEPS * QR_ANIM_DELAY_MS / 1000.0)
        # pixel rects (x0, y0, x1, y1) in logical order (slot 0 = newest)
        self._rects = []
        for key in QR_SLOT_ORDER:
            fx0, fy0, fx1, fy1 = QR_SLOT_RECTS_FRAC[key]
            self._rects.append((round(fx0 * self.w), round(fy0 * self.h),
                                round(fx1 * self.w), round(fy1 * self.h)))
        x0, y0, x1, y1 = self._rects[0]
        pad = int(QR_SLOT_PADDING_PX)
        self._size = max(8, min(x1 - x0, y1 - y0) - 2 * pad)

    def _origin(self, slot: int):
        x0, y0, x1, y1 = self._rects[slot]
        return (x0 + x1 - self._size) // 2, (y0 + y1 - self._size) // 2

    def push(self, qr_rgb: np.ndarray) -> None:
        """Add a new QR (RGB uint8 array); shifts others, drops the oldest."""
        small = cv2.resize(qr_rgb, (self._size, self._size), interpolation=cv2.INTER_NEAREST)
        now = self._clock()
        with self._lock:
            for it in self._items:
                it.prev_slot = it.slot
                it.slot += 1
                it.t0 = now
            self._items = [it for it in self._items if it.slot < len(self._rects)]
            self._items.insert(0, _Item(small, 0, None, now))
            self._last_push = now

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._last_push = None

    def __len__(self):
        with self._lock:
            return len(self._items)

    def render(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Draw QRs onto frame_rgb in place (and return it)."""
        now = self._clock()
        with self._lock:
            if self._last_push is not None and now - self._last_push >= QR_QUEUE_RESET_S:
                self._items = []
                self._last_push = None
            # draw oldest first so newest ends on top
            for it in reversed(self._items):
                x, y = self._origin(it.slot)
                if it.prev_slot is not None and self._anim_s > 0:
                    t = (now - it.t0) / self._anim_s
                    if t < 1.0:
                        px, py = self._origin(it.prev_slot)
                        k = t * t * (3 - 2 * t)  # smoothstep
                        x = int(round(px + (x - px) * k))
                        y = int(round(py + (y - py) * k))
                self._blit(frame_rgb, it.img, x, y)
        return frame_rgb

    @staticmethod
    def _blit(frame, img, x, y):
        fh, fw = frame.shape[:2]
        h, w = img.shape[:2]
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(fw, x + w), min(fh, y + h)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        frame[dy0:dy1, dx0:dx1] = img[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
