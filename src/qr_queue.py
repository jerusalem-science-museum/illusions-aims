"""Moving QR queue: the newest QR sits in a big slot, older ones move through the archive cells.

All state is guarded by a lock: push() is called from the capture_flow worker
thread, render() from the main loop. Resized copies are cached per slot size.
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
    __slots__ = ("src", "sized", "slot", "prev_slot", "t0")

    def __init__(self, src, slot, prev_slot, t0):
        self.src = src              # original QR (RGB uint8)
        self.sized = {}             # side length -> resized copy
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
        pad = int(QR_SLOT_PADDING_PX)
        # each slot has its own QR side length (the big slot is larger than the archive cells)
        self._sizes = [max(8, min(x1 - x0, y1 - y0) - 2 * pad) for x0, y0, x1, y1 in self._rects]

    def _origin(self, slot: int, size: int):
        x0, y0, x1, y1 = self._rects[slot]
        return (x0 + x1 - size) // 2, (y0 + y1 - size) // 2

    @staticmethod
    def _sized(it: "_Item", size: int) -> np.ndarray:
        img = it.sized.get(size)
        if img is None:
            img = cv2.resize(it.src, (size, size), interpolation=cv2.INTER_NEAREST)
            if len(it.sized) > 6:  # animation visits many intermediate sizes; keep the cache small
                it.sized.clear()
            it.sized[size] = img
        return img

    def push(self, qr_rgb: np.ndarray) -> None:
        """Add a new QR (RGB uint8 array); shifts others, drops the oldest."""
        src = np.ascontiguousarray(qr_rgb)
        now = self._clock()
        with self._lock:
            for it in self._items:
                it.prev_slot = it.slot
                it.slot += 1
                it.t0 = now
            self._items = [it for it in self._items if it.slot < len(self._rects)]
            self._items.insert(0, _Item(src, 0, None, now))
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
                size = self._sizes[it.slot]
                x, y = self._origin(it.slot, size)
                if it.prev_slot is not None and self._anim_s > 0:
                    t = (now - it.t0) / self._anim_s
                    if t < 1.0:
                        psize = self._sizes[it.prev_slot]
                        px, py = self._origin(it.prev_slot, psize)
                        k = t * t * (3 - 2 * t)  # smoothstep
                        size = int(round(psize + (size - psize) * k))
                        x = int(round(px + (x - px) * k))
                        y = int(round(py + (y - py) * k))
                self._blit(frame_rgb, self._sized(it, size), x, y)
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
