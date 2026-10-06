"""Moving QR queue: the newest QR sits in a big slot, older ones move through the archive cells.

All state is guarded by a lock: push() is called from the capture thread,
render() from the main loop. Resized copies are cached per slot size.
Animation is time-based (evaluated per render), no threads or sleeps.
"""
import threading
import time
from typing import Callable, List, Optional

import cv2
import numpy as np

import constant as cfg


class _Item:
    __slots__ = ("src", "sized", "slot", "prev_slot", "t0")

    def __init__(self, src, slot, prev_slot, t0):
        self.src = src              # original QR image
        self.sized = {}             # side length -> resized copy
        self.slot = slot            # logical position (0 = newest)
        self.prev_slot = prev_slot  # position before the last shift (None = new)
        self.t0 = t0


class QRQueue:
    def __init__(self, width: int, height: int, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._items: List[_Item] = []   # index 0 = newest
        self._last_push: Optional[float] = None
        # pixel rects (x0, y0, x1, y1) in logical order (slot 0 = newest)
        self._rects = []
        for key in cfg.QR_SLOT_ORDER:
            fx0, fy0, fx1, fy1 = cfg.QR_SLOT_RECTS_FRAC[key]
            self._rects.append((round(fx0 * width), round(fy0 * height),
                                round(fx1 * width), round(fy1 * height)))
        pad = int(cfg.QR_SLOT_PADDING_PX)
        # each slot has its own QR side length (the big slot is larger than the archive cells)
        self._sizes = [max(8, min(x1 - x0, y1 - y0) - 2 * pad) for x0, y0, x1, y1 in self._rects]
        # Everything the queue can draw on (animations move between slots, so stay inside this)
        self.bbox = (min(r[0] for r in self._rects), min(r[1] for r in self._rects),
                     max(r[2] for r in self._rects), max(r[3] for r in self._rects))

    def _origin(self, slot: int, size: int):
        x0, y0, x1, y1 = self._rects[slot]
        return (x0 + x1 - size) // 2, (y0 + y1 - size) // 2

    @staticmethod
    def _sized(it: _Item, size: int) -> np.ndarray:
        img = it.sized.get(size)
        if img is None:
            img = cv2.resize(it.src, (size, size), interpolation=cv2.INTER_NEAREST)
            if len(it.sized) > 6:  # animation visits many intermediate sizes; keep the cache small
                it.sized.clear()
            it.sized[size] = img
        return img

    def push(self, qr: np.ndarray) -> None:
        """Add a new QR image; shifts the others and drops the oldest."""
        now = self._clock()
        with self._lock:
            for it in self._items:
                it.prev_slot = it.slot
                it.slot += 1
                it.t0 = now
            self._items = [it for it in self._items if it.slot < len(self._rects)]
            self._items.insert(0, _Item(np.ascontiguousarray(qr), 0, None, now))
            self._last_push = now

    def render(self, frame: np.ndarray) -> None:
        """Draw QRs onto frame in place."""
        now = self._clock()
        with self._lock:
            if self._last_push is not None and now - self._last_push >= cfg.QR_QUEUE_RESET_S:
                self._items = []
                self._last_push = None
            # draw oldest first so newest ends on top
            for it in reversed(self._items):
                size = self._sizes[it.slot]
                x, y = self._origin(it.slot, size)
                if it.prev_slot is not None and cfg.QR_ANIM_S > 0:
                    t = (now - it.t0) / cfg.QR_ANIM_S
                    if t < 1.0:
                        psize = self._sizes[it.prev_slot]
                        px, py = self._origin(it.prev_slot, psize)
                        k = t * t * (3 - 2 * t)  # smoothstep
                        size = int(round(psize + (size - psize) * k))
                        x = int(round(px + (x - px) * k))
                        y = int(round(py + (y - py) * k))
                img = self._sized(it, size)
                frame[y:y + size, x:x + size] = img
