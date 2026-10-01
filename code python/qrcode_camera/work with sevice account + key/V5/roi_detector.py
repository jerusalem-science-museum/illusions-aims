"""
ROI detection, placement, and trigger logic.

This module is intentionally UI-agnostic: it does not depend on Tkinter.
It is used by graphics.py to decide where the ROI should be placed and how it behaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from constant import (
    ROI_X, ROI_Y, ROI_W, ROI_H,
    BASELINE_SECONDS, TRIGGER_DIST_THRESHOLD, HOLD_SECONDS, COOLDOWN_SECONDS,
)


@dataclass(frozen=True)
class ROI:
    """ROI rectangle (x, y, w, h) + optional shape."""
    x: int
    y: int
    w: int
    h: int
    shape: str = "rect"  # 'rect' or 'circle' (circle is drawn inside this bounding box)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), int(hi)))


def decide_roi_cam(frame_w: int, frame_h: int) -> ROI:
    """
    Decide the ROI position in CAMERA coordinates.

    By default, uses ROI_X/ROI_Y/ROI_W/ROI_H from constant.py, clamped to the frame size.
    If you add ROI_SHAPE in constant.py ('rect' or 'circle'), it will be used automatically.
    """
    frame_w = max(1, int(frame_w))
    frame_h = max(1, int(frame_h))

    rx = _clamp_int(ROI_X, 0, frame_w - 1)
    ry = _clamp_int(ROI_Y, 0, frame_h - 1)
    rw = _clamp_int(ROI_W, 1, frame_w - rx)
    rh = _clamp_int(ROI_H, 1, frame_h - ry)

    # Optional: allow ROI_SHAPE without requiring it to exist
    try:
        from constant import ROI_SHAPE  # type: ignore
        shape = str(ROI_SHAPE).strip().lower() or "rect"
    except Exception:
        shape = "rect"

    if shape not in ("rect", "circle"):
        shape = "rect"

    return ROI(rx, ry, rw, rh, shape=shape)


def map_roi_cam_to_preview(roi_cam: ROI, frame_w: int, frame_h: int, preview_w: int, preview_h: int) -> ROI:
    """
    Map an ROI expressed in CAMERA coordinates to the current preview (processing) size.
    """
    frame_w = max(1, int(frame_w))
    frame_h = max(1, int(frame_h))
    preview_w = max(1, int(preview_w))
    preview_h = max(1, int(preview_h))

    sx = preview_w / frame_w
    sy = preview_h / frame_h

    x = int(roi_cam.x * sx)
    y = int(roi_cam.y * sy)
    w = max(1, int(roi_cam.w * sx))
    h = max(1, int(roi_cam.h * sy))
    return ROI(x, y, w, h, shape=roi_cam.shape)


def roi_mean_rgb(preview_rgb: np.ndarray, roi: ROI) -> np.ndarray:
    """
    Compute the mean RGB vector of the ROI region in a preview frame.
    Returns float32 array of shape (3,).
    """
    H, W = preview_rgb.shape[:2]
    x1 = _clamp_int(roi.x, 0, W - 1)
    y1 = _clamp_int(roi.y, 0, H - 1)
    x2 = _clamp_int(roi.x + roi.w, 0, W)
    y2 = _clamp_int(roi.y + roi.h, 0, H)

    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    patch = preview_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


class ROITriggerDetector:
    """
    Maintain a baseline and decide when the ROI has changed enough to trigger a capture.

    Trigger rule:
      - Collect baseline for BASELINE_SECONDS
      - Compute Euclidean distance between current ROI mean (RGB) and baseline mean
      - Require distance >= TRIGGER_DIST_THRESHOLD for HOLD_SECONDS
      - Apply COOLDOWN_SECONDS after a trigger
    """

    def __init__(
        self,
        baseline_seconds: float = float(BASELINE_SECONDS),
        trigger_dist_threshold: float = float(TRIGGER_DIST_THRESHOLD),
        hold_seconds: float = float(HOLD_SECONDS),
        cooldown_seconds: float = float(COOLDOWN_SECONDS),
    ):
        self.baseline_seconds = float(baseline_seconds)
        self.trigger_dist_threshold = float(trigger_dist_threshold)
        self.hold_seconds = float(hold_seconds)
        self.cooldown_seconds = float(cooldown_seconds)

        self._baseline_start = None  # type: Optional[float]
        self._baseline_samples = []  # list[np.ndarray]
        self._baseline_mean = None   # type: Optional[np.ndarray]

        self._active_since = None    # type: Optional[float]
        self._cooldown_until = 0.0
        self._disabled_until = 0.0

    @property
    def baseline_ready(self) -> bool:
        return self._baseline_mean is not None

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        return self._baseline_mean

    def reset_baseline(self, now: float) -> None:
        self._baseline_start = float(now)
        self._baseline_samples.clear()
        self._baseline_mean = None
        self._active_since = None
        self._cooldown_until = 0.0

    def disable_until(self, until_ts: float) -> None:
        self._disabled_until = max(self._disabled_until, float(until_ts))

    def disable_for(self, seconds: float, now: float) -> None:
        self.disable_until(float(now) + float(seconds))

    def update(self, roi_mean: np.ndarray, now: float, allow_trigger: bool) -> Tuple[bool, bool, bool, float]:
        """
        Update detector with the latest ROI mean.

        Returns:
          (baseline_ready, roi_enabled, should_trigger, distance)

        - baseline_ready: baseline is computed and stable
        - roi_enabled: ROI is allowed to trigger (not in cooldown, not disabled, allow_trigger=True)
        - should_trigger: True for exactly one update call when trigger conditions are met
        - distance: current distance from baseline (0.0 if baseline not ready)
        """
        now = float(now)

        if self._baseline_start is None:
            self.reset_baseline(now)

        # Baseline acquisition
        if self._baseline_mean is None:
            self._baseline_samples.append(np.asarray(roi_mean, dtype=np.float32))
            if (now - float(self._baseline_start)) >= self.baseline_seconds and len(self._baseline_samples) > 0:
                arr = np.stack(self._baseline_samples, axis=0)
                self._baseline_mean = arr.mean(axis=0).astype(np.float32)
            return (self._baseline_mean is not None, False, False, 0.0)

        # Gating
        roi_enabled = allow_trigger and (now >= self._cooldown_until) and (now >= self._disabled_until)
        if not roi_enabled:
            self._active_since = None
            dist = float(np.linalg.norm(np.asarray(roi_mean, dtype=np.float32) - self._baseline_mean))
            return (True, False, False, dist)

        # Trigger logic
        dist = float(np.linalg.norm(np.asarray(roi_mean, dtype=np.float32) - self._baseline_mean))
        if dist >= self.trigger_dist_threshold:
            if self._active_since is None:
                self._active_since = now
            held = now - float(self._active_since)
            if held >= self.hold_seconds:
                # Fire once, start cooldown
                self._active_since = None
                self._cooldown_until = now + self.cooldown_seconds
                return (True, True, True, dist)
        else:
            self._active_since = None

        return (True, True, False, dist)
