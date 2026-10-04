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
import constant as _c

# Optional tunables (fall back to defaults if missing from constant.py)
NOISE_SIGMA_MULT = float(getattr(_c, "NOISE_SIGMA_MULT", 6.0))
BASELINE_ADAPT_SECONDS = float(getattr(_c, "BASELINE_ADAPT_SECONDS", 5.0))
HOLD_GRACE_SECONDS = float(getattr(_c, "HOLD_GRACE_SECONDS", 0.2))
BRIGHTNESS_COMP = bool(getattr(_c, "BRIGHTNESS_COMP", False))


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

        self._baseline_noise = 0.0   # std-based noise floor (vector norm)
        self._last_ts = None         # type: Optional[float]
        self._below_since = None     # type: Optional[float]

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
        self._baseline_noise = 0.0
        self._last_ts = None
        self._below_since = None
        self._active_since = None
        self._cooldown_until = 0.0

    def _distance(self, roi_mean: np.ndarray) -> float:
        """Distance from baseline; ignores pure brightness (auto-exposure) shifts partly."""
        cur = np.asarray(roi_mean, dtype=np.float32)
        base = self._baseline_mean
        diff = cur - base
        if BRIGHTNESS_COMP:
            # Remove the common (gray) shift, keep colour change; then take the larger of
            # the colour change and a de-weighted brightness change.
            gray = float(diff.mean())
            chroma = float(np.linalg.norm(diff - gray))
            # Gain-style exposure drift scales all channels proportionally
            bm = float(base.mean()) + 1e-6
            gain = float(cur.mean()) / bm
            resid = float(np.linalg.norm(cur - base * gain))
            return max(min(resid, float(np.linalg.norm(diff))), chroma)
        return float(np.linalg.norm(diff))

    def _effective_threshold(self) -> float:
        return max(self.trigger_dist_threshold, NOISE_SIGMA_MULT * self._baseline_noise)

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
                self._baseline_noise = float(np.linalg.norm(arr.std(axis=0)))
                self._last_ts = now
            return (self._baseline_mean is not None, False, False, 0.0)

        dt = 0.0 if self._last_ts is None else max(0.0, now - self._last_ts)
        self._last_ts = now
        dist = self._distance(roi_mean)
        thr = self._effective_threshold()

        # Adaptive baseline: follow slow lighting drift, only while clearly quiet
        if dist < thr and self._active_since is None and BASELINE_ADAPT_SECONDS > 0 and dt > 0:
            a = min(1.0, dt / BASELINE_ADAPT_SECONDS)
            self._baseline_mean = (self._baseline_mean + a * (np.asarray(roi_mean, dtype=np.float32) - self._baseline_mean)).astype(np.float32)

        # Gating
        roi_enabled = allow_trigger and (now >= self._cooldown_until) and (now >= self._disabled_until)
        if not roi_enabled:
            self._active_since = None
            self._below_since = None
            return (True, False, False, dist)

        # Trigger logic (short dropouts below threshold don't reset the hold timer)
        if dist >= thr:
            self._below_since = None
            if self._active_since is None:
                self._active_since = now
            if now - float(self._active_since) >= self.hold_seconds:
                self._active_since = None
                self._cooldown_until = now + self.cooldown_seconds
                return (True, True, True, dist)
        elif self._active_since is not None:
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since > HOLD_GRACE_SECONDS:
                self._active_since = None
                self._below_since = None

        return (True, True, False, dist)
