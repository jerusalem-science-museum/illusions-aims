"""ROI trigger: fires when the ROI's mean color stays far enough from its baseline for long enough."""
from typing import Optional, Tuple

import numpy as np

import constant as cfg


def roi_rect(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """ROI as (x0, y0, x1, y1) in camera pixels, clamped to the frame."""
    x0 = min(max(0, cfg.ROI_X), frame_w - 1)
    y0 = min(max(0, cfg.ROI_Y), frame_h - 1)
    return x0, y0, min(frame_w, x0 + max(1, cfg.ROI_W)), min(frame_h, y0 + max(1, cfg.ROI_H))


def roi_mean(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0).astype(np.float32)


class ROITriggerDetector:
    """
    - Collect a baseline mean color for BASELINE_SECONDS (and its noise level)
    - Distance = Euclidean distance between the current ROI mean and the baseline
    - Trigger when distance >= threshold for HOLD_SECONDS (short dips are tolerated)
    - Then COOLDOWN_SECONDS, plus any disable window set by disable_for()
    """

    def __init__(self):
        self._baseline_start: Optional[float] = None
        self._baseline_samples = []
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_noise = 0.0   # std-based noise floor (vector norm)
        self._last_ts: Optional[float] = None
        self._below_since: Optional[float] = None
        self._active_since: Optional[float] = None
        self._cooldown_until = 0.0
        self._disabled_until = 0.0

    def _distance(self, cur: np.ndarray) -> float:
        base = self._baseline_mean
        diff = cur - base
        if cfg.BRIGHTNESS_COMP:
            # Remove the common (gray) shift, keep colour change; then take the larger of
            # the colour change and a de-weighted brightness change.
            gray = float(diff.mean())
            chroma = float(np.linalg.norm(diff - gray))
            # Gain-style exposure drift scales all channels proportionally
            gain = float(cur.mean()) / (float(base.mean()) + 1e-6)
            resid = float(np.linalg.norm(cur - base * gain))
            return max(min(resid, float(np.linalg.norm(diff))), chroma)
        return float(np.linalg.norm(diff))

    def disable_for(self, seconds: float, now: float) -> None:
        self._disabled_until = max(self._disabled_until, now + seconds)

    def update(self, mean: np.ndarray, now: float, allow_trigger: bool) -> Tuple[bool, bool, bool]:
        """Returns (baseline_ready, roi_enabled, should_trigger). should_trigger is True once per trigger."""
        # Baseline acquisition
        if self._baseline_mean is None:
            if self._baseline_start is None:
                self._baseline_start = now
            self._baseline_samples.append(mean)
            if now - self._baseline_start >= cfg.BASELINE_SECONDS:
                arr = np.stack(self._baseline_samples)
                self._baseline_mean = arr.mean(axis=0).astype(np.float32)
                self._baseline_noise = float(np.linalg.norm(arr.std(axis=0)))
                self._baseline_samples = []
                self._last_ts = now
            return self._baseline_mean is not None, False, False

        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        dist = self._distance(mean)
        thr = max(cfg.TRIGGER_DIST_THRESHOLD, cfg.NOISE_SIGMA_MULT * self._baseline_noise)

        # Adaptive baseline: follow slow lighting drift, only while clearly quiet
        if dist < thr and self._active_since is None and cfg.BASELINE_ADAPT_SECONDS > 0 and dt > 0:
            a = min(1.0, dt / cfg.BASELINE_ADAPT_SECONDS)
            self._baseline_mean = (self._baseline_mean + a * (mean - self._baseline_mean)).astype(np.float32)

        enabled = allow_trigger and now >= self._cooldown_until and now >= self._disabled_until
        if not enabled:
            self._active_since = None
            self._below_since = None
            return True, False, False

        if dist >= thr:
            self._below_since = None
            if self._active_since is None:
                self._active_since = now
            if now - self._active_since >= cfg.HOLD_SECONDS:
                self._active_since = None
                self._cooldown_until = now + cfg.COOLDOWN_SECONDS
                return True, True, True
        elif self._active_since is not None:
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since > cfg.HOLD_GRACE_SECONDS:
                self._active_since = None
                self._below_since = None

        return True, True, False
