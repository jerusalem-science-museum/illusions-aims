"""ROI trigger: fires when the ROI's mean color stays far enough from its baseline for long enough."""
from typing import Optional, Tuple

import numpy as np

import constant as cfg


def roi_rect(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """ROI as (x0, y0, x1, y1) in camera pixels, clamped to the frame."""
    x0 = min(max(0, cfg.ROI_X), frame_w - 1)
    y0 = min(max(0, cfg.ROI_Y), frame_h - 1)
    return x0, y0, min(frame_w, x0 + max(1, cfg.ROI_W)), min(frame_h, y0 + max(1, cfg.ROI_H))


def roi_patch(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return frame[y0:y1, x0:x1]


def roi_mean(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    return roi_patch(frame, rect).reshape(-1, 3).mean(axis=0).astype(np.float32)


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
        # Reference ROI pixels, kept in step with the baseline mean (only when update() gets a patch).
        # Used by the debug view; the trigger itself only looks at the mean.
        self.reference: Optional[np.ndarray] = None
        self._ref_sum: Optional[np.ndarray] = None
        self._ref_n = 0
        self.last_dist = 0.0
        self.last_thr = 0.0
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

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        return self._baseline_mean

    @property
    def baseline_noise(self) -> float:
        return self._baseline_noise

    def update(self, mean: np.ndarray, now: float, allow_trigger: bool,
               patch: Optional[np.ndarray] = None) -> Tuple[bool, bool, bool]:
        """Returns (baseline_ready, roi_enabled, should_trigger). should_trigger is True once per trigger.

        patch (the ROI pixels) is optional; when given, a reference patch is kept for the debug view.
        """
        # Baseline acquisition
        if self._baseline_mean is None:
            if self._baseline_start is None:
                self._baseline_start = now
            self._baseline_samples.append(mean)
            if patch is not None:
                self._ref_sum = patch.astype(np.float32) + (0 if self._ref_sum is None else self._ref_sum)
                self._ref_n += 1
            if now - self._baseline_start >= cfg.BASELINE_SECONDS:
                arr = np.stack(self._baseline_samples)
                self._baseline_mean = arr.mean(axis=0).astype(np.float32)
                self._baseline_noise = float(np.linalg.norm(arr.std(axis=0)))
                self._baseline_samples = []
                if self._ref_n:
                    self.reference = self._ref_sum / self._ref_n
                    self._ref_sum, self._ref_n = None, 0
                self._last_ts = now
            return self._baseline_mean is not None, False, False

        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        dist = self._distance(mean)
        thr = max(cfg.TRIGGER_DIST_THRESHOLD, cfg.NOISE_SIGMA_MULT * self._baseline_noise)
        self.last_dist, self.last_thr = dist, thr

        # Adaptive baseline: follow slow lighting drift, only while clearly quiet
        if dist < thr and self._active_since is None and cfg.BASELINE_ADAPT_SECONDS > 0 and dt > 0:
            a = min(1.0, dt / cfg.BASELINE_ADAPT_SECONDS)
            self._baseline_mean = (self._baseline_mean + a * (mean - self._baseline_mean)).astype(np.float32)
            if patch is not None and self.reference is not None and patch.shape == self.reference.shape:
                self.reference += a * (patch - self.reference)

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
