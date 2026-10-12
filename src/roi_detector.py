"""ROI trigger: fires when the ROI's pixels stay different enough from a reference patch for long enough."""
from typing import Optional, Tuple

import cv2
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


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two gray patches (1 = same structure)."""
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


class ROITriggerDetector:
    """
    - Collect a reference patch for BASELINE_SECONDS, plus each pixel's noise (std over time)
    - Per frame, two signals against the reference:
        changed: fraction of pixels whose largest channel difference exceeds
                 max(PIXEL_DIFF_MIN, PIXEL_SIGMA_MULT * that pixel's noise)
        ncc:     gray correlation with the reference (ignores gain/offset); only used when the
                 reference has enough texture (gray std >= NCC_MIN_STD)
    - score = max(changed / CHANGED_FRAC_THRESHOLD, (1 - ncc) / (1 - NCC_THRESHOLD)); covered when score >= 1
    - Trigger when covered for HOLD_SECONDS (short dips are tolerated)
    - Then COOLDOWN_SECONDS, plus any disable window set by disable_for()
    """

    def __init__(self):
        self._baseline_start: Optional[float] = None
        self._sum: Optional[np.ndarray] = None
        self._sum_sq: Optional[np.ndarray] = None
        self._n = 0
        self.reference: Optional[np.ndarray] = None   # float32 HxWx3, follows slow drift
        self._pix_thr: Optional[np.ndarray] = None    # float32 HxW, per-pixel change threshold
        self._baseline_noise = 0.0                    # mean per-pixel std, for the debug view
        self.last_dist = 0.0       # score (see class doc)
        self.last_thr = 1.0
        self.last_changed = 0.0
        self.last_ncc: Optional[float] = None  # None when the reference is too flat for NCC
        self._last_ts: Optional[float] = None
        self._below_since: Optional[float] = None
        self._active_since: Optional[float] = None
        self._cooldown_until = 0.0
        self._disabled_until = 0.0

    def disable_for(self, seconds: float, now: float) -> None:
        self._disabled_until = max(self._disabled_until, now + seconds)

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        return None if self.reference is None else self.reference.reshape(-1, 3).mean(axis=0)

    @property
    def baseline_noise(self) -> float:
        return self._baseline_noise

    def _score(self, cur: np.ndarray) -> float:
        ref = self.reference
        if cfg.BRIGHTNESS_COMP:
            # Gain-style exposure drift scales all channels proportionally
            ref = ref * (float(cur.mean()) / (float(ref.mean()) + 1e-6))
        diff = np.abs(cur - ref).max(axis=2)
        self.last_changed = float((diff > self._pix_thr).mean())
        score = self.last_changed / cfg.CHANGED_FRAC_THRESHOLD

        ref_gray = self.reference.mean(axis=2)
        if ref_gray.std() >= cfg.NCC_MIN_STD:
            self.last_ncc = _ncc(cur.mean(axis=2), ref_gray)
            score = max(score, (1.0 - self.last_ncc) / (1.0 - cfg.NCC_THRESHOLD))
        else:
            self.last_ncc = None
        return score

    def update(self, patch: np.ndarray, now: float, allow_trigger: bool) -> Tuple[bool, bool, bool]:
        """patch: the ROI pixels (uint8 BGR). Returns (baseline_ready, roi_enabled, should_trigger);
        should_trigger is True once per trigger."""
        cur = patch.astype(np.float32)
        if cfg.ROI_BLUR_KSIZE > 1:
            cur = cv2.GaussianBlur(cur, (cfg.ROI_BLUR_KSIZE, cfg.ROI_BLUR_KSIZE), 0)

        # Baseline acquisition
        if self.reference is None:
            if self._baseline_start is None:
                self._baseline_start = now
                self._sum = np.zeros_like(cur)
                self._sum_sq = np.zeros_like(cur)
            self._sum += cur
            self._sum_sq += cur * cur
            self._n += 1
            if now - self._baseline_start >= cfg.BASELINE_SECONDS:
                mean = self._sum / self._n
                std = np.sqrt(np.maximum(0.0, self._sum_sq / self._n - mean * mean)).max(axis=2)
                self.reference = mean
                self._pix_thr = np.maximum(cfg.PIXEL_DIFF_MIN, cfg.PIXEL_SIGMA_MULT * std).astype(np.float32)
                self._baseline_noise = float(std.mean())
                self._sum = self._sum_sq = None
                self._last_ts = now
            return self.reference is not None, False, False

        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        score = self._score(cur)
        self.last_dist = score

        # Adaptive baseline: follow slow lighting drift, only while clearly quiet
        if score < 1.0 and self._active_since is None and cfg.BASELINE_ADAPT_SECONDS > 0 and dt > 0:
            a = min(1.0, dt / cfg.BASELINE_ADAPT_SECONDS)
            self.reference += a * (cur - self.reference)

        enabled = allow_trigger and now >= self._cooldown_until and now >= self._disabled_until
        if not enabled:
            self._active_since = None
            self._below_since = None
            return True, False, False

        if score >= 1.0:
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
