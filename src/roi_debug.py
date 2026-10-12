"""
ROI debug view (main.py --debug): compares the detector's reference ROI with the current ROI.

- An on-screen panel: the reference and current ROI pixels side by side, their per-channel
  histograms overlaid, and the change metrics (the detector's mean distance plus alternatives).
- A CSV time series in LOG_FOLDER (roi_stats_<start>.csv) to see lighting drift over a day.
- On every trigger, a PNG of reference | current | |diff| in LOG_FOLDER/roi_triggers/.

Metrics (all on the ROI only, so they are cheap):
- dist / thr: what the detector actually uses (RGB distance of the means vs its threshold)
- bhatt: mean Bhattacharyya distance of the per-channel histograms (0 = same, 1 = disjoint)
- changed: fraction of pixels whose largest channel difference exceeds DEBUG_PIXEL_DIFF
- ncc: normalized cross-correlation of the gray patches (1 = same structure; ignores gain/offset)
"""
import csv
import datetime
import os
from typing import Optional

import cv2
import numpy as np

import constant as cfg
from log import get_logger

log = get_logger()

_CH_COLORS = ((255, 120, 0), (0, 200, 0), (0, 0, 255))  # B, G, R channel plot colors (BGR)


def _hist(patch: np.ndarray) -> np.ndarray:
    """Per-channel normalized histograms, shape (3, bins), float32."""
    bins = cfg.DEBUG_HIST_BINS
    h = np.stack([cv2.calcHist([patch], [c], None, [bins], [0, 256]).ravel() for c in range(3)])
    return (h / max(1.0, h.sum(axis=1, keepdims=True).max())).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    va, vb = float((a * a).sum()), float((b * b).sum())
    flat = 1e-3 * a.size  # a flat patch (std < ~0.03) has no structure to correlate
    if va < flat or vb < flat:
        return 1.0 if va < flat and vb < flat else 0.0
    return float((a * b).sum() / np.sqrt(va * vb))


class ROIDebug:
    def __init__(self, detector):
        self.detector = detector
        self.stats: dict = {}
        self._ref_hist: Optional[np.ndarray] = None
        self._cur_hist: Optional[np.ndarray] = None
        self._ref_img: Optional[np.ndarray] = None
        self._cur_img: Optional[np.ndarray] = None
        self._next_csv = 0.0

        os.makedirs(cfg.LOG_FOLDER, exist_ok=True)
        self.trigger_dir = os.path.join(cfg.LOG_FOLDER, "roi_triggers")
        os.makedirs(self.trigger_dir, exist_ok=True)
        start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(cfg.LOG_FOLDER, f"roi_stats_{start}.csv")
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8", buffering=1)
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["time", "t", "state", "cur_b", "cur_g", "cur_r", "base_b", "base_g", "base_r",
                            "dist", "thr", "noise", "bhatt", "changed", "ncc", "cur_luma", "ref_luma"])
        log.info("ROI debug: stats -> %s, trigger snapshots -> %s", self.csv_path, self.trigger_dir)

    def update(self, patch: np.ndarray, now: float, state: str) -> None:
        """Compute the metrics for this frame (call after detector.update) and log them to the CSV."""
        det = self.detector
        ref = det.reference
        self._cur_img = patch
        self._cur_hist = _hist(patch)
        cur_mean = patch.reshape(-1, 3).mean(axis=0)
        s = {"state": state, "cur": cur_mean, "dist": det.last_dist, "thr": det.last_thr,
             "noise": det.baseline_noise, "base": det.baseline_mean,
             "cur_luma": float(cur_mean.mean())}
        if ref is not None and ref.shape == patch.shape:
            ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
            self._ref_img = ref_u8
            self._ref_hist = _hist(ref_u8)
            s["bhatt"] = float(np.mean([cv2.compareHist(self._ref_hist[c], self._cur_hist[c],
                                                        cv2.HISTCMP_BHATTACHARYYA) for c in range(3)]))
            diff = np.abs(patch.astype(np.float32) - ref).max(axis=2)
            s["changed"] = float((diff > cfg.DEBUG_PIXEL_DIFF).mean())
            s["ncc"] = _ncc(patch.astype(np.float32).mean(axis=2), ref.mean(axis=2))
            s["ref_luma"] = float(ref.mean())
        self.stats = s

        if now >= self._next_csv and s["base"] is not None:
            self._next_csv = now + 1.0 / cfg.DEBUG_CSV_HZ
            f = lambda v: "" if v is None else f"{v:.3f}"
            self._csv.writerow([datetime.datetime.now().isoformat(timespec="milliseconds"), f"{now:.3f}", state,
                                *(f"{v:.2f}" for v in s["cur"]), *(f"{v:.2f}" for v in s["base"]),
                                f(s["dist"]), f(s["thr"]), f(s["noise"]), f(s.get("bhatt")),
                                f(s.get("changed")), f(s.get("ncc")), f(s["cur_luma"]), f(s.get("ref_luma"))])

    def on_trigger(self) -> None:
        """Log the metrics at the trigger and save reference | current | diff as a PNG."""
        s = self.stats
        log.info("ROI trigger stats: dist=%.1f thr=%.1f bhatt=%s changed=%s ncc=%s cur=%s base=%s",
                 s.get("dist", 0), s.get("thr", 0), _fmt(s.get("bhatt")), _fmt(s.get("changed")),
                 _fmt(s.get("ncc")), np.round(s.get("cur"), 1), np.round(s.get("base"), 1))
        if self._ref_img is None or self._cur_img is None:
            return
        diff = cv2.absdiff(self._cur_img, self._ref_img)
        strip = np.hstack([self._ref_img, self._cur_img, diff])
        strip = cv2.resize(strip, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        name = datetime.datetime.now().strftime("trigger_%Y%m%d_%H%M%S_%f.png")
        cv2.imwrite(os.path.join(self.trigger_dir, name), strip)

    def draw(self, img: np.ndarray) -> None:
        """Draw the panel in the top-left corner of img (the window-sized video)."""
        s = self.stats
        if not s:
            return
        pw, ph, pad = 440, 290, 10
        if img.shape[0] < ph or img.shape[1] < pw:
            return
        panel = img[:ph, :pw]
        panel[:] = (panel * 0.25).astype(np.uint8)  # darken under the panel

        # Reference and current ROI pixels, scaled up
        tile = 96
        for i, (label, im) in enumerate((("reference", self._ref_img), ("current", self._cur_img))):
            x = pad + i * (tile + pad)
            if im is not None:
                panel[pad + 16:pad + 16 + tile, x:x + tile] = cv2.resize(im, (tile, tile),
                                                                         interpolation=cv2.INTER_NEAREST)
            _text(panel, label, (x, pad + 11))

        # Histograms: reference thin, current thick
        hx, hy, hw, hh = 2 * tile + 3 * pad, pad + 16, pw - (2 * tile + 4 * pad), tile
        cv2.rectangle(panel, (hx, hy), (hx + hw, hy + hh), (90, 90, 90), 1)
        _text(panel, "hist ref(thin) cur(thick)", (hx, pad + 11))
        for hist, thick in ((self._ref_hist, 1), (self._cur_hist, 2)):
            if hist is None:
                continue
            for c in range(3):
                h = hist[c] / max(1e-6, float(hist[c].max()))
                xs = hx + np.linspace(0, hw, len(h))
                ys = hy + hh - h * (hh - 2)
                pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
                cv2.polylines(panel, [pts], False, _CH_COLORS[c], thick, cv2.LINE_AA)

        # Metrics
        hot, ok = (80, 80, 255), (200, 255, 200)
        above = s["dist"] >= s["thr"] > 0
        lines = [
            (f"state: {s['state']}", (255, 255, 255)),
            (f"mean dist {s['dist']:6.1f}  thr {s['thr']:5.1f}  noise {s['noise']:4.1f}", hot if above else ok),
            (f"cur  BGR {_vec(s['cur'])}   luma {s['cur_luma']:5.1f}", (255, 255, 255)),
            (f"base BGR {_vec(s['base'])}   luma {_fmt(s.get('ref_luma'), 5, 1)}", (255, 255, 255)),
            (f"hist bhatt {_fmt(s.get('bhatt'))}   changed px {_pct(s.get('changed'))}"
             f"   ncc {_fmt(s.get('ncc'))}", (255, 255, 255)),
        ]
        y = hy + hh + 26
        for text, color in lines:
            _text(panel, text, (pad, y), color)
            y += 22

        # Distance vs threshold bar
        bx, by, bw = pad, y, pw - 2 * pad
        full = max(s["thr"] * 2, s["dist"], 1e-6)
        cv2.rectangle(panel, (bx, by), (bx + bw, by + 12), (90, 90, 90), 1)
        cv2.rectangle(panel, (bx, by), (bx + int(bw * min(1.0, s["dist"] / full)), by + 12),
                      hot if above else (0, 180, 0), -1)
        tx = bx + int(bw * s["thr"] / full)
        cv2.line(panel, (tx, by - 3), (tx, by + 15), (255, 255, 255), 2)

    def close(self) -> None:
        self._csv_file.close()


def _text(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _fmt(v, width=5, prec=3):
    return "  -  " if v is None else f"{v:{width}.{prec}f}"


def _pct(v):
    return "  - " if v is None else f"{100 * v:3.0f}%"


def _vec(v):
    return "   -   " if v is None else " ".join(f"{x:5.1f}" for x in v)
