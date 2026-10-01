import time
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import qrcode
from PIL import Image

try:
    from PIL import ImageTk
except Exception:
    ImageTk = None

try:
    import tkinter as tk
except Exception:
    tk = None

from constant import *
from log import get_logger

from roi_detector import ROI, decide_roi_cam, map_roi_cam_to_preview, roi_mean_rgb, ROITriggerDetector


# File-based logger (DateBasedFileHandler from log.py)
log = get_logger()


# =========================================================
# Utility: flip
# =========================================================
def apply_flip(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    """Flip helper used for preview/capture."""
    mode = (mode or "").lower().strip()
    if mode == "h":
        return cv2.flip(img_bgr, 1)
    if mode == "v":
        return cv2.flip(img_bgr, 0)
    if mode == "hv":
        return cv2.flip(img_bgr, -1)
    return img_bgr


# =========================================================
# 1) Camera placement inside a canvas (letterbox + anchor)
# =========================================================
class CameraCanvasPlacer:
    """
    Place an RGB frame inside a canvas while keeping aspect ratio.
    Returns the composed canvas and the rectangle (x0, y0, w, h) where the frame was drawn.
    """

    def letterbox_place(
        self,
        frame_rgb: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        anchor: str,
        margin_px: int,
        bg_rgb: Tuple[int, int, int],
    ):
        canvas_w = max(1, int(canvas_w))
        canvas_h = max(1, int(canvas_h))
        margin_px = int(margin_px)

        src_h, src_w = frame_rgb.shape[:2]
        scale = min(canvas_w / max(1, src_w), canvas_h / max(1, src_h))
        fit_w = max(1, int(src_w * scale))
        fit_h = max(1, int(src_h * scale))

        anchor = (anchor or "c").lower().strip()
        if anchor in ("tl", "lt"):
            x0, y0 = margin_px, margin_px
        elif anchor in ("tr", "rt"):
            x0, y0 = canvas_w - fit_w - margin_px, margin_px
        elif anchor in ("bl", "lb"):
            x0, y0 = margin_px, canvas_h - fit_h - margin_px
        elif anchor in ("br", "rb"):
            x0, y0 = canvas_w - fit_w - margin_px, canvas_h - fit_h - margin_px
        else:
            x0, y0 = (canvas_w - fit_w) // 2, (canvas_h - fit_h) // 2

        # Clamp (in case margin is too big)
        x0 = max(0, min(x0, canvas_w - fit_w))
        y0 = max(0, min(y0, canvas_h - fit_h))

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = np.array(bg_rgb, dtype=np.uint8)

        interp = cv2.INTER_AREA if (fit_w <= src_w and fit_h <= src_h) else cv2.INTER_LINEAR
        resized = cv2.resize(frame_rgb, (fit_w, fit_h), interpolation=interp)

        canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
        return canvas, (x0, y0, fit_w, fit_h)


def compute_layout(win_w: int, win_h: int) -> dict:
    """
    Compute QR and preview layout using constants.
    Returns a dict with:
      - qr_size, qr_canvas_h
      - display_w, display_h (preview area)
      - preview_w, preview_h (processing size)
      - gap
    """
    win_w = max(320, int(win_w))
    win_h = max(240, int(win_h))

    gap = int(QR_GAP)

    # QR size
    if str(QR_SIZE_MODE).lower() == "fixed":
        qr_size = int(QR_FIXED_SIZE_PX)
    else:
        # Auto sizing: keep QR items visible with margins
        max_qr = int((win_w - (QR_HISTORY + 1) * gap) / max(1, QR_HISTORY))
        qr_size = int(max(QR_SIZE_MIN, min(QR_SIZE_MAX, max_qr)))

    qr_canvas_h = qr_size + 2 * gap
    preview_h = max(200, win_h - qr_canvas_h)

    return {
        "qr_size": qr_size,
        "qr_canvas_h": qr_canvas_h,
        "display_w": win_w,
        "display_h": preview_h,
        "preview_w": int(PREVIEW_W),
        "preview_h": int(PREVIEW_H),
        "gap": gap,
    }


# =========================================================
# 2) QR helpers (generation + history strip)
# =========================================================
def make_qr_image(url: str) -> Image.Image:
    """
    Build a QR PIL image for the given URL.
    Resizing is handled by QRStrip (so resizing stays in graphics.py only).
    """
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


class QRStrip:
    """
    Manage the bottom QR strip (history) in a Tkinter Canvas.
    Handles dynamic sizing + positioning + animation. Resizing is done here.
    """

    def __init__(self, root, canvas):
        if tk is None or ImageTk is None:
            raise RuntimeError("QRStrip requires Tkinter + PIL.ImageTk.")
        self.root = root
        self.canvas = canvas

        self.qr_size = int(QR_FIXED_SIZE_PX)
        self.gap = int(QR_GAP)

        self._qr_items: List[Dict] = []
        self._animating = False

    def apply_layout(self, win_w: int, qr_size: int, qr_canvas_h: int, gap: int):
        self.qr_size = int(qr_size)
        self.gap = int(gap)
        try:
            self.canvas.config(width=int(win_w), height=int(qr_canvas_h), bg=QR_BAR_BG)
        except Exception:
            log.exception("QRStrip.apply_layout failed")
        self.reset()

    def reset(self):
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        self._qr_items.clear()
        self._animating = False

    def _target_centers(self):
        w = max(1, int(self.canvas.winfo_width() or 1))
        gap = int(self.gap)
        size = int(self.qr_size)

        total = QR_HISTORY * size + (QR_HISTORY + 1) * gap
        align = str(QR_STRIP_ALIGN).lower()
        margin = int(QR_STRIP_MARGIN_PX)

        if align == "left":
            left = max(0, margin)
        elif align == "right":
            left = max(0, int(w - total - margin))
        else:
            left = max(0, int((w - total) / 2) + margin)

        centers = []
        for i in range(QR_HISTORY):
            x_left = left + gap + i * (size + gap)
            centers.append(x_left + size // 2)

        cy = gap + size // 2
        return centers, cy

    def _animate_to_targets(self, start_positions, target_positions, on_done=None):
        self._animating = True
        steps = int(QR_ANIM_STEPS)
        delay = int(QR_ANIM_DELAY_MS)

        def step(k):
            t = (k / steps) if steps > 0 else 1.0
            _, cy = self._target_centers()
            for idx, item in enumerate(self._qr_items):
                x0 = start_positions[idx]
                x1 = target_positions[idx]
                x = x0 + (x1 - x0) * t
                self.canvas.coords(item["id"], x, cy)

            if k < steps:
                self.root.after(delay, lambda: step(k + 1))
            else:
                self._animating = False
                if on_done:
                    on_done()

        step(0)

    def push(self, qr_pil_img: Image.Image):
        size = int(self.qr_size)
        qr_pil_img = qr_pil_img.resize((size, size))
        qr_tk = ImageTk.PhotoImage(qr_pil_img)

        centers, cy = self._target_centers()
        start_x_new = -size // 2
        new_id = self.canvas.create_image(start_x_new, cy, image=qr_tk)
        self._qr_items.insert(0, {"id": new_id, "img": qr_tk})

        to_remove = None
        if len(self._qr_items) > QR_HISTORY:
            to_remove = self._qr_items[-1]

        start_positions = [self.canvas.coords(it["id"])[0] for it in self._qr_items]
        target_positions = []
        for i in range(len(self._qr_items)):
            if i < QR_HISTORY:
                target_positions.append(centers[i])
            else:
                target_positions.append(int(self.canvas.winfo_width() or 1) + size)

        def cleanup():
            nonlocal to_remove
            if to_remove is not None:
                try:
                    self.canvas.delete(to_remove["id"])
                except Exception:
                    pass
                try:
                    self._qr_items.remove(to_remove)
                except ValueError:
                    pass

            # Snap remaining
            centers2, cy2 = self._target_centers()
            for i, it in enumerate(self._qr_items[:QR_HISTORY]):
                self.canvas.coords(it["id"], centers2[i], cy2)

        if self._animating:
            cleanup()
            return

        self._animate_to_targets(start_positions, target_positions, on_done=cleanup)


# =========================================================
# 3) ROI manager (placement + trigger + drawing)
# =========================================================
def _draw_roi_shape(preview_bgr: np.ndarray, roi: ROI, active: bool) -> np.ndarray:
    """Draw the ROI shape on the preview frame (rect or circle)."""
    if not DRAW_ROI_RECT:
        return preview_bgr

    color = (0, 255, 0) if active else (255, 0, 0)

    x1, y1 = int(roi.x), int(roi.y)
    x2, y2 = int(roi.x + roi.w), int(roi.y + roi.h)

    if roi.shape == "circle":
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r = int(min(roi.w, roi.h) / 2)
        cv2.circle(preview_bgr, (cx, cy), max(1, r), color, 2)
    else:
        cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), color, 2)

    return preview_bgr


class ROIManager:
    """
    High-level ROI helper used by main.py.

    - Uses roi_detector.py to decide ROI placement in camera coords
    - Maps ROI to preview coords
    - Computes mean in ROI
    - Maintains baseline + trigger logic
    - Provides a single call per frame: process_frame(...)
    """

    def __init__(self):
        self.detector = ROITriggerDetector()
        self._baseline_logged = False

        # Optional "disable after capture" comes from constant.py; default to 0 if missing
        try:
            self._disable_after_capture_s = float(ROI_DISABLE_AFTER_CAPTURE_S)  # type: ignore
        except Exception:
            self._disable_after_capture_s = 0.0

    def on_capture_done(self, now: float):
        """Disable ROI triggers for a while after a successful capture."""
        if self._disable_after_capture_s > 0:
            self.detector.disable_for(self._disable_after_capture_s, now)

    def process_frame(
        self,
        frame_w: int,
        frame_h: int,
        preview_w: int,
        preview_h: int,
        preview_bgr: np.ndarray,
        now: float,
        allow_trigger: bool,
    ) -> Tuple[np.ndarray, bool]:
        """
        Process ROI for one preview frame.

        Returns:
          (preview_bgr_with_roi_drawn, should_trigger)
        """
        roi_cam = decide_roi_cam(frame_w, frame_h)
        roi_preview = map_roi_cam_to_preview(roi_cam, frame_w, frame_h, preview_w, preview_h)

        mean = roi_mean_rgb(preview_bgr, roi_preview)
        baseline_ready, roi_enabled, should_trigger, _dist = self.detector.update(mean, now, allow_trigger=allow_trigger)

        if baseline_ready and not self._baseline_logged:
            log.info("ROI baseline is ready.")
            self._baseline_logged = True

        active = bool(baseline_ready and roi_enabled and allow_trigger)
        preview_bgr = _draw_roi_shape(preview_bgr, roi_preview, active=active)
        return preview_bgr, bool(should_trigger)


# =========================================================
# 4) Countdown + flash (moved from main.py)
# =========================================================
class CountdownController:
    """Owns countdown/flash state and draws it on the preview frame."""

    def __init__(self):
        self._countdown_text: Optional[str] = None
        self._flash_until: float = 0.0
        self._flash_color: str = "white"

    def start(self, root, seconds: int, flash_duration_s: float, flash_color: str, on_after_flash):
        """
        Schedule countdown numbers, then flash, then call on_after_flash().

        This function must be called from the Tk thread.
        """
        seconds = max(1, int(seconds))
        self._flash_color = str(flash_color or "white")

        def show_number(n: int):
            self._countdown_text = str(n)

        def clear_countdown():
            self._countdown_text = None

        def do_flash():
            clear_countdown()
            self._flash_until = time.monotonic() + float(flash_duration_s)

        for i in range(seconds, 0, -1):
            delay_ms = (seconds - i) * 1000
            root.after(delay_ms, lambda n=i: show_number(n))

        root.after(seconds * 1000, do_flash)
        root.after(int(seconds * 1000 + float(flash_duration_s) * 1000), on_after_flash)

    def apply(self, preview_bgr: np.ndarray, now: float) -> np.ndarray:
        """Apply flash and countdown overlay to the preview frame."""
        if float(now) < float(self._flash_until):
            if str(self._flash_color).lower() == "black":
                preview_bgr[:] = 0
            else:
                preview_bgr[:] = 255

        if self._countdown_text is not None:
            text = str(self._countdown_text)
            h, w = preview_bgr.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(1.0, min(w, h) / 250.0)
            thickness = max(2, int(scale * 2.5))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int((w - tw) / 2)
            y = int((h + th) / 2)
            cv2.putText(preview_bgr, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
            cv2.putText(preview_bgr, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return preview_bgr


# =========================================================
# 5) Frame + logo overlays (burned into saved/uploaded photo)
# =========================================================
def overlay_rgba(dst_rgb: np.ndarray, src_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlay an RGBA image onto an RGB destination at (x, y)."""
    if src_rgba is None:
        return dst_rgb

    h, w = src_rgba.shape[:2]
    H, W = dst_rgb.shape[:2]

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return dst_rgb

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = dst_rgb[y1:y2, x1:x2]
    src = src_rgba[sy1:sy2, sx1:sx2]

    if src.shape[2] == 3:
        roi[:] = src
        dst_rgb[y1:y2, x1:x2] = roi
        return dst_rgb

    src_rgb = src[:, :, :3].astype(np.float32)
    alpha = (src[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    roi_f = roi.astype(np.float32)

    out = alpha * src_rgb + (1.0 - alpha) * roi_f
    dst_rgb[y1:y2, x1:x2] = out.astype(np.uint8)
    return dst_rgb

def apply_custom_template_with_video(frame_rgb: np.ndarray, template_path: str) -> np.ndarray:
    """
    Blends the camera video inside the exact pixel coordinates of the template.
    Fixes the camera blue tint by converting ONLY the video crop to match the BGR template.
    """
    if not os.path.exists(template_path):
        return frame_rgb

    # 1. טעינת קובץ הגרפיקה הכתום (נטען כ-BGR, נשאר כחול/כתום נכון ומקורי לפי התמונה)
    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        return frame_rgb

    # 2. וידוא מימדי התבנית המקורית (900x506) כדי למלא את המסך
    target_w = 900
    target_h = 506
    if template_bgr.shape[1] != target_w or template_bgr.shape[0] != target_h:
        template_bgr = cv2.resize(template_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 3. המיקומים המדויקים בפיקסלים בתוך קובץ ה-PNG עבור המצלמה
    y_min, x_min = 90, 85
    y_max, x_max = 410, 620

    box_w = int(x_max - x_min)
    box_h = int(y_max - y_min)

    # 4. Center Crop - חיתוך פרופורציונלי של המצלמה (frame_rgb) למניעת מריחה
    f_h, f_w = frame_rgb.shape[:2]
    target_aspect = box_w / box_h
    frame_aspect = f_w / f_h

    if frame_aspect > target_aspect:
        new_w = int(f_h * target_aspect)
        start_x = int((f_w - new_w) // 2)
        cropped_video = frame_rgb[:, start_x:start_x + new_w]
    else:
        new_h = int(f_w / target_aspect)
        start_y = int((f_h - new_h) // 2)
        cropped_video = frame_rgb[start_y:start_y + new_h, :]

    # 5. שינוי גודל המצלמה לאיכות חדה
    resized_video = cv2.resize(cropped_video, (box_w, box_h), interpolation=cv2.INTER_CUBIC)

    # 6. התיקון הקריטי: המרת ריבוע הווידאו בלבד מ-RGB ל-BGR כדי שיתאים לרקע הכתום
    resized_video_bgr = cv2.cvtColor(resized_video, cv2.COLOR_RGB2BGR)

    # 7. השתלת המצלמה המתוקנת לתוך התבנית
    output_image = template_bgr.copy()
    output_image[y_min:y_max, x_min:x_max] = resized_video_bgr

    return output_image
    
def apply_frame_and_logo(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Main entry point for overlays. Resolves whether to use the custom full-screen template
    or fallback/resize the camera feed to prevent numpy shape mismatch crashes in main.py.
    """
    # ---- Custom Template Mode ----
    if USE_CUSTOM_TEMPLATE_MODE:
        if os.path.exists(MOCKUP_PNG):
            combined_result = apply_custom_template_with_video(frame_rgb, MOCKUP_PNG)
            if combined_result is not None:
                return combined_result
        else:
            log.warning("Custom mode enabled but asset missing: %s. Falling back to clean resize.", MOCKUP_PNG)

    # ---- Clean Live Preview / Fallback Mode ----
    # If custom template is disabled or missing, stretch the clean camera feed
    # to perfectly match PREVIEW_W and PREVIEW_H so np.vstack doesn't crash.
    target_w = int(PREVIEW_W)
    target_h = int(PREVIEW_H)
    
    if frame_rgb.shape[1] != target_w or frame_rgb.shape[0] != target_h:
        frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
    return frame_rgb

def apply_frame_and_logo_in_rect(canvas_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Apply overlays ONLY inside a rectangle of a canvas (useful when preview is letterboxed)."""
    Hc, Wc = canvas_rgb.shape[:2]
    x = max(0, min(int(x), Wc - 1))
    y = max(0, min(int(y), Hc - 1))
    w = max(1, min(int(w), Wc - x))
    h = max(1, min(int(h), Hc - y))

    sub = canvas_rgb[y:y + h, x:x + w]
    sub2 = apply_frame_and_logo(sub)
    canvas_rgb[y:y + h, x:x + w] = sub2
    return canvas_rgb
