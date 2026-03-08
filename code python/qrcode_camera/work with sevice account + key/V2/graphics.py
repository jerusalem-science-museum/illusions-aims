import cv2
import numpy as np
import qrcode
from PIL import Image, ImageTk
import tkinter as tk

from constant import *
from log import get_logger

# File-based logger (DateBasedFileHandler from log.py)
log = get_logger()


# =========================================================
# OUTILS: flip
# =========================================================
def apply_flip(img_bgr, mode: str):
    if mode == "h":
        return cv2.flip(img_bgr, 1)
    if mode == "v":
        return cv2.flip(img_bgr, 0)
    if mode == "hv":
        return cv2.flip(img_bgr, -1)
    return img_bgr


# =========================================================
# 1) Placement de la caméra dans un canevas (letterbox + anchor)
# =========================================================
class CameraCanvasPlacer:
    """
    Place a frame inside a canvas while keeping aspect ratio.
    Returns the composed canvas and the rectangle (x0, y0, w, h) where the frame was drawn.
    """

    def letterbox_place(
        self,
        frame_bgr: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        anchor: str,
        margin_px: int,
        bg_bgr: tuple[int, int, int],
    ):
        canvas_w = max(1, int(canvas_w))
        canvas_h = max(1, int(canvas_h))
        margin_px = int(margin_px)

        src_h, src_w = frame_bgr.shape[:2]
        scale = min(canvas_w / max(1, src_w), canvas_h / max(1, src_h))
        fit_w = max(1, int(src_w * scale))
        fit_h = max(1, int(src_h * scale))

        anchor = (anchor or "center").lower().strip()
        if anchor == "tl":
            x0, y0 = margin_px, margin_px
        elif anchor == "tr":
            x0, y0 = canvas_w - fit_w - margin_px, margin_px
        elif anchor == "bl":
            x0, y0 = margin_px, canvas_h - fit_h - margin_px
        elif anchor == "br":
            x0, y0 = canvas_w - fit_w - margin_px, canvas_h - fit_h - margin_px
        else:
            x0, y0 = (canvas_w - fit_w) // 2, (canvas_h - fit_h) // 2

        # Clamp (in case margin is too big)
        x0 = max(0, min(x0, canvas_w - fit_w))
        y0 = max(0, min(y0, canvas_h - fit_h))

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = np.array(bg_bgr, dtype=np.uint8)

        interp = cv2.INTER_AREA if (fit_w <= src_w and fit_h <= src_h) else cv2.INTER_LINEAR
        resized = cv2.resize(frame_bgr, (fit_w, fit_h), interpolation=interp)

        canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
        return canvas, (x0, y0, fit_w, fit_h)


def compute_layout(win_w: int, win_h: int):
    """
    Compute QR and preview layout using constants.
    Returns a dict with:
      - qr_size, qr_canvas_h
      - display_w, display_h (preview area)
      - preview_w, preview_h (processing size)
    """
    win_w = max(320, int(win_w))
    win_h = max(240, int(win_h))

    gap = int(QR_GAP)
    if str(QR_SIZE_MODE).lower() == "fixed":
        qr_size = int(QR_FIXED_SIZE_PX)
    else:
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
# 2) Fonctions QR rassemblées (génération + bandeau historique)
# =========================================================
def make_qr_image(url: str, size: int = 260) -> Image.Image:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size))


class QRStrip:
    """
    Manage the bottom QR strip (history) in Tkinter Canvas.
    Handles dynamic sizing + positioning + animation.
    """

    def __init__(self, root: tk.Tk, canvas: tk.Canvas):
        self.root = root
        self.canvas = canvas

        self.qr_size = int(QR_FIXED_SIZE_PX)
        self.gap = int(QR_GAP)

        self._qr_items: list[dict] = []
        self._animating = False

    def apply_layout(self, win_w: int, qr_size: int, qr_canvas_h: int, gap: int):
        self.qr_size = int(qr_size)
        self.gap = int(gap)
        try:
            self.canvas.config(width=int(win_w), height=int(qr_canvas_h), bg=QR_BAR_BG)
        except Exception:
            log.exception("QRStrip.apply_layout failed")

        # When resizing, reset to avoid stretched / misaligned images.
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
# 3) Création du ROI et sa position dans la caméra
# =========================================================
def roi_cam_to_preview(frame_w: int, frame_h: int, preview_w: int, preview_h: int):
    """
    ROI_* are defined in CAMERA coordinates (FRAME_WIDTH/FRAME_HEIGHT).
    This maps them to current preview (processing) size.
    """
    if frame_w <= 0 or frame_h <= 0:
        return 0, 0, 1, 1

    sx = preview_w / frame_w
    sy = preview_h / frame_h

    rx = max(0, min(frame_w - 1, int(ROI_X)))
    ry = max(0, min(frame_h - 1, int(ROI_Y)))
    rw = max(1, min(frame_w - rx, int(ROI_W)))
    rh = max(1, min(frame_h - ry, int(ROI_H)))

    x = int(rx * sx)
    y = int(ry * sy)
    w = max(1, int(rw * sx))
    h = max(1, int(rh * sy))
    return x, y, w, h


def get_roi_mean_bgr(preview_bgr, roi_x=ROI_X, roi_y=ROI_Y, roi_w=ROI_W, roi_h=ROI_H) -> np.ndarray:
    H, W = preview_bgr.shape[:2]
    x1 = max(0, min(W - 1, int(roi_x)))
    y1 = max(0, min(H - 1, int(roi_y)))
    x2 = max(0, min(W, x1 + int(roi_w)))
    y2 = max(0, min(H, y1 + int(roi_h)))

    roi = preview_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return roi.reshape(-1, 3).mean(axis=0).astype(np.float32)


def draw_roi_rect(preview_bgr: np.ndarray, roi_x: int, roi_y: int, roi_w: int, roi_h: int, active: bool):
    """
    Draw ROI rectangle on the preview frame.
    """
    if not DRAW_ROI_RECT:
        return preview_bgr
    color = (0, 255, 0) if active else (0, 0, 255)
    x1, y1 = int(roi_x), int(roi_y)
    x2, y2 = int(roi_x + roi_w), int(roi_y + roi_h)
    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), color, 2)
    return preview_bgr


# =========================================================
# 4) Création + emplacement du logo + frame (sur capture et/ou preview)
# =========================================================
def overlay_rgba(dst_bgr, src_rgba, x, y):
    if src_rgba is None:
        return dst_bgr

    h, w = src_rgba.shape[:2]
    H, W = dst_bgr.shape[:2]

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return dst_bgr

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = dst_bgr[y1:y2, x1:x2]
    src = src_rgba[sy1:sy2, sx1:sx2]

    if src.shape[2] == 3:
        roi[:] = src
        dst_bgr[y1:y2, x1:x2] = roi
        return dst_bgr

    src_rgb = src[:, :, :3].astype(np.float32)
    alpha = (src[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    roi_f = roi.astype(np.float32)

    out = alpha * src_rgb + (1.0 - alpha) * roi_f
    dst_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return dst_bgr


def apply_frame_and_logo(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Add the decorative frame + logo overlays on top of an image.

    - Works on any resolution (camera capture, preview, etc.)
    - Logo size is relative to the image width (LOGO_SCALE) OR fixed pixels
    - Position anchor or manual
    - Uses caching for performance
    """
    H, W = frame_bgr.shape[:2]

    if not hasattr(apply_frame_and_logo, "_frame_cache"):
        apply_frame_and_logo._frame_cache = {}  # (W,H)->rgba
        apply_frame_and_logo._logo_orig = None  # original rgba
        apply_frame_and_logo._logo_cache = {}   # target_w->rgba

    # frame overlay
    frame_rgba = apply_frame_and_logo._frame_cache.get((W, H))
    if frame_rgba is None:
        src_rgba = cv2.imread(FRAME_PNG, cv2.IMREAD_UNCHANGED)
        if src_rgba is not None:
            frame_rgba = cv2.resize(src_rgba, (W, H), interpolation=cv2.INTER_AREA)
            apply_frame_and_logo._frame_cache[(W, H)] = frame_rgba
        else:
            log.warning("FRAME_PNG introuvable: %s", FRAME_PNG)

    if frame_rgba is not None:
        frame_bgr = overlay_rgba(frame_bgr, frame_rgba, 0, 0)

    # logo overlay
    if apply_frame_and_logo._logo_orig is None:
        apply_frame_and_logo._logo_orig = cv2.imread(LOGO_PNG, cv2.IMREAD_UNCHANGED)
        if apply_frame_and_logo._logo_orig is None:
            log.warning("LOGO_PNG introuvable: %s", LOGO_PNG)

    logo_rgba = apply_frame_and_logo._logo_orig
    if logo_rgba is None:
        return frame_bgr

    if str(LOGO_SIZE_MODE).lower() == "pixels" and LOGO_TARGET_W_PX is not None:
        target_w = max(1, int(LOGO_TARGET_W_PX))
    else:
        target_w = max(1, int(W * float(LOGO_SCALE)))
    target_w = min(target_w, W)

    cached = apply_frame_and_logo._logo_cache.get(target_w)
    if cached is None:
        ratio = target_w / max(1, logo_rgba.shape[1])
        target_h = max(1, int(logo_rgba.shape[0] * ratio))
        target_h = min(target_h, H)
        cached = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        apply_frame_and_logo._logo_cache[target_w] = cached
    else:
        target_h = cached.shape[0]

    avail_x = max(0, W - target_w)
    avail_y = max(0, H - target_h)

    def _pos(val, avail, margin, anchor_side):
        if val is None:
            if anchor_side == "min":
                return int(margin)
            if anchor_side == "center":
                return int(avail / 2)
            return int(avail - margin)
        try:
            v = float(val)
        except Exception:
            v = 0.0
        if 0.0 <= v <= 1.0:
            return int(v * avail)
        return int(v)

    anchor = str(LOGO_ANCHOR).lower().strip()
    if anchor in ("br", "rb"):
        ax, ay = "max", "max"
    elif anchor in ("tr", "rt"):
        ax, ay = "max", "min"
    elif anchor in ("bl", "lb"):
        ax, ay = "min", "max"
    elif anchor in ("tl", "lt"):
        ax, ay = "min", "min"
    else:
        ax, ay = "center", "center"

    x = _pos(LOGO_POS_X, avail_x, LOGO_MARGIN_X, ax)
    y = _pos(LOGO_POS_Y, avail_y, LOGO_MARGIN_Y, ay)

    x = max(0, min(x, avail_x))
    y = max(0, min(y, avail_y))

    frame_bgr = overlay_rgba(frame_bgr, cached, x, y)
    return frame_bgr


def apply_frame_and_logo_in_rect(canvas_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Apply overlays ONLY inside a rectangle of a canvas (useful when preview is letterboxed).
    """
    Hc, Wc = canvas_bgr.shape[:2]
    x = max(0, min(int(x), Wc - 1))
    y = max(0, min(int(y), Hc - 1))
    w = max(1, min(int(w), Wc - x))
    h = max(1, min(int(h), Hc - y))

    sub = canvas_bgr[y:y + h, x:x + w]
    sub2 = apply_frame_and_logo(sub)
    canvas_bgr[y:y + h, x:x + w] = sub2
    return canvas_bgr
