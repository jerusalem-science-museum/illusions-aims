import os
import threading
import datetime
import socket
import time
import queue
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


# =========================================================
# CONFIG
# =========================================================
CAPTURE_DIR = "captures"
PORT = 8000

PREVIEW_W = 640
PREVIEW_H = 360

FORCE_ADVERTISE_IP = None   # ex: "192.168.1.50"
CAM_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------- PNG OVERLAYS ----------
PIC_DIR = "pic"
FRAME_PNG = os.path.join(PIC_DIR, "frame.png")
LOGO_PNG  = os.path.join(PIC_DIR, "logo.png")

LOGO_SCALE = 0.5
LOGO_POS_X = 230      # None => bas-droite auto
LOGO_POS_Y = 350      # None => bas-droite auto
LOGO_MARGIN_X = 20
LOGO_MARGIN_Y = 20

# ---------- FLIP ----------
FLIP_PREVIEW = True      # flip de l’aperçu
FLIP_CAPTURE = True      # flip de la photo enregistrée
FLIP_MODE = "h"          # "h"=horizontal, "v"=vertical, "hv"=les deux

# ---------- UX CAPTURE ----------
COUNTDOWN_SECONDS = 3
FLASH_DURATION_S = 0.12
FLASH_COLOR = "white"     # "white" ou "black"

# ---------- DÉCLENCHEMENT PAR ROI ----------
ROI_W = 20
ROI_H = 20
ROI_X = 600
ROI_Y = 10

BASELINE_SECONDS = 2.0
TRIGGER_DIST_THRESHOLD = 50.0
HOLD_SECONDS = 0.3          # ✅ doit rester 2s sur le carré (variable)
COOLDOWN_SECONDS = 2.0

# ✅ nouveau: désactive le carré après capture/QR
ROI_DISABLE_AFTER_CAPTURE_S = 10.0

DRAW_ROI_RECT = True

# ---------- QR HISTORY ----------
QR_HISTORY = 4
QR_SIZE = 130
QR_GAP = 10
QR_ANIM_STEPS = 12
QR_ANIM_DELAY_MS = 15
# =========================================================


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
# ÉTAPE 1) appliquer frame + logo sur l'image CAPTURE
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


def apply_frame_and_logo(frame_bgr):
    H, W = frame_bgr.shape[:2]

    frame_rgba = cv2.imread(FRAME_PNG, cv2.IMREAD_UNCHANGED)
    if frame_rgba is not None:
        frame_rgba = cv2.resize(frame_rgba, (W, H), interpolation=cv2.INTER_AREA)
        frame_bgr = overlay_rgba(frame_bgr, frame_rgba, 0, 0)

    logo_rgba = cv2.imread(LOGO_PNG, cv2.IMREAD_UNCHANGED)
    if logo_rgba is not None:
        target_w = max(1, int(W * LOGO_SCALE))
        ratio = target_w / max(1, logo_rgba.shape[1])
        target_h = max(1, int(logo_rgba.shape[0] * ratio))
        logo_rgba = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

        x = (W - target_w - LOGO_MARGIN_X) if LOGO_POS_X is None else int(LOGO_POS_X)
        y = (H - target_h - LOGO_MARGIN_Y) if LOGO_POS_Y is None else int(LOGO_POS_Y)

        frame_bgr = overlay_rgba(frame_bgr, logo_rgba, x, y)

    return frame_bgr


# =========================================================
# ÉTAPE 3) localhost + URL + stockage photo
# =========================================================
def get_local_ip() -> str:
    if FORCE_ADVERTISE_IP:
        return FORCE_ADVERTISE_IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class NoListRequestHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404, "No directory listing")
        return None

    def log_message(self, format, *args):
        return


class LocalFileServer:
    def __init__(self, directory: str, port: int):
        self.directory = os.path.abspath(directory)
        self.port = port
        self._httpd = None
        self._thread = None

    def start(self):
        os.makedirs(self.directory, exist_ok=True)
        handler_cls = NoListRequestHandler

        def handler(*args, **kwargs):
            return handler_cls(*args, directory=self.directory, **kwargs)

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        self._httpd = None
        self._thread = None


class CaptureStorage:
    def __init__(self, capture_dir: str, port: int):
        self.capture_dir = capture_dir
        self.port = port
        os.makedirs(self.capture_dir, exist_ok=True)

        self.server = LocalFileServer(self.capture_dir, self.port)
        self.server.start()

        self.advertise_ip = get_local_ip()
        self.base_url = f"http://{self.advertise_ip}:{self.port}"

    def save_frame(self, frame_bgr) -> str:
        ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")
        filename = f"capture_{ts}.jpg"
        filepath = os.path.join(self.capture_dir, filename)

        cv2.imwrite(filepath, frame_bgr)

        # ✅ URL unique (chaque QR => chaque photo)
        return f"{self.base_url}/{filename}"

    def close(self):
        self.server.stop()


# =========================================================
# ÉTAPE 4) URL -> QR code
# =========================================================
def make_qr_image(url: str, size: int = 260) -> Image.Image:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size))


# =========================================================
# ÉTAPE 5) ROI mean
# =========================================================
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


# =========================================================
# APP
# =========================================================
class CameraAppGUI:
    def __init__(self):
        # Webcam
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam. Change CAM_INDEX (0/1/2).")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Locks/state
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._last_frame_bgr = None
        self._running = True

        # UI safe queue
        self._uiq = queue.Queue()

        # UX capture
        self._countdown_text = None
        self._flash_until = 0.0
        self._flash_color = FLASH_COLOR

        # Baseline
        self._baseline_start = time.monotonic()
        self._baseline_samples = []
        self._baseline_mean = None

        # Hold timer
        self._roi_active_since = None

        # Cooldown / séquence
        self._cooldown_until = 0.0
        self._sequence_running = False

        # ✅ disable ROI after capture
        self._roi_disabled_until = 0.0

        # Storage + server
        self.storage = CaptureStorage(CAPTURE_DIR, PORT)

        # Tkinter UI
        self.root = tk.Tk()
        self.root.title("Camera Local URL → QR (ROI hold-to-trigger)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Initialisation… (laisse le ROI visible)")
        self.url_var = tk.StringVar(value=f"{self.storage.base_url}/...")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Auto CAPTURE via ROI (hold)", font=("Arial", 16)).pack(pady=(0, 6))
        ttk.Label(
            main,
            text="Démarrage: ne cache pas le ROI (baseline). Ensuite pose le doigt sur le ROI et garde-le.",
            wraplength=720
        ).pack(pady=(0, 8))

        self.preview_label = ttk.Label(main)
        self.preview_label.pack(pady=(0, 8))

        ttk.Label(main, textvariable=self.status_var, wraplength=720).pack(pady=(0, 6))
        ttk.Label(main, textvariable=self.url_var, wraplength=720).pack(pady=(0, 8))

        # QR strip
        self.qr_canvas_w = QR_HISTORY * QR_SIZE + (QR_HISTORY + 1) * QR_GAP
        self.qr_canvas_h = QR_SIZE + 2 * QR_GAP
        self.qr_canvas = tk.Canvas(main, width=self.qr_canvas_w, height=self.qr_canvas_h, highlightthickness=0)
        self.qr_canvas.pack(pady=(4, 0))

        self._qr_items = []
        self._qr_animating = False

        self._tk_preview_img = None

        # start loops
        self.root.after(10, self.update_preview)
        self.root.after(25, self._process_ui_queue)

    # ---------- UI queue ----------
    def ui(self, fn):
        self._uiq.put(fn)

    def _process_ui_queue(self):
        if not self._running:
            return
        try:
            while True:
                fn = self._uiq.get_nowait()
                try:
                    fn()
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.root.after(25, self._process_ui_queue)

    # ---------- QR strip helpers ----------
    def _qr_target_centers(self):
        centers = []
        for i in range(QR_HISTORY):
            x_left = QR_GAP + i * (QR_SIZE + QR_GAP)
            centers.append(x_left + QR_SIZE // 2)
        cy = QR_GAP + QR_SIZE // 2
        return centers, cy

    def _animate_qr_to_targets(self, start_positions, target_positions, steps=QR_ANIM_STEPS, delay=QR_ANIM_DELAY_MS, on_done=None):
        self._qr_animating = True

        def step(k):
            t = (k / steps)
            _, cy = self._qr_target_centers()
            for idx, item in enumerate(self._qr_items):
                x0 = start_positions[idx]
                x1 = target_positions[idx]
                x = x0 + (x1 - x0) * t
                self.qr_canvas.coords(item["id"], x, cy)

            if k < steps:
                self.root.after(delay, lambda: step(k + 1))
            else:
                self._qr_animating = False
                if on_done:
                    on_done()

        step(0)

    def push_qr_to_history(self, qr_pil_img: Image.Image):
        qr_pil_img = qr_pil_img.resize((QR_SIZE, QR_SIZE))
        qr_tk = ImageTk.PhotoImage(qr_pil_img)

        centers, cy = self._qr_target_centers()
        start_x_new = -QR_SIZE // 2
        new_id = self.qr_canvas.create_image(start_x_new, cy, image=qr_tk)
        self._qr_items.insert(0, {"id": new_id, "img": qr_tk})

        to_remove = None
        if len(self._qr_items) > QR_HISTORY:
            to_remove = self._qr_items[-1]

        start_positions = [self.qr_canvas.coords(it["id"])[0] for it in self._qr_items]
        target_positions = []
        for i in range(len(self._qr_items)):
            if i < QR_HISTORY:
                target_positions.append(centers[i])
            else:
                target_positions.append(self.qr_canvas_w + QR_SIZE)

        def cleanup():
            nonlocal to_remove
            if to_remove is not None:
                try:
                    self.qr_canvas.delete(to_remove["id"])
                except Exception:
                    pass
                try:
                    self._qr_items.remove(to_remove)
                except ValueError:
                    pass

            centers2, cy2 = self._qr_target_centers()
            for i, it in enumerate(self._qr_items[:QR_HISTORY]):
                self.qr_canvas.coords(it["id"], centers2[i], cy2)

        if self._qr_animating:
            cleanup()
            return

        self._animate_qr_to_targets(start_positions, target_positions, on_done=cleanup)

    # ---------- Preview loop ----------
    def update_preview(self):
        if not self._running:
            return

        ok, frame = self.cap.read()
        if ok and frame is not None:
            with self._frame_lock:
                self._last_frame_bgr = frame

            small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)

            # ✅ flip preview (avant ROI + rectangle)
            if FLIP_PREVIEW:
                small = apply_flip(small, FLIP_MODE)

            now = time.monotonic()
            roi_mean = get_roi_mean_bgr(small)

            # Baseline
            if self._baseline_mean is None:
                elapsed = now - self._baseline_start
                self._baseline_samples.append(roi_mean)
                if elapsed < BASELINE_SECONDS:
                    self.status_var.set(f"Calibration ROI… {int(BASELINE_SECONDS - elapsed + 1)}s (ne cache pas)")
                else:
                    arr = np.stack(self._baseline_samples, axis=0)
                    self._baseline_mean = arr.mean(axis=0)
                    self.status_var.set("Baseline OK ✅ Pose le doigt sur le ROI et garde-le.")
            else:
                # ROI enabled/disabled
                roi_enabled = (
                    (now >= self._cooldown_until) and
                    (not self._sequence_running) and
                    (now >= self._roi_disabled_until)
                )

                if not roi_enabled:
                    self._roi_active_since = None
                else:
                    # Détection HOLD
                    dist = float(np.linalg.norm(roi_mean - self._baseline_mean))
                    if dist >= TRIGGER_DIST_THRESHOLD:
                        if self._roi_active_since is None:
                            self._roi_active_since = now

                        held = now - self._roi_active_since
                        remaining = max(0.0, HOLD_SECONDS - held)
                        if remaining > 0:
                            self.status_var.set(f"Hold… {remaining:.1f}s")
                        else:
                            # déclenche
                            self._roi_active_since = None
                            self._cooldown_until = now + COOLDOWN_SECONDS
                            self.start_countdown_then_capture(COUNTDOWN_SECONDS)
                    else:
                        self._roi_active_since = None

            # Rectangle ROI (vert si actif, rouge si désactivé)
            if DRAW_ROI_RECT:
                roi_is_active = (self._baseline_mean is not None) and (now >= self._roi_disabled_until) and (not self._sequence_running) and (now >= self._cooldown_until)
                color = (0, 255, 0) if roi_is_active else (0, 0, 255)  # BGR
                x1 = int(ROI_X)
                y1 = int(ROI_Y)
                x2 = int(ROI_X + ROI_W)
                y2 = int(ROI_Y + ROI_H)
                cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)

            # flash
            if now < self._flash_until:
                small[:] = 0 if self._flash_color.lower() == "black" else 255

            # countdown
            if self._countdown_text is not None:
                text = str(self._countdown_text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 3.0
                thickness = 8
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                x = int((PREVIEW_W - tw) / 2)
                y = int((PREVIEW_H + th) / 2)
                cv2.putText(small, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
                cv2.putText(small, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self._tk_preview_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self._tk_preview_img)

        self.root.after(33, self.update_preview)

    # ---------- countdown + flash + capture ----------
    def start_countdown_then_capture(self, seconds: int = 3):
        if self._sequence_running:
            return
        if not self._capture_lock.acquire(blocking=False):
            return

        self._sequence_running = True

        def show_number(n: int):
            self._countdown_text = str(n)
            self.status_var.set(f"Capturing in {n}…")

        def clear_countdown():
            self._countdown_text = None

        def do_flash():
            clear_countdown()
            self._flash_until = time.monotonic() + FLASH_DURATION_S

        def start_capture_thread():
            threading.Thread(target=self.capture_flow, daemon=True).start()

        for i in range(seconds, 0, -1):
            delay_ms = (seconds - i) * 1000
            self.root.after(delay_ms, lambda n=i: show_number(n))

        self.root.after(seconds * 1000, do_flash)
        self.root.after(int(seconds * 1000 + FLASH_DURATION_S * 1000), start_capture_thread)

    def capture_flow(self):
        try:
            with self._frame_lock:
                frame = None if self._last_frame_bgr is None else self._last_frame_bgr.copy()

            if frame is None:
                self.ui(lambda: self.status_var.set("No frame yet ❌"))
                return

            # ✅ flip capture
            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)

            # overlays
            frame = apply_frame_and_logo(frame)

            # storage + url unique
            url = self.storage.save_frame(frame)

            # QR
            qr_img = make_qr_image(url, size=260)

            # ✅ disable ROI juste après capture
            disable_until = time.monotonic() + ROI_DISABLE_AFTER_CAPTURE_S

            self.ui(lambda: self.url_var.set(url))
            self.ui(lambda: self.push_qr_to_history(qr_img))
            self.ui(lambda: self.status_var.set("Done ✅ (scan QR)"))
            self._roi_disabled_until = disable_until

        except Exception as e:
            self.ui(lambda: self.status_var.set(f"Failed ❌ ({e})"))
        finally:
            self._sequence_running = False
            try:
                self._capture_lock.release()
            except Exception:
                pass

    def on_close(self):
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.storage.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    CameraAppGUI().run()
