import os
import sys
import threading
import datetime
import time
import queue
from typing import Optional

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

PREVIEW_W = 640
PREVIEW_H = 360

CAM_INDEX = 0
CAMERA_RESOLUTION = (1640, 1232)
FRAME_WIDTH, FRAME_HEIGHT = CAMERA_RESOLUTION

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
FLIP_PREVIEW = True
FLIP_CAPTURE = True
FLIP_MODE = "h"          # "h"=horizontal, "v"=vertical, "hv"=les deux

# ---------- UX CAPTURE ----------
COUNTDOWN_SECONDS = 3
FLASH_DURATION_S = 0.12
FLASH_COLOR = "white"

# ---------- DÉCLENCHEMENT PAR ROI ----------
ROI_W = 20
ROI_H = 20
ROI_X = 600
ROI_Y = 10

BASELINE_SECONDS = 2.0
TRIGGER_DIST_THRESHOLD = 50.0
HOLD_SECONDS = 0.3
COOLDOWN_SECONDS = 2.0

# Désactive le ROI juste après l’envoi du QR
ROI_DISABLE_AFTER_CAPTURE_S = 10.0

DRAW_ROI_RECT = True

# ---------- QR HISTORY ----------
QR_HISTORY = 4
QR_SIZE = 130
QR_GAP = 10
QR_ANIM_STEPS = 12
QR_ANIM_DELAY_MS = 15

# ---------- GOOGLE DRIVE (service account) ----------
BASIC_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASIC_PATH)

KEYS_PATH = os.path.join(BASIC_PATH, 'keys')
GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'logger-995ad2d4b91d.json')

GOOGLE_DRIVE_FOLDER_ID = None
GOOGLE_DRIVE_MAKE_PUBLIC = True

ENABLE_URL_SHORTENER = False
SHORTENER_BACKEND = "tinyurl"


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
# GOOGLE DRIVE UPLOADER
# =========================================================
class GoogleDriveUploader:
    def __init__(
        self,
        service_account_json: str,
        folder_id: Optional[str] = None,
        make_public: bool = True,
        enable_shortener: bool = False,
        shortener_backend: str = "tinyurl",
    ):
        self.service_account_json = service_account_json
        self.folder_id = folder_id
        self.make_public = make_public

        self.enable_shortener = enable_shortener
        self.shortener_backend = shortener_backend
        self._shortener = None
        self._drive = None

        self._init_drive()
        self._init_shortener()

    def _init_drive(self):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except Exception as e:
            raise RuntimeError(
                "Librairies Google manquantes. Installe:\n"
                "  pip install google-api-python-client google-auth-httplib2 google-auth\n"
                f"Détail: {e}"
            )

        if not os.path.isfile(self.service_account_json):
            raise FileNotFoundError(f"Service account JSON introuvable: {self.service_account_json}")

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        creds = service_account.Credentials.from_service_account_file(self.service_account_json, scopes=scopes)
        self._drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _init_shortener(self):
        if not self.enable_shortener:
            return
        try:
            import pyshorteners
            self._shortener = pyshorteners.Shortener()
        except Exception:
            self._shortener = None

    def upload_and_get_url(self, filepath: str) -> str:
        from googleapiclient.http import MediaFileUpload

        filename = os.path.basename(filepath)
        metadata = {"name": filename}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]

        media = MediaFileUpload(filepath, mimetype="image/jpeg", resumable=True)
        created = self._drive.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]

        if self.make_public:
            try:
                self._drive.permissions().create(
                    fileId=file_id,
                    body={"type": "anyone", "role": "reader"},
                    fields="id",
                ).execute()
            except Exception:
                pass

        info = self._drive.files().get(fileId=file_id, fields="webViewLink,webContentLink").execute()
        url = info.get("webViewLink") or info.get("webContentLink") or f"https://drive.google.com/file/d/{file_id}/view"

        if self._shortener is not None:
            try:
                short_fn = getattr(self._shortener, self.shortener_backend).short
                url = short_fn(url)
            except Exception:
                pass

        return url


class CaptureStorage:
    def __init__(self, capture_dir: str, uploader: GoogleDriveUploader):
        self.capture_dir = capture_dir
        self.uploader = uploader
        os.makedirs(self.capture_dir, exist_ok=True)

    def save_frame_and_upload(self, frame_bgr) -> tuple[str, str]:
        # microseconds -> évite collisions et garantit une URL/QR par photo
        ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")
        filename = f"capture_{ts}.jpg"
        filepath = os.path.join(self.capture_dir, filename)

        ok = cv2.imwrite(filepath, frame_bgr)
        if not ok:
            raise IOError(f"Impossible d'écrire l'image: {filepath}")

        url = self.uploader.upload_and_get_url(filepath)
        return filepath, url

    def close(self):
        pass


# =========================================================
# URL -> QR code
# =========================================================
def make_qr_image(url: str, size: int = 260) -> Image.Image:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size))


# =========================================================
# ROI mean
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

        # Baseline (pas de texte à l’écran)
        self._baseline_start = time.monotonic()
        self._baseline_samples = []
        self._baseline_mean = None

        # Hold timer
        self._roi_active_since = None

        # Cooldown / séquence
        self._cooldown_until = 0.0
        self._sequence_running = False

        # disable ROI after capture
        self._roi_disabled_until = 0.0

        # Google uploader + storage
        uploader = GoogleDriveUploader(
            service_account_json=GOOGLE_SERVICE_ACCOUNT_JSON,
            folder_id=GOOGLE_DRIVE_FOLDER_ID,
            make_public=GOOGLE_DRIVE_MAKE_PUBLIC,
            enable_shortener=ENABLE_URL_SHORTENER,
            shortener_backend=SHORTENER_BACKEND,
        )
        self.storage = CaptureStorage(CAPTURE_DIR, uploader)

        # Tkinter UI (sans lien, sans phrases ROI)
        self.root = tk.Tk()
        self.root.title("Camera → Drive → QR")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Camera → QR", font=("Arial", 16)).pack(pady=(0, 8))

        self.preview_label = ttk.Label(main)
        self.preview_label.pack(pady=(0, 10))

        # QR strip
        self.qr_canvas_w = QR_HISTORY * QR_SIZE + (QR_HISTORY + 1) * QR_GAP
        self.qr_canvas_h = QR_SIZE + 2 * QR_GAP
        self.qr_canvas = tk.Canvas(main, width=self.qr_canvas_w, height=self.qr_canvas_h, highlightthickness=0)
        self.qr_canvas.pack(pady=(4, 0))

        self._qr_items = []
        self._qr_animating = False

        self._tk_preview_img = None

        # loops
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

            if FLIP_PREVIEW:
                small = apply_flip(small, FLIP_MODE)

            now = time.monotonic()
            roi_mean = get_roi_mean_bgr(small)

            # Baseline silencieuse
            if self._baseline_mean is None:
                elapsed = now - self._baseline_start
                self._baseline_samples.append(roi_mean)
                if elapsed >= BASELINE_SECONDS:
                    arr = np.stack(self._baseline_samples, axis=0)
                    self._baseline_mean = arr.mean(axis=0)
            else:
                roi_enabled = (
                    (now >= self._cooldown_until) and
                    (not self._sequence_running) and
                    (now >= self._roi_disabled_until)
                )

                if not roi_enabled:
                    self._roi_active_since = None
                else:
                    dist = float(np.linalg.norm(roi_mean - self._baseline_mean))
                    if dist >= TRIGGER_DIST_THRESHOLD:
                        if self._roi_active_since is None:
                            self._roi_active_since = now

                        held = now - self._roi_active_since
                        if held >= HOLD_SECONDS:
                            self._roi_active_since = None
                            self._cooldown_until = now + COOLDOWN_SECONDS
                            self.start_countdown_then_capture(COUNTDOWN_SECONDS)
                    else:
                        self._roi_active_since = None

            # ROI rectangle: vert si actif, rouge si désactivé (toujours visible)
            if DRAW_ROI_RECT:
                roi_is_active = (
                    self._baseline_mean is not None
                    and now >= self._roi_disabled_until
                    and not self._sequence_running
                    and now >= self._cooldown_until
                )
                color = (0, 255, 0) if roi_is_active else (0, 0, 255)
                x1 = int(ROI_X)
                y1 = int(ROI_Y)
                x2 = int(ROI_X + ROI_W)
                y2 = int(ROI_Y + ROI_H)
                cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)

            # flash
            if now < self._flash_until:
                small[:] = 0 if self._flash_color.lower() == "black" else 255

            # countdown (uniquement sur l’image)
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
                return

            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)

            frame = apply_frame_and_logo(frame)

            local_path, url = self.storage.save_frame_and_upload(frame)
            if not url:
                return

            qr_img = make_qr_image(url, size=260)

            # Désactive ROI après capture
            self._roi_disabled_until = time.monotonic() + ROI_DISABLE_AFTER_CAPTURE_S

            # Push QR dans l’historique
            self.ui(lambda: self.push_qr_to_history(qr_img))

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
