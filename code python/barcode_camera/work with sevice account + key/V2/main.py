import threading
import datetime
import time
import queue

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from constant import *

from log import get_logger
from graphics import (
    apply_flip,
    apply_frame_and_logo,
    make_qr_image,
    get_roi_mean_bgr,
    roi_cam_to_preview,
    draw_roi_rect,
    CameraCanvasPlacer,
    QRStrip,
    compute_layout,
)

# Google upload helpers
try:
    from google_upload import GoogleDriveUploader, GoogleSheetsLogger, CaptureStorage
except Exception:
    GoogleDriveUploader = None
    GoogleSheetsLogger = None
    CaptureStorage = None

log = get_logger()


# =========================================================
# APP (main = orchestration)
# =========================================================
class CameraAppGUI:
    """
    Main application orchestrating:
      1) camera acquisition
      2) UI + layout
      3) ROI detection -> countdown -> capture
      4) overlays -> upload -> QR creation -> QR history strip
      5) logging (file logs via log.py + optional Google Sheets)
    """

    def __init__(self):
        # -----------------------------
        # Runtime state / thread safety
        # -----------------------------
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._last_frame_bgr = None

        self._running = True
        self._sequence_running = False

        # UI-safe queue (call Tk from the Tk thread only)
        self._uiq = queue.Queue()

        # UX capture
        self._countdown_text = None
        self._flash_until = 0.0
        self._flash_color = FLASH_COLOR

        # Baseline for ROI trigger
        self._baseline_start = time.monotonic()
        self._baseline_samples = []
        self._baseline_mean = None

        # Hold timer for ROI trigger
        self._roi_active_since = None

        # Cooldown / disable ROI after capture
        self._cooldown_until = 0.0
        self._roi_disabled_until = 0.0

        # Layout (dynamic)
        self.preview_w = int(PREVIEW_W)
        self.preview_h = int(PREVIEW_H)
        self.display_w = int(PREVIEW_W)
        self.display_h = int(PREVIEW_H)
        self.qr_size = int(QR_FIXED_SIZE_PX)
        self.qr_canvas_h = self.qr_size + 2 * int(QR_GAP)

        self._layout_pending = False
        self._last_layout_key = None

        # Services
        self.storage = None
        self.sheets_logger = None

        # Graphics helpers
        self.placer = CameraCanvasPlacer()

        # -----------------------------
        # Init camera and services (logical order)
        # -----------------------------
        self._init_camera()
        self._init_services()

        # -----------------------------
        # Tkinter UI
        # -----------------------------
        self.root = tk.Tk()
        self.root.title("Camera → Drive → QR")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Fullscreen by default + Escape toggles
        self._is_fullscreen = True
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.toggle_fullscreen())

        # Main container
        main = ttk.Frame(self.root, padding=0)
        main.pack(fill="both", expand=True)

        # Preview label (top)
        self.preview_label = ttk.Label(main)
        self.preview_label.pack(fill="both", expand=True)

        # QR strip (bottom)
        self.qr_canvas = tk.Canvas(main, highlightthickness=0, bg=QR_BAR_BG)
        self.qr_canvas.pack(side="bottom", fill="x")

        # QR strip controller (all QR functions grouped in graphics.py)
        self.qr_strip = QRStrip(self.root, self.qr_canvas)

        self._tk_preview_img = None

        # Recompute layout when window size changes (debounced)
        self.root.bind("<Configure>", self._on_configure)

        # loops
        self.root.after(10, self.update_preview)
        self.root.after(25, self._process_ui_queue)

        log.info("UI initialisée")

    # =====================================================
    # Init camera
    # =====================================================
    def _init_camera(self):
        backend = None
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        else:
            backend = getattr(cv2, "CAP_V4L2", None)

        if backend is None:
            self.cap = cv2.VideoCapture(CAM_INDEX)
        else:
            self.cap = cv2.VideoCapture(CAM_INDEX, backend)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam. Change CAM_INDEX (0/1/2).")

        desired_w = int(globals().get("FRAME_WIDTH", globals().get("CAMERA_RESOLUTION", (1280, 720))[0]))
        desired_h = int(globals().get("FRAME_HEIGHT", globals().get("CAMERA_RESOLUTION", (1280, 720))[1]))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("CAM requested=%sx%s actual=%sx%s", desired_w, desired_h, actual_w, actual_h)

    # =====================================================
    # Init services (Drive + Sheets)
    # =====================================================
    def _init_services(self):
        """
        Initialise:
          - self.storage (Drive uploader wrapper)
          - self.sheets_logger (optional)
        Uses constants if present. If missing, app runs without upload.
        """
        def pick_const(*names, default=None):
            for n in names:
                if n in globals() and globals()[n] not in (None, ""):
                    return globals()[n]
            return default

        # Drive config
        sa_json = pick_const("GOOGLE_SERVICE_ACCOUNT_JSON", "SERVICE_ACCOUNT_JSON", "SERVICE_ACCOUNT_FILE")
        folder_id = pick_const("DRIVE_FOLDER_ID", "GOOGLE_DRIVE_FOLDER_ID", "FOLDER_ID", default=None)
        make_public = bool(pick_const("DRIVE_MAKE_PUBLIC", "MAKE_PUBLIC", default=True))
        enable_shortener = bool(pick_const("DRIVE_ENABLE_SHORTENER", default=False))
        shortener_backend = pick_const("DRIVE_SHORTENER_BACKEND", default="tinyurl")

        uploader = None

        try:
            if sa_json and GoogleDriveUploader is not None:
                uploader = GoogleDriveUploader(
                    service_account_json=str(sa_json),
                    folder_id=folder_id,
                    make_public=make_public,
                    enable_shortener=enable_shortener,
                    shortener_backend=str(shortener_backend),
                )
                log.info("Drive uploader = Service Account (folder_id=%s)", folder_id)
            else:
                log.warning("Drive uploader NON initialisé (JSON manquant ou module google_upload indisponible).")
        except Exception:
            log.exception("Erreur init uploader Drive")

        if uploader is not None and CaptureStorage is not None:
            try:
                self.storage = CaptureStorage(uploader)
                log.info("CaptureStorage initialisé")
            except Exception:
                self.storage = None
                log.exception("Erreur init CaptureStorage")

        # Sheets logging (optional)
        sheet_id = pick_const("SHEETS_SPREADSHEET_ID", "SPREADSHEET_ID", "GOOGLE_SHEETS_SPREADSHEET_ID")
        sheet_tab = pick_const("SHEETS_WORKSHEET_NAME", default="logs")

        if sa_json and sheet_id and GoogleSheetsLogger is not None:
            try:
                self.sheets_logger = GoogleSheetsLogger(
                    service_account_json=str(sa_json),
                    spreadsheet_id=str(sheet_id),
                    worksheet_name=str(sheet_tab),
                )
                log.info("Sheets logger initialisé (tab=%s)", sheet_tab)
            except Exception:
                self.sheets_logger = None
                log.exception("Erreur init GoogleSheetsLogger")

    # =====================================================
    # Fullscreen toggle
    # =====================================================
    def toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.root.attributes("-fullscreen", self._is_fullscreen)

    # =====================================================
    # UI queue
    # =====================================================
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
                    log.exception("UI callback failed")
        except queue.Empty:
            pass
        self.root.after(25, self._process_ui_queue)

    # =====================================================
    # Layout / resize handling
    # =====================================================
    def _on_configure(self, event):
        if self._layout_pending:
            return
        self._layout_pending = True
        self.root.after(120, self._apply_layout)

    def _apply_layout(self):
        self._layout_pending = False
        if not self._running:
            return

        win_w = int(self.root.winfo_width() or PREVIEW_W)
        win_h = int(self.root.winfo_height() or PREVIEW_H)

        lay = compute_layout(win_w, win_h)
        key = (lay["display_w"], lay["display_h"], lay["qr_size"], lay["qr_canvas_h"])
        if self._last_layout_key == key:
            return
        self._last_layout_key = key

        self.display_w = lay["display_w"]
        self.display_h = lay["display_h"]
        self.preview_w = lay["preview_w"]
        self.preview_h = lay["preview_h"]
        self.qr_size = lay["qr_size"]
        self.qr_canvas_h = lay["qr_canvas_h"]

        self.qr_strip.apply_layout(win_w=lay["display_w"], qr_size=lay["qr_size"], qr_canvas_h=lay["qr_canvas_h"], gap=lay["gap"])

        log.info("Layout: win=%sx%s preview=%sx%s qr=%s", lay["display_w"], win_h, self.preview_w, self.preview_h, self.qr_size)

    # =====================================================
    # Preview loop
    # =====================================================
    def update_preview(self):
        if not self._running:
            return

        ok, frame = self.cap.read()
        if ok and frame is not None:
            frame_h, frame_w = frame.shape[:2]

            with self._frame_lock:
                self._last_frame_bgr = frame

            # Resize to processing size
            interp = cv2.INTER_AREA if (self.preview_w <= frame_w and self.preview_h <= frame_h) else cv2.INTER_LINEAR
            small = cv2.resize(frame, (self.preview_w, self.preview_h), interpolation=interp)

            if FLIP_PREVIEW:
                small = apply_flip(small, FLIP_MODE)

            now = time.monotonic()

            # ROI mapping (camera -> preview)
            roi_x, roi_y, roi_w, roi_h = roi_cam_to_preview(frame_w, frame_h, self.preview_w, self.preview_h)
            roi_mean = get_roi_mean_bgr(small, roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h)

            # Baseline collection
            if self._baseline_mean is None:
                elapsed = now - self._baseline_start
                self._baseline_samples.append(roi_mean)
                if elapsed >= BASELINE_SECONDS and len(self._baseline_samples) > 0:
                    arr = np.stack(self._baseline_samples, axis=0)
                    self._baseline_mean = arr.mean(axis=0)
                    log.info("Baseline ROI prêt (samples=%s)", len(self._baseline_samples))
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

            roi_is_active = (
                self._baseline_mean is not None
                and now >= self._roi_disabled_until
                and not self._sequence_running
                and now >= self._cooldown_until
            )
            small = draw_roi_rect(small, roi_x, roi_y, roi_w, roi_h, active=bool(roi_is_active))

            # flash overlay
            if now < self._flash_until:
                small[:] = 0 if self._flash_color.lower() == "black" else 255

            # countdown overlay
            if self._countdown_text is not None:
                text = str(self._countdown_text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(1.0, min(self.preview_w, self.preview_h) / 250.0)
                thickness = max(2, int(scale * 2.5))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                x = int((self.preview_w - tw) / 2)
                y = int((self.preview_h + th) / 2)
                cv2.putText(small, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
                cv2.putText(small, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Optional live preview overlays
            if PREVIEW_APPLY_OVERLAYS:
                small = apply_frame_and_logo(small)

            # Place in display canvas (letterbox + anchor)
            disp_w = max(1, int(self.display_w))
            disp_h = max(1, int(self.display_h))
            display_bgr, _rect = self.placer.letterbox_place(
                small,
                canvas_w=disp_w,
                canvas_h=disp_h,
                anchor=str(CAMERA_ANCHOR),
                margin_px=int(CAMERA_ANCHOR_MARGIN_PX),
                bg_bgr=tuple(LETTERBOX_BG_BGR),
            )

            # Convert for Tkinter display
            rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self._tk_preview_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self._tk_preview_img)

        self.root.after(33, self.update_preview)

    # =====================================================
    # Countdown + flash + capture
    # =====================================================
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
        """
        Capture last frame, apply overlays (frame+logo) on the saved/uploaded image if enabled,
        upload to Drive, generate QR, update QR strip.
        """
        try:
            with self._frame_lock:
                frame = None if self._last_frame_bgr is None else self._last_frame_bgr.copy()
            if frame is None:
                log.warning("capture_flow: frame is None")
                return

            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)

            # Burn overlays into SAVED/UPLOADED photo
            if CAPTURE_APPLY_OVERLAYS:
                frame = apply_frame_and_logo(frame)

            if self.storage is None:
                log.error("Capture demandée mais storage=None (Drive non initialisé).")
                return

            try:
                local_path, url = self.storage.save_frame_and_upload(frame)
            except Exception:
                log.exception("Upload vers Drive a échoué")
                if self.sheets_logger is not None:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "UPLOAD_ERROR", "", "", "exception"])
                    except Exception:
                        log.exception("Sheets append_row failed (UPLOAD_ERROR)")
                return

            if not url:
                log.error("Upload OK mais URL vide.")
                if self.sheets_logger is not None:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "URL_EMPTY", local_path, "", ""])
                    except Exception:
                        log.exception("Sheets append_row failed (URL_EMPTY)")
                return

            log.info("URL reçue: %s", url)
            if self.sheets_logger is not None:
                try:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.sheets_logger.append_row([ts, "UPLOAD_OK", local_path, url, ""])
                except Exception:
                    log.exception("Sheets append_row failed (UPLOAD_OK)")

            # Generate QR
            try:
                qr_img = make_qr_image(url, size=int(self.qr_size))
            except Exception:
                log.exception("Conversion URL -> QR a échoué")
                if self.sheets_logger is not None and SHEETS_LOG_QR_EVENTS:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "QR_ERROR", local_path, url, "exception"])
                    except Exception:
                        log.exception("Sheets append_row failed (QR_ERROR)")
                return

            if self.sheets_logger is not None and SHEETS_LOG_QR_EVENTS:
                try:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.sheets_logger.append_row([ts, "QR_OK", local_path, url, ""])
                except Exception:
                    log.exception("Sheets append_row failed (QR_OK)")

            # Disable ROI temporarily after capture
            self._roi_disabled_until = time.monotonic() + ROI_DISABLE_AFTER_CAPTURE_S

            # Push QR to history (UI thread)
            self.ui(lambda: self.qr_strip.push(qr_img))

        finally:
            self._sequence_running = False
            try:
                self._capture_lock.release()
            except Exception:
                pass

    # =====================================================
    # Close
    # =====================================================
    def on_close(self):
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            if self.storage:
                self.storage.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        log.info("Entrée dans mainloop (la fenêtre doit rester ouverte).")
        self.root.mainloop()
        log.info("Sortie de mainloop (fenêtre fermée).")


if __name__ == "__main__":
    log.info("Démarrage de l'application Camera → Drive → QR")
    CameraAppGUI().run()
