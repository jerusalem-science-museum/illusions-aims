import threading
import datetime
import time
import queue
from typing import Optional

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
    CameraCanvasPlacer,
    QRStrip,
    compute_layout,
    ROIManager,
    CountdownController,
)

# Google upload helpers
try:
    from google_upload import GoogleDriveUploader, GoogleSheetsLogger, CaptureStorage, extract_spreadsheet_id, _read_first_nonempty_line
except Exception:
    GoogleDriveUploader = None
    GoogleSheetsLogger = None
    CaptureStorage = None
    extract_spreadsheet_id = None
    _read_first_nonempty_line = None





class CameraAppGUI:
    """
    Main application (orchestration only).

    - Camera acquisition
    - UI layout (preview + QR strip)
    - ROI detection (delegated to graphics/roi_detector)
    - Countdown + flash (delegated to graphics)
    - Capture flow: save/upload + QR generation + QR strip push
    - Logging: file logs via log.py, optional Google Sheets
    """

    def __init__(self):
        # -----------------------------
        # Runtime state / thread safety
        # -----------------------------
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._last_frame_rgb: Optional[np.ndarray] = None

        self._running = True
        self._sequence_running = False

        # UI-safe queue (call Tk from the Tk thread only)
        self._uiq = queue.Queue()

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
        self.roi_manager = ROIManager()
        self.countdown = CountdownController()

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

        self.qr_strip = QRStrip(self.root, self.qr_canvas)

        self._tk_preview_img = None

        # Recompute layout when window size changes (debounced)
        self.root.bind("<Configure>", self._on_configure)

        # Loops
        self.root.after(10, self.update_preview)
        self.root.after(25, self._process_ui_queue)

        log.info("UI initialized.")

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
            raise RuntimeError("Cannot open the camera. Try changing CAM_INDEX (0/1/2).")

        desired_w = int(globals().get("FRAME_WIDTH", globals().get("CAMERA_RESOLUTION", (1280, 720))[0]))
        desired_h = int(globals().get("FRAME_HEIGHT", globals().get("CAMERA_RESOLUTION", (1280, 720))[1]))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_h)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera resolution requested=%sx%s actual=%sx%s", desired_w, desired_h, actual_w, actual_h)

    # =====================================================
    # Init services (Drive + Sheets)
    # =====================================================
    def _init_services(self):
        """
        Initialize:
          - self.storage (Drive uploader wrapper)
          - self.sheets_logger (optional)

        Uses constants from constant.py.
        If missing, the app still runs (but without upload / sheets logs).
        """
        sa_json = globals().get("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        folder_id = globals().get("GOOGLE_DRIVE_FOLDER_ID", None)
        make_public = bool(globals().get("GOOGLE_DRIVE_MAKE_PUBLIC", True))

        enable_shortener = bool(globals().get("ENABLE_URL_SHORTENER", False))
        shortener_backend = str(globals().get("SHORTENER_BACKEND", "tinyurl"))

        uploader = None

        try:
            if sa_json and GoogleDriveUploader is not None:
                uploader = GoogleDriveUploader(
                    service_account_json=str(sa_json),
                    folder_id=folder_id,
                    make_public=make_public,
                    enable_shortener=enable_shortener,
                    shortener_backend=shortener_backend,
                )
                log.info("Drive uploader initialized (folder_id=%s).", folder_id)
            else:
                log.warning("Drive uploader NOT initialized (missing JSON or google_upload unavailable).")
        except Exception:
            log.exception("Drive uploader initialization failed.")

        if uploader is not None and CaptureStorage is not None:
            try:
                self.storage = CaptureStorage(uploader)
                log.info("CaptureStorage initialized.")
            except Exception:
                self.storage = None
                log.exception("CaptureStorage initialization failed.")

        # Sheets logging (optional)
        enable_sheets = bool(globals().get("ENABLE_SHEETS_LOG", False))
        sheet_id = globals().get("GOOGLE_SHEETS_SPREADSHEET_ID", None)
        sheet_id_file = globals().get("GOOGLE_SHEETS_SPREADSHEET_ID_FILE", None)
        sheet_tab = str(globals().get("GOOGLE_SHEETS_WORKSHEET_NAME", "logs"))

        if enable_sheets and not sheet_id and sheet_id_file and _read_first_nonempty_line is not None:
            raw = _read_first_nonempty_line(str(sheet_id_file))
            if raw and extract_spreadsheet_id is not None:
                sheet_id = extract_spreadsheet_id(raw)

        if enable_sheets and sa_json and sheet_id and GoogleSheetsLogger is not None:
            try:
                self.sheets_logger = GoogleSheetsLogger(
                    service_account_json=str(sa_json),
                    spreadsheet_id=str(sheet_id),
                    worksheet_name=str(sheet_tab),
                )
                log.info("Sheets logger initialized (tab=%s).", sheet_tab)
            except Exception:
                self.sheets_logger = None
                log.exception("GoogleSheetsLogger initialization failed.")

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
                    log.exception("UI callback failed.")
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

        self.qr_strip.apply_layout(
            win_w=lay["display_w"],
            qr_size=lay["qr_size"],
            qr_canvas_h=lay["qr_canvas_h"],
            gap=lay["gap"],
        )

        log.info(
            "Layout updated: window=%sx%s preview=%sx%s qr_size=%s",
            lay["display_w"], win_h, self.preview_w, self.preview_h, self.qr_size
        )

    # =====================================================
    # Preview loop
    # =====================================================
    def update_preview(self):
        if not self._running:
            return

        ok, frame_bgr = self.cap.read()
        if ok and frame_bgr is not None:
            frame_h, frame_w = frame_bgr.shape[:2]

            # Work in RGB everywhere (OpenCV delivers BGR)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with self._frame_lock:
                self._last_frame_rgb = frame

            # Resize to processing size
            interp = cv2.INTER_AREA if (self.preview_w <= frame_w and self.preview_h <= frame_h) else cv2.INTER_LINEAR
            small = cv2.resize(frame, (self.preview_w, self.preview_h), interpolation=interp)

            if FLIP_PREVIEW:
                small = apply_flip(small, FLIP_MODE)

            now = time.monotonic()

            # ROI + trigger
            allow_trigger = (not self._sequence_running)
            small, should_trigger = self.roi_manager.process_frame(
                frame_w=frame_w,
                frame_h=frame_h,
                preview_w=self.preview_w,
                preview_h=self.preview_h,
                preview_bgr=small,
                now=now,
                allow_trigger=allow_trigger,
            )

            if should_trigger:
                self.start_countdown_then_capture(int(COUNTDOWN_SECONDS))

            # Countdown + flash overlays
            small = self.countdown.apply(small, now)

            # Optional live preview overlays
            if PREVIEW_APPLY_OVERLAYS:
                small = apply_frame_and_logo(small)

            # Place into display canvas (letterbox + anchor)
            disp_w = max(1, int(self.display_w))
            disp_h = max(1, int(self.display_h))
            display_bgr, _rect = self.placer.letterbox_place(
                small,
                canvas_w=disp_w,
                canvas_h=disp_h,
                anchor=str(CAMERA_ANCHOR),
                margin_px=int(CAMERA_ANCHOR_MARGIN_PX),
                bg_rgb=tuple(reversed(LETTERBOX_BG_BGR)),
            )

            # Convert for Tkinter display
            pil = Image.fromarray(display_bgr)
            self._tk_preview_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self._tk_preview_img)

        self.root.after(33, self.update_preview)

    # =====================================================
    # Countdown + flash + capture (countdown moved to graphics.py)
    # =====================================================
    def start_countdown_then_capture(self, seconds: int = 3):
        """Start countdown, flash, then capture in a worker thread."""
        if self._sequence_running:
            return
        if not self._capture_lock.acquire(blocking=False):
            return

        self._sequence_running = True

        def start_capture_thread():
            threading.Thread(target=self.capture_flow, daemon=True).start()

        # Countdown scheduling is owned by graphics.CountdownController
        self.countdown.start(
            root=self.root,
            seconds=int(seconds),
            flash_duration_s=float(FLASH_DURATION_S),
            flash_color=str(FLASH_COLOR),
            on_after_flash=start_capture_thread,
        )

    def capture_flow(self):
        """
        Capture the last camera frame, burn overlays into the saved/uploaded image,
        upload to Drive, generate QR, update QR strip.
        """
        try:
            with self._frame_lock:
                frame = None if self._last_frame_rgb is None else self._last_frame_rgb.copy()
            if frame is None:
                log.warning("capture_flow: last frame is None.")
                return

            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)

            # Burn overlays into SAVED/UPLOADED photo
            if CAPTURE_APPLY_OVERLAYS:
                frame = apply_frame_and_logo(frame)

            if self.storage is None:
                log.error("Capture requested but storage is None (Drive not initialized).")
                return

            # CaptureStorage expects BGR for cv2.imwrite; convert back from RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                local_name, url = self.storage.save_frame_and_upload(frame_bgr)
            except Exception:
                log.exception("Drive upload failed.")
                self._log_sheet_event("UPLOAD_ERROR", "", "", "exception")
                return

            if not url:
                log.error("Upload succeeded but returned an empty URL.")
                self._log_sheet_event("URL_EMPTY", local_name, "", "")
                return

            log.info("Drive URL received: %s", url)
            self._log_sheet_event("UPLOAD_OK", local_name, url, "")

            # Generate QR (resizing is handled by QRStrip)
            try:
                qr_img = make_qr_image(url)
            except Exception:
                log.exception("Failed to generate QR from URL.")
                if bool(globals().get("SHEETS_LOG_QR_EVENTS", False)):
                    self._log_sheet_event("QR_ERROR", local_name, url, "exception")
                return

            if bool(globals().get("SHEETS_LOG_QR_EVENTS", False)):
                self._log_sheet_event("QR_OK", local_name, url, "")

            # Disable ROI temporarily after capture (handled inside graphics ROIManager)
            self.roi_manager.on_capture_done(time.monotonic())

            # Push QR to history (UI thread)
            self.ui(lambda: self.qr_strip.push(qr_img))

        finally:
            self._sequence_running = False
            try:
                self._capture_lock.release()
            except Exception:
                pass

    def _log_sheet_event(self, event: str, local_name: str, url: str, note: str):
        if self.sheets_logger is None:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.sheets_logger.append_row([ts, event, local_name, url, note])
        except Exception:
            log.exception("Failed to append row to Google Sheets (event=%s).", event)

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
        log.info("Entering Tk mainloop (window stays open).")
        self.root.mainloop()
        log.info("Exited Tk mainloop (window closed).")


if __name__ == "__main__":
    log = get_logger()
    log.info("Starting Camera → Drive → QR application.")
    CameraAppGUI().run()
