import datetime
import math
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from constant import *
from log import get_logger
from graphics import apply_flip, apply_frame_and_logo, make_qr_image, ROIManager

# Google upload helpers
try:
    from google_upload import (
        GoogleDriveUploader,
        GoogleSheetsLogger,
        CaptureStorage,
        extract_spreadsheet_id,
        _read_first_nonempty_line,
    )
except Exception:
    GoogleDriveUploader = None
    GoogleSheetsLogger = None
    CaptureStorage = None
    extract_spreadsheet_id = None
    _read_first_nonempty_line = None


log = get_logger()


class CameraGrabber:
    """Continuously read the camera and keep only the latest frame."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open the camera on index {index}.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RESOLUTION[0]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RESOLUTION[1]))

        self.lock = threading.Lock()
        self.last_frame = None
        self.running = True

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        log.info(
            "Camera opened. requested=%sx%s actual=%sx%s fps=%s fourcc=%s",
            CAMERA_RESOLUTION[0],
            CAMERA_RESOLUTION[1],
            actual_w,
            actual_h,
            actual_fps,
            fourcc_str or "?",
        )

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


class FFplayViewer:
    """Display RGB frames through ffplay fullscreen without Tkinter."""

    def __init__(self, width: int, height: int, fps: int = 25):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "ffplay",
            "-loglevel",
            "warning",
            "-fs",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",
            "-sync",
            "ext",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        log.info("ffplay started.")

    def write(self, frame_rgb: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            return
        if self.proc.poll() is not None:
            raise RuntimeError("ffplay exited unexpectedly.")
        try:
            self.proc.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffplay pipe is closed.") from exc

    def close(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class TerminalCameraApp:
    def __init__(self):
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._last_frame_rgb: Optional[np.ndarray] = None

        self._running = True
        self._sequence_running = False
        self._capture_started = False
        self._countdown_end: Optional[float] = None
        self._flash_until: float = 0.0

        self.preview_w = int(PREVIEW_W)
        self.preview_h = int(PREVIEW_H)
        self.qr_size = int(QR_FIXED_SIZE_PX)
        self.qr_gap = int(QR_GAP)
        self.qr_bar_h = self.qr_size + 2 * self.qr_gap
        self.display_w = self.preview_w
        self.display_h = self.preview_h + self.qr_bar_h
        self.preview_fps = 25
        self.frame_interval_s = 1.0 / self.preview_fps

        self.storage = None
        self.sheets_logger = None
        self.roi_manager = ROIManager()
        self.grabber = CameraGrabber(CAM_INDEX)
        self.viewer = FFplayViewer(self.display_w, self.display_h, self.preview_fps)
        self.qr_history_rgb: List[np.ndarray] = []
        self.qr_output_dir = Path(LOG_FOLDER) / "qr_codes"
        self.qr_output_dir.mkdir(parents=True, exist_ok=True)

        self._init_services()

    def _init_services(self):
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

    def _log_sheet_event(self, event: str, local_name: str, url: str, note: str):
        if self.sheets_logger is None:
            return
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.sheets_logger.append_row([ts, event, local_name, url, note])
        except Exception:
            log.exception("Failed to append row to Google Sheets (event=%s).", event)

    def start_countdown_then_capture(self, seconds: int = 3):
        if self._sequence_running:
            return
        if not self._capture_lock.acquire(blocking=False):
            return

        now = time.monotonic()
        self._sequence_running = True
        self._capture_started = False
        self._countdown_end = now + max(1, int(seconds))
        self._flash_until = self._countdown_end + float(FLASH_DURATION_S)
        log.info("Countdown started for %s seconds.", seconds)

    def _maybe_start_capture_after_flash(self, now: float):
        if not self._sequence_running or self._capture_started:
            return
        if self._countdown_end is None:
            return
        if now >= self._flash_until:
            self._capture_started = True
            threading.Thread(target=self.capture_flow, daemon=True).start()

    def _apply_countdown_and_flash(self, frame_rgb: np.ndarray, now: float) -> np.ndarray:
        if not self._sequence_running or self._countdown_end is None:
            return frame_rgb

        if now < self._countdown_end:
            remaining = max(1, int(math.ceil(self._countdown_end - now)))
            text = str(remaining)
            h, w = frame_rgb.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(1.0, min(w, h) / 250.0)
            thickness = max(2, int(scale * 2.5))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int((w - tw) / 2)
            y = int((h + th) / 2)
            cv2.putText(frame_rgb, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
            cv2.putText(frame_rgb, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            return frame_rgb

        if now < self._flash_until:
            if str(FLASH_COLOR).lower() == "black":
                frame_rgb[:] = 0
            else:
                frame_rgb[:] = 255
        return frame_rgb

    def _push_qr_to_history(self, qr_rgb: np.ndarray):
        self.qr_history_rgb.insert(0, qr_rgb)
        self.qr_history_rgb = self.qr_history_rgb[: int(QR_HISTORY)]

    def _build_qr_strip(self) -> np.ndarray:
        bar = np.zeros((self.qr_bar_h, self.display_w, 3), dtype=np.uint8)
        bg = str(QR_BAR_BG).lstrip("#")
        if len(bg) == 6:
            try:
                bar[:] = tuple(int(bg[i : i + 2], 16) for i in (0, 2, 4))
            except Exception:
                pass

        total = int(QR_HISTORY) * self.qr_size + (int(QR_HISTORY) + 1) * self.qr_gap
        align = str(QR_STRIP_ALIGN).lower()
        margin = int(QR_STRIP_MARGIN_PX)
        if align == "left":
            left = max(0, margin)
        elif align == "right":
            left = max(0, self.display_w - total - margin)
        else:
            left = max(0, (self.display_w - total) // 2 + margin)

        y = self.qr_gap
        for i, qr_rgb in enumerate(self.qr_history_rgb[: int(QR_HISTORY)]):
            x = left + self.qr_gap + i * (self.qr_size + self.qr_gap)
            qr_small = cv2.resize(qr_rgb, (self.qr_size, self.qr_size), interpolation=cv2.INTER_AREA)
            bar[y : y + self.qr_size, x : x + self.qr_size] = qr_small

        return bar

    def _compose_display(self, preview_rgb: np.ndarray) -> np.ndarray:
        qr_bar = self._build_qr_strip()
        return np.vstack([preview_rgb, qr_bar])

    def capture_flow(self):
        try:
            with self._frame_lock:
                frame = None if self._last_frame_rgb is None else self._last_frame_rgb.copy()
            if frame is None:
                log.warning("capture_flow: last frame is None.")
                return

            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)

            if CAPTURE_APPLY_OVERLAYS:
                frame = apply_frame_and_logo(frame)

            if self.storage is None:
                log.error("Capture requested but storage is None (Drive not initialized).")
                return

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
            print(f"\nUPLOAD OK: {url}")
            self._log_sheet_event("UPLOAD_OK", local_name, url, "")

            try:
                qr_img = make_qr_image(url)
                qr_rgb = np.array(qr_img.convert("RGB"), dtype=np.uint8)
                self._push_qr_to_history(qr_rgb)

                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                qr_path = self.qr_output_dir / f"qr_{stamp}.png"
                qr_img.save(str(qr_path))
                log.info("QR saved locally: %s", qr_path)
            except Exception:
                log.exception("Failed to generate QR from URL.")
                if bool(globals().get("SHEETS_LOG_QR_EVENTS", False)):
                    self._log_sheet_event("QR_ERROR", local_name, url, "exception")
                return

            if bool(globals().get("SHEETS_LOG_QR_EVENTS", False)):
                self._log_sheet_event("QR_OK", local_name, url, "")

            self.roi_manager.on_capture_done(time.monotonic())

        finally:
            self._sequence_running = False
            self._capture_started = False
            self._countdown_end = None
            self._flash_until = 0.0
            try:
                self._capture_lock.release()
            except Exception:
                pass

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        try:
            self.grabber.release()
        except Exception:
            pass
        try:
            if self.storage:
                self.storage.close()
        except Exception:
            pass
        try:
            self.viewer.close()
        except Exception:
            pass

    def run(self):
        print("App démarrée sans Tkinter.")
        print("q ou Esc dans ffplay pour fermer la fenêtre. Ctrl+C dans le terminal pour arrêter Python.")
        print("Le ROI, le compte à rebours, le flash et la barre QR sont affichés dans ffplay.")

        self.viewer.start()
        next_frame_time = time.monotonic()

        try:
            while self._running:
                frame_bgr = self.grabber.get_latest_frame()
                if frame_bgr is None:
                    time.sleep(0.01)
                    continue

                frame_h, frame_w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._last_frame_rgb = frame_rgb.copy()

                interp = cv2.INTER_AREA if (self.preview_w <= frame_w and self.preview_h <= frame_h) else cv2.INTER_LINEAR
                small = cv2.resize(frame_rgb, (self.preview_w, self.preview_h), interpolation=interp)

                if FLIP_PREVIEW:
                    small = apply_flip(small, FLIP_MODE)

                now = time.monotonic()
                allow_trigger = not self._sequence_running
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

                small = self._apply_countdown_and_flash(small, now)

                if PREVIEW_APPLY_OVERLAYS:
                    small = apply_frame_and_logo(small)

                display_rgb = self._compose_display(small)
                self.viewer.write(display_rgb)
                self._maybe_start_capture_after_flash(now)

                next_frame_time += self.frame_interval_s
                sleep_s = next_frame_time - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_time = time.monotonic()

        finally:
            self.close()


if __name__ == "__main__":
    app = TerminalCameraApp()

    def _handle_signal(signum, _frame):
        log.info("Signal received: %s", signum)
        app.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Starting terminal camera application (ffplay backend, no Tkinter).")
    app.run()
