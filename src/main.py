"""
Ames room exhibit: live camera inside a branded template, shown fullscreen through ffplay.
Covering the ROI starts a countdown and flash, then the photo is uploaded to Google Drive
and its QR code appears in the template's QR slots.

Per frame, all work happens at camera resolution until a single resize into the template
window, and the output is one reused buffer written to ffplay without copies.
"""
import datetime
import math
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import constant as cfg
from log import get_logger
from qr_queue import QRQueue
from render import Template, draw_countdown, draw_roi, flash, flip, make_qr
from roi_detector import ROITriggerDetector, roi_mean, roi_rect
from google_upload import DriveUploader, SheetsLogger, resolve_spreadsheet_id

log = get_logger()


class Camera:
    """Reads the camera on a background thread and keeps only the latest frame."""

    def __init__(self, index: int):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open the camera on index {index}.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, cfg.PREVIEW_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_RESOLUTION[1])
        log.info("Camera opened: %dx%d @ %s fps",
                 self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 self.cap.get(cv2.CAP_PROP_FPS))
        if cfg.CAM_LOCK_AUTO:
            self._lock_auto()

        # cap.read() returns a new array each time and nobody mutates it, so no copies are needed.
        self.latest: Optional[np.ndarray] = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _lock_auto(self):
        """Freezes exposure, gain, white balance and focus so scene changes can't trigger the ROI.

        Auto exposure runs briefly first so the frozen value suits the room's lighting.
        Support varies by camera and driver, so every step is best effort.
        """
        cap = self.cap
        end = time.time() + cfg.CAM_LOCK_SETTLE_S
        while time.time() < end:
            cap.read()

        exposure = cfg.CAM_EXPOSURE if cfg.CAM_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = cfg.CAM_GAIN if cfg.CAM_GAIN is not None else cap.get(cv2.CAP_PROP_GAIN)
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)

        # V4L2 uses 1 for manual exposure, DirectShow uses 0.25.
        manual = 1 if hasattr(cv2, "CAP_V4L2") and cap.getBackendName() == "V4L2" else 0.25
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if wb > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        log.info("Camera locked: auto_exposure=%s exposure=%s gain=%s wb=%s",
                 cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_EXPOSURE),
                 cap.get(cv2.CAP_PROP_GAIN), cap.get(cv2.CAP_PROP_WB_TEMPERATURE))

    def _reader(self):
        fails = 0
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                fails = 0
                self.latest = frame
            else:
                fails += 1
                if fails in (1, 30, 120):
                    log.warning("Camera read failed (%d). Retrying...", fails)
                time.sleep(0.01)

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


class FFplay:
    """Fullscreen ffplay fed raw BGR frames on stdin."""

    def __init__(self, w: int, h: int, fps: int):
        self.proc = subprocess.Popen(
            ["ffplay", "-loglevel", "error", "-nostats", "-fs",
             "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop", "-sync", "ext",
             "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{w}x{h}",
             "-framerate", str(fps), "-i", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0,
        )
        log.info("ffplay started.")

    def write(self, frame: np.ndarray) -> bool:
        """Write one frame; False once ffplay is gone (window closed with q/Esc)."""
        if self.proc.poll() is not None:
            return False
        view = memoryview(frame).cast("B")
        try:
            while view:  # an unbuffered pipe write may be partial
                view = view[self.proc.stdin.write(view):]
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class Exhibit:
    """Rendering and the trigger → countdown → flash → capture sequence (no camera or display I/O)."""

    def __init__(self, drive: Optional[DriveUploader], sheets: Optional[SheetsLogger]):
        self.drive = drive
        self.sheets = sheets
        self.template = Template(cfg.MOCKUP_PNG, cfg.PREVIEW_W, cfg.PREVIEW_H)
        self.screen = self.template.bgr.copy()  # output buffer, reused every frame
        self.qr_queue = QRQueue(cfg.PREVIEW_W, cfg.PREVIEW_H)
        self.detector = ROITriggerDetector()
        self.qr_dir = Path(cfg.LOG_FOLDER) / "qr_codes"
        self.qr_dir.mkdir(parents=True, exist_ok=True)

        self.countdown_end: Optional[float] = None  # set while counting down / flashing
        self.capturing = False                      # set while the capture thread runs

    def render(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Advance the sequence and draw one screen frame from a raw camera frame."""
        flash_end = None
        if self.countdown_end is not None:
            flash_end = self.countdown_end + cfg.FLASH_DURATION_S
            if now >= flash_end:
                self.countdown_end = flash_end = None
                self.capturing = True
                threading.Thread(target=self._capture, args=(frame,), daemon=True).start()

        view = flip(frame) if cfg.FLIP_PREVIEW else frame
        rect = roi_rect(view.shape[1], view.shape[0])
        idle = self.countdown_end is None and not self.capturing
        ready, enabled, trigger = self.detector.update(roi_mean(view, rect), now, allow_trigger=idle)
        if trigger:
            log.info("ROI triggered; countdown started.")
            self.countdown_end = now + max(1, int(cfg.COUNTDOWN_SECONDS))

        video, mapping = self.template.fit(view)
        if cfg.DRAW_ROI_RECT:
            draw_roi(video, rect, mapping, active=ready and enabled)
        if self.countdown_end is not None and now < self.countdown_end:
            draw_countdown(video, math.ceil(self.countdown_end - now))
        elif flash_end is not None:
            flash(video)
        self.template.blend(self.screen, video)

        # Restore the template under the QR area, then draw the queue (it may have moved/cleared)
        x0, y0, x1, y1 = self.qr_queue.bbox
        self.screen[y0:y1, x0:x1] = self.template.bgr[y0:y1, x0:x1]
        self.qr_queue.render(self.screen)
        return self.screen

    def _sheet(self, event: str, name: str = "", url: str = "", note: str = ""):
        if self.sheets is not None:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.sheets.append([ts, event, name, url, note])

    def _capture(self, frame: np.ndarray):
        try:
            if self.drive is None:
                log.error("Capture requested but Google Drive is not available.")
                return
            if cfg.FLIP_CAPTURE:
                frame = flip(frame)
            ok, jpg = cv2.imencode(".jpg", self.template.photo(frame))
            if not ok:
                log.error("JPEG encoding failed.")
                return
            name = datetime.datetime.now().strftime("capture_%Y_%m_%d__%H_%M_%S__%f.jpg")
            try:
                url = self.drive.upload_jpeg(jpg.tobytes(), name)
            except Exception:
                log.exception("Drive upload failed.")
                self._sheet("UPLOAD_ERROR", note="exception")
                return
            log.info("UPLOAD OK: %s", url)
            self._sheet("UPLOAD_OK", name, url)

            try:
                qr = make_qr(url)
                self.qr_queue.push(qr)
                cv2.imwrite(str(self.qr_dir / datetime.datetime.now().strftime("qr_%Y%m%d_%H%M%S.png")), qr)
            except Exception:
                log.exception("Failed to generate QR from URL.")
                if cfg.SHEETS_LOG_QR_EVENTS:
                    self._sheet("QR_ERROR", name, url, "exception")
                return
            if cfg.SHEETS_LOG_QR_EVENTS:
                self._sheet("QR_OK", name, url)
        finally:
            self.detector.disable_for(cfg.ROI_DISABLE_AFTER_CAPTURE_S, time.monotonic())
            self.capturing = False


def init_google():
    """Drive uploader and Sheets logger; either is None if unavailable (the exhibit still runs)."""
    drive = sheets = None
    try:
        drive = DriveUploader(
            cfg.GOOGLE_SERVICE_ACCOUNT_JSON,
            folder_id=cfg.GOOGLE_DRIVE_FOLDER_ID,
            make_public=cfg.GOOGLE_DRIVE_MAKE_PUBLIC,
            shortener_backend=cfg.SHORTENER_BACKEND if cfg.ENABLE_URL_SHORTENER else None,
        )
        log.info("Google Drive ready.")
    except Exception:
        log.exception("Google Drive not available; captures will not be uploaded.")

    if cfg.ENABLE_SHEETS_LOG:
        sheet_id = resolve_spreadsheet_id(cfg.GOOGLE_SHEETS_SPREADSHEET_ID, cfg.GOOGLE_SHEETS_SPREADSHEET_ID_FILE)
        if sheet_id:
            try:
                sheets = SheetsLogger(cfg.GOOGLE_SERVICE_ACCOUNT_JSON, sheet_id, cfg.GOOGLE_SHEETS_WORKSHEET_NAME)
                log.info("Google Sheets logging ready (tab=%s).", cfg.GOOGLE_SHEETS_WORKSHEET_NAME)
            except Exception:
                log.exception("Google Sheets logging not available.")
    return drive, sheets


def main():
    log.info("Starting. Press q or Esc in the ffplay window to quit, or Ctrl+C here.")
    exhibit = Exhibit(*init_google())
    camera = Camera(cfg.CAM_INDEX)
    viewer = FFplay(cfg.PREVIEW_W, cfg.PREVIEW_H, cfg.PREVIEW_FPS)

    running = True

    def stop(signum, _frame):
        nonlocal running
        log.info("Signal received: %s", signum)
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    interval = 1.0 / cfg.PREVIEW_FPS
    next_t = time.monotonic()
    try:
        while running:
            frame = camera.latest
            if frame is None:
                time.sleep(0.01)
                continue
            if not viewer.write(exhibit.render(frame, time.monotonic())):
                log.info("ffplay closed. Exiting.")
                break
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind: don't try to catch up
    finally:
        camera.release()
        viewer.close()


if __name__ == "__main__":
    main()
