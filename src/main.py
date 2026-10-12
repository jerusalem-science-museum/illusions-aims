"""
Ames room exhibit: live camera inside a branded template, shown fullscreen through ffplay.
Covering the ROI starts a countdown and flash, then the photo is uploaded to Google Drive
and its QR code appears in the template's QR slots.

Per frame, all work happens at camera resolution until a single resize into the template
window, and the output is one reused buffer written to ffplay without copies.
"""
import argparse
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
from roi_debug import ROIDebug
from roi_detector import ROITriggerDetector, roi_mean, roi_patch, roi_rect
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
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        log.info("Camera opened: %dx%d @ %s fps, format %s",
                 self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 self.cap.get(cv2.CAP_PROP_FPS), fourcc.to_bytes(4, "little").decode("ascii", "replace"))
        if cfg.CAM_LOCK_AUTO:
            self._lock_auto()

        # cap.read() returns a new array each time and nobody mutates it, so no copies are needed.
        self.latest: Optional[np.ndarray] = None
        self.frames = 0  # frames read so far, for the FPS log
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
                self.frames += 1
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

    def __init__(self, w: int, h: int, fps: int, fullscreen: bool = True):
        self.proc = subprocess.Popen(
            ["ffplay", "-loglevel", "error", "-nostats", *(["-fs"] if fullscreen else []),
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

    def __init__(self, drive: Optional[DriveUploader], sheets: Optional[SheetsLogger],
                 offline: bool = False, debug: bool = False):
        self.drive = drive
        self.sheets = sheets
        self.offline = offline  # save captures locally instead of uploading
        self.template = Template(cfg.MOCKUP_PNG, cfg.PREVIEW_W, cfg.PREVIEW_H)
        # The uploaded photo has its own resolution, so a smaller preview doesn't shrink it
        self.photo_template = (self.template if (cfg.PHOTO_W, cfg.PHOTO_H) == (cfg.PREVIEW_W, cfg.PREVIEW_H)
                               else Template(cfg.MOCKUP_PNG, cfg.PHOTO_W, cfg.PHOTO_H))
        self.screen = self.template.bgr.copy()  # output buffer, reused every frame
        self.qr_queue = QRQueue(cfg.PREVIEW_W, cfg.PREVIEW_H)
        self.detector = ROITriggerDetector()
        self.debug = ROIDebug(self.detector) if debug else None
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
        patch = roi_patch(view, rect)
        ready, enabled, trigger = self.detector.update(
            roi_mean(view, rect), now, allow_trigger=idle, patch=patch if self.debug else None)
        if self.debug:
            state = ("baseline" if not ready else "TRIGGER" if trigger else "armed" if enabled
                     else "busy" if not idle else "cooldown")
            self.debug.update(patch, now, state)
        if trigger:
            log.info("ROI triggered; countdown started.")
            if self.debug:
                self.debug.on_trigger()
            self.countdown_end = now + max(1, int(cfg.COUNTDOWN_SECONDS))

        video, mapping = self.template.fit(view)
        if cfg.DRAW_ROI_RECT:
            draw_roi(video, rect, mapping, active=ready and enabled)
        if self.countdown_end is not None and now < self.countdown_end:
            draw_countdown(video, math.ceil(self.countdown_end - now))
        elif flash_end is not None:
            flash(video)
        if self.debug:
            self.debug.draw(video)
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
            if self.drive is None and not self.offline:
                log.error("Capture requested but Google Drive is not available.")
                return
            if cfg.FLIP_CAPTURE:
                frame = flip(frame)
            ok, jpg = cv2.imencode(".jpg", self.photo_template.photo(frame))
            if not ok:
                log.error("JPEG encoding failed.")
                return
            name = datetime.datetime.now().strftime("capture_%Y_%m_%d__%H_%M_%S__%f.jpg")
            if self.offline:
                path = Path(cfg.OFFLINE_CAPTURE_DIR) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpg.tobytes())
                url = path.as_uri()
                log.info("SAVED (offline): %s", path)
            else:
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


class FPSLog:
    """Logs the output and camera frame rates, the average render and ffplay write times, and the CPU temperature."""

    def __init__(self, period: float):
        self.period = period
        self._reset(time.monotonic(), 0)

    def _reset(self, now: float, cam_frames: int):
        self.start, self.cam0 = now, cam_frames
        self.n, self.render_s, self.write_s = 0, 0.0, 0.0

    def tick(self, render_s: float, write_s: float, cam_frames: int) -> None:
        self.n += 1
        self.render_s += render_s
        self.write_s += write_s
        now = time.monotonic()
        if self.period <= 0 or now - self.start < self.period:
            return
        span = now - self.start
        log.info("FPS out %.1f (target %d), camera %.1f | render %.1f ms, write %.1f ms%s",
                 self.n / span, cfg.PREVIEW_FPS, (cam_frames - self.cam0) / span,
                 1000 * self.render_s / self.n, 1000 * self.write_s / self.n, _cpu_temp())
        self._reset(now, cam_frames)


def _cpu_temp() -> str:
    try:  # Linux / Raspberry Pi
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return f" | CPU {int(f.read()) / 1000:.0f} C"
    except (OSError, ValueError):
        return ""


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


def parse_args():
    p = argparse.ArgumentParser(description="Ames room exhibit")
    p.add_argument("--offline", action="store_true",
                   help="don't use Google: save captures to logs/captures/ and make the QR from the file path")
    p.add_argument("--debug", action="store_true",
                   help="ROI stats panel, logs/roi_stats_*.csv and logs/roi_triggers/*.png")
    p.add_argument("--windowed", action="store_true", help="don't run ffplay fullscreen")
    p.add_argument("--cam", type=int, default=cfg.CAM_INDEX, help=f"camera index (default {cfg.CAM_INDEX})")
    return p.parse_args()


def main():
    args = parse_args()
    log.info("Starting%s%s. Press q or Esc in the ffplay window to quit, or Ctrl+C here.",
             " OFFLINE" if args.offline else "", " with ROI debug" if args.debug else "")
    google = (None, None) if args.offline else init_google()
    exhibit = Exhibit(*google, offline=args.offline, debug=args.debug)
    camera = Camera(args.cam)
    viewer = FFplay(cfg.PREVIEW_W, cfg.PREVIEW_H, cfg.PREVIEW_FPS, fullscreen=not args.windowed)

    running = True

    def stop(signum, _frame):
        nonlocal running
        log.info("Signal received: %s", signum)
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    interval = 1.0 / cfg.PREVIEW_FPS
    next_t = time.monotonic()
    fps = FPSLog(cfg.FPS_LOG_SECONDS)
    try:
        while running:
            frame = camera.latest
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.monotonic()
            out = exhibit.render(frame, t0)
            t1 = time.monotonic()
            if not viewer.write(out):
                log.info("ffplay closed. Exiting.")
                break
            fps.tick(t1 - t0, time.monotonic() - t1, camera.frames)
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind: don't try to catch up
    finally:
        camera.release()
        viewer.close()
        if exhibit.debug:
            exhibit.debug.close()


if __name__ == "__main__":
    main()
