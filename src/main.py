"""
Ames room exhibit: live camera inside a branded template, shown fullscreen on the GPU.
Covering the ROI starts a countdown and flash, then the photo is uploaded to Google Drive
and its QR code appears in the template's QR slots.

Per frame, the CPU only works at camera resolution (YUV conversion, ROI, overlays). The GPU scales
the camera image into the template's window and composites the template and QRs (see display.py).
"""
import argparse
import datetime
import math
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import constant as cfg
from log import get_logger
from camera import calibrate, open_camera
from display import Display
from qr_queue import QRQueue
from render import Template, draw_countdown, draw_roi, flash, make_qr
from roi_debug import ROIDebug
from roi_detector import ROITriggerDetector, roi_patch, roi_rect
from google_upload import DriveUploader, SheetsLogger, resolve_spreadsheet_id

log = get_logger()


class Exhibit:
    """Rendering and the trigger → countdown → flash → capture sequence (no camera or display I/O)."""

    def __init__(self, drive: Optional[DriveUploader], sheets: Optional[SheetsLogger],
                 offline: bool = False, debug: bool = False):
        self.camera = None  # set once opened; _capture takes the photo from it
        self.drive = drive
        self.sheets = sheets
        self.offline = offline  # save captures locally instead of uploading
        self.template = Template(cfg.MOCKUP_PNG, cfg.PREVIEW_W, cfg.PREVIEW_H)
        # The uploaded photo has its own resolution, so a smaller preview doesn't shrink it
        self.photo_template = (self.template if (cfg.PHOTO_W, cfg.PHOTO_H) == (cfg.PREVIEW_W, cfg.PREVIEW_H)
                               else Template(cfg.MOCKUP_PNG, cfg.PHOTO_W, cfg.PHOTO_H))
        self.qr_queue = QRQueue(cfg.PREVIEW_W, cfg.PREVIEW_H)
        self.detector = ROITriggerDetector()
        self.debug = ROIDebug(self.detector) if debug else None
        self.qr_dir = Path(cfg.LOG_FOLDER) / "qr_codes"
        self.qr_dir.mkdir(parents=True, exist_ok=True)

        self.stage_s = [0.0] * 2  # time spent in detect / draw, summed since the last FPS log
        self.countdown_end: Optional[float] = None  # set while counting down / flashing
        self.capturing = False                      # set while the capture thread runs

    def render(self, frame: np.ndarray, now: float):
        """
        Advance the sequence and prepare one screen frame.
        frame: the camera's BGRA image (camera.bgra(): already mirrored, and a new array, since the
        overlays are drawn on it).
        Returns (video, crop, panel): frame with the overlays drawn on it, the (x, y, w, h) part of
        it that fills the template window, and the --debug panel (or None).
        """
        flash_end = None
        if self.countdown_end is not None:
            flash_end = self.countdown_end + cfg.FLASH_DURATION_S
            if now >= flash_end:
                self.countdown_end = flash_end = None
                self.capturing = True
                threading.Thread(target=self._capture, daemon=True).start()

        t0 = time.perf_counter()
        rect = roi_rect(frame.shape[1], frame.shape[0])
        idle = self.countdown_end is None and not self.capturing
        patch = np.ascontiguousarray(roi_patch(frame, rect)[:, :, :3])  # BGR, as the detector expects
        ready, enabled, trigger = self.detector.update(patch, now, allow_trigger=idle)
        if self.debug:
            state = ("baseline" if not ready else "TRIGGER" if trigger else "armed" if enabled
                     else "busy" if not idle else "cooldown")
            self.debug.update(patch, now, state)
        if trigger:
            log.info("ROI triggered; countdown started.")
            if self.debug:
                self.debug.on_trigger()
            self.countdown_end = now + max(1, int(cfg.COUNTDOWN_SECONDS))

        t1 = time.perf_counter()
        img = frame
        crop = self.template.crop(img.shape[1], img.shape[0])
        cx, cy, cw, ch = crop
        video = img[cy:cy + ch, cx:cx + cw]  # a view: drawing on it draws on img
        if cfg.DRAW_ROI_RECT:
            draw_roi(video, rect, (cx, cy, 1.0), active=ready and enabled)
        if self.countdown_end is not None and now < self.countdown_end:
            draw_countdown(video, math.ceil(self.countdown_end - now))
        elif flash_end is not None:
            flash(video)
        panel = None
        if self.debug:
            panel = np.zeros((290, 440, 3), np.uint8)
            self.debug.draw(panel)
            panel = cv2.cvtColor(panel, cv2.COLOR_BGR2BGRA)
        t2 = time.perf_counter()
        self.stage_s[0] += t1 - t0
        self.stage_s[1] += t2 - t1
        return img, crop, panel

    def _sheet(self, event: str, name: str = "", url: str = "", note: str = ""):
        if self.sheets is not None:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.sheets.append([ts, event, name, url, note])

    def _capture(self):
        try:
            if self.drive is None and not self.offline:
                log.error("Capture requested but Google Drive is not available.")
                return
            frame = self.camera.photo() if self.camera is not None else None
            if frame is None:
                log.error("Capture requested but the camera gave no photo frame.")
                return
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
    """Logs the output and camera frame rates, the average render and display times, the camera's
    exposure and gains (to confirm they stay fixed) and the CPU temperature."""

    def __init__(self, period: float):
        self.period = period
        self._reset(time.monotonic(), 0)

    def _reset(self, now: float, cam_frames: int):
        self.start, self.cam0 = now, cam_frames
        self.n, self.convert_s, self.render_s, self.write_s = 0, 0.0, 0.0, 0.0

    def tick(self, convert_s: float, render_s: float, write_s: float, cam_frames: int, cam_status: str = "",
             stage_s=(0, 0), display_s=(0, 0, 0)) -> None:
        self.n += 1
        self.convert_s += convert_s
        self.render_s += render_s
        self.write_s += write_s
        now = time.monotonic()
        if self.period <= 0 or now - self.start < self.period:
            return
        span = now - self.start
        log.info("FPS out %.1f (target %d), camera %.1f | convert %.1f ms, render %.1f ms (detect %.1f, draw %.1f), display %.1f ms (upload %.1f, draw %.1f, present %.1f)%s%s",
                 self.n / span, cfg.PREVIEW_FPS, (cam_frames - self.cam0) / span, 1000 * self.convert_s / self.n,
                 1000 * self.render_s / self.n, *(1000 * s / self.n for s in stage_s),
                 1000 * self.write_s / self.n, *(1000 * s / self.n for s in display_s), cam_status, _cpu_temp())
        stage_s[:] = [0.0] * len(stage_s)
        display_s[:] = [0.0] * len(display_s)
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
    p.add_argument("--windowed", action="store_true", help="show in a window instead of fullscreen")
    p.add_argument("--camera", choices=("picam", "usb"), default=cfg.CAM_BACKEND,
                   help=f"Raspberry Pi camera or USB webcam (default {cfg.CAM_BACKEND})")
    p.add_argument("--cam", type=int, default=cfg.CAM_INDEX, help=f"USB camera index (default {cfg.CAM_INDEX})")
    p.add_argument("--calibrate-camera", action="store_true",
                   help="Pi camera: find the fixed exposure, gain and white balance for constant.py, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    # OpenCV splits some calls across its thread pool and waits for every worker. With other threads
    # keeping the Pi's cores busy, a worker is often descheduled and the whole call stalls.
    cv2.setNumThreads(1)
    if args.calibrate_camera:
        bw, bh = Template(cfg.MOCKUP_PNG, cfg.PREVIEW_W, cfg.PREVIEW_H).win_size
        calibrate(bw / bh)
        return
    log.info("Starting%s%s. Press q or Esc to quit, or Ctrl+C here.",
             " OFFLINE" if args.offline else "", " with ROI debug" if args.debug else "")
    google = (None, None) if args.offline else init_google()
    exhibit = Exhibit(*google, offline=args.offline, debug=args.debug)
    bw, bh = exhibit.template.win_size
    camera = exhibit.camera = open_camera(args.camera, args.cam, bw / bh)
    viewer = Display(exhibit.template, fullscreen=not args.windowed)

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
            raw = camera.latest
            if raw is None:
                time.sleep(0.01)
                continue
            t0 = time.monotonic()
            frame = camera.bgra(raw)
            t1 = time.monotonic()
            video, crop, panel = exhibit.render(frame, t1)
            t2 = time.monotonic()
            if not viewer.show(video, crop, exhibit.qr_queue.placements(), panel):
                log.info("Window closed. Exiting.")
                break
            fps.tick(t1 - t0, t2 - t1, time.monotonic() - t2, camera.frames, camera.status(),
                     exhibit.stage_s, viewer.stage_s)
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
