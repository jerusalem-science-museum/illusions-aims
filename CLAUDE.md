# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A museum exhibit (Ames room / "illusions-aims") that runs on a Raspberry Pi with a USB camera. It shows a live camera feed fullscreen. When a visitor covers a small Region Of Interest (ROI) in the frame, it runs a countdown and flash, then captures a photo. The photo is composited into a branded template, uploaded to Google Drive, and turned into a QR code that links to it. Events can also be logged to a Google Sheet.

## Layout

- `src/`: the active application. All work happens here.
- `legacy/`: an older snapshot of the same app (it uses a different keys subfolder, mockup and QR history size). Treat it as reference only and don't edit it unless asked.
- `pic/`: overlay and template PNGs. `keys/`: Google service account JSON and `sheet_id.txt`. It is gitignored; the real keys live in the museum's OneDrive (link in `README.md`).
- `constant.py` sets `BASIC_PATH` to the **parent of `src/`**, so `keys/` and `pic/` are resolved at the repo root, not inside `src/`. Logs and saved QR PNGs go to `src/logs/` (gitignored).

## Running

There are no tests, linter or build step.

```bash
# Raspberry Pi deployment (from src/): apt packages, AnyDesk, .venv, X11, desktop autologin,
# autostart entry that launches run.sh, disables screen blanking
chmod +x setup.sh && ./setup.sh

# Run (activates src/.venv, then runs main.py)
./run.sh
# or directly, from inside src/ (modules import each other by bare name)
python3 main.py
```

`run_s.sh` is an alternative apt-based dependency installer and check. Its comments are in French.

Runtime requirements: `ffplay` (from ffmpeg) must be on PATH, and the camera must be at `CAM_INDEX`. Python deps are in `requirements.txt` (the root and `src/` copies differ slightly; `setup.sh` installs from the one next to it). The app exits when the ffplay window is closed (q or Esc) or on SIGINT/SIGTERM.

## Architecture

**Display is ffplay, not Tkinter.** `main.py`'s `TerminalCameraApp` renders every frame as a raw RGB numpy array and pipes it to an `ffplay` subprocess's stdin (`FFplayViewer`). Any drawing (ROI box, countdown digits, flash, template, QR) must be baked into the frame array. Tkinter-era code is still present but unused: `QRStrip`, `CountdownController`, `compute_layout` and `CameraCanvasPlacer` in `graphics.py`, and the Tkinter mentions in the root `README.md`.

**Threads:**
- `CameraGrabber` has a background thread that keeps only the latest camera frame.
- The main loop in `run()` is paced to about 25 fps. It resizes and flips the frame, runs `ROIManager.process_frame` (draws the ROI and decides whether to trigger), applies the countdown and flash, applies overlays, composes the frame and writes it to ffplay.
- After the flash, `capture_flow` runs on a separate thread: flip, overlays, save a temp JPEG, Drive upload, optional shortener, generate the QR, append to the QR history, then Sheets logging. Its `finally` block always resets the sequence flags, disables the ROI for `ROI_DISABLE_AFTER_CAPTURE_S` and resets the ROI baseline.

**ROI trigger** (`roi_detector.py`, UI-agnostic):
1. Averages the ROI's mean RGB over `BASELINE_SECONDS`.
2. Triggers when the RGB distance from that baseline stays above `TRIGGER_DIST_THRESHOLD` for `HOLD_SECONDS`.
3. Then applies a cooldown and a disable window.

ROI coordinates are in camera space and are mapped to preview space.

**Template mode** (`USE_CUSTOM_TEMPLATE_MODE = True`, the current default): `apply_custom_template_with_video` pastes a center-cropped camera image into a hardcoded pixel box (`y 90–410, x 85–620`) of `MOCKUP_PNG`, resized to 900×506. The `TEMPLATE_BOXES` and `DYNAMIC_QR_*` constants are not used for this. In this mode the display is exactly `PREVIEW_W × PREVIEW_H` and no QR strip is shown on screen. QR history is collected (`_push_qr_to_history` / `_build_qr_strip`) but not yet composited into the output; this is what the `feat-qr-code-w-graphics` branch is working on.

**Color channels are a recurring trap.** The pipeline mostly carries RGB arrays, even in variables named `*_bgr`. The template is loaded with `cv2.imread` (BGR) and returned as-is, and only the pasted video crop is converted RGB→BGR to match. Check channel order end to end before changing any conversion; the Hebrew comments in `graphics.py` and `main.py` describe earlier tint fixes.

**Non-template branch is broken:** when `USE_CUSTOM_TEMPLATE_MODE = False`, `_compose_display` references `self.qr_lock`, `self.qr_list` and `self.qr_manager`, which are never defined.

**Google services fail soft.** `google_upload` imports are wrapped in try/except. Missing keys or a failed init only log warnings, and captures are then skipped with `storage is None`. The Sheets ID comes from `GOOGLE_SHEETS_SPREADSHEET_ID` or from the first line of `keys/sheet_id.txt` (an ID or full URL).

**Configuration:** all tunables (camera, ROI, timing, overlays, flip, QR, Google, logging) are module-level constants in `src/constant.py`, pulled in with `from constant import *`. `main.py` also reads some of them with `globals().get(...)`.

**Logging:** `log.py` provides a custom date-based rotating handler. Files are named `log_YYYY-MM-DD[(n)][_to_YYYY-MM-DD].txt`, rotate at 1 MB and keep at most `BACKUP_COUNT` files. `google_upload.py` uses `print` instead of the logger.

Comments in the code are a mix of English, Hebrew and French.
