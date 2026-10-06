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

To check rendering speed or output without the camera or ffplay, call `Exhibit(None, None).render(frame, t)` with a synthetic 640×480 BGR frame.

## Architecture

**Display is ffplay.** `main.py` renders every frame into one reused 1920×1080 buffer and writes it, without copying, to an `ffplay` subprocess's stdin (`-pixel_format bgr24`). Everything on screen (ROI box, countdown, flash, template, QRs) is drawn into that buffer.

**Images are BGR everywhere** (OpenCV's native order): camera frames, the template, the ffplay input and the saved JPEG. There are no color conversions anywhere, and it should stay that way. Draw colors are `(B, G, R)`.

**Modules:**
- `main.py`:
  - `Camera` keeps only the latest frame, using a background thread.
  - `FFplay` is the display.
  - `Exhibit` holds the rendering and the trigger → countdown → flash → capture sequence. It does no I/O.
  - `main()` is the loop, paced to `PREVIEW_FPS`.
- `render.py`: `Template` loads `MOCKUP_PNG` once, premultiplied, at the screen size. The picture window is the bounding box of its transparent area.
  - `fit()` center-crops the camera frame to the window's aspect ratio without stretching, then resizes it once.
  - `blend()` puts the result under the template.
  - `photo()` renders the same thing and crops it to the gold frame, so the live preview matches the uploaded photo.
  - The file also has the countdown, flash, ROI box and QR helpers.
- `roi_detector.py`: computes the ROI rectangle and mean color (in camera pixels, on the mirrored image) and contains `ROITriggerDetector`.
  - Baseline: averaged over `BASELINE_SECONDS`.
  - Trigger: the RGB distance stays above `max(TRIGGER_DIST_THRESHOLD, NOISE_SIGMA_MULT × noise)` for `HOLD_SECONDS`.
  - Also handles the adaptive baseline, the grace period, the cooldown and the disable window after a capture.
- `qr_queue.py`: the newest QR goes in the big slot and older ones slide through the 3 archive cells (`QR_SLOT_*`). The queue clears after `QR_QUEUE_RESET_S`. Before the queue draws each frame, the screen restores the template under `QRQueue.bbox`.
- `google_upload.py`:
  - `DriveUploader` uploads JPEG bytes from memory over one shared, locked httplib2 connection, with a 30 s timeout and a 45 s keepalive. The "anyone with the link" permission is set asynchronously.
  - `SheetsLogger` appends rows from its own worker thread.

**Per-frame work stays at camera resolution** (flip, ROI, crop) until the single resize into the window. Only the window and the QR area of the output buffer are rewritten each frame. Keep it that way: per-frame numpy work at full HD is what made the Pi slow.

**Capture:** when the flash ends, `_capture` runs on a thread with that frame. It flips the frame, renders `Template.photo`, encodes the JPEG and uploads it, pushes the QR, saves the QR PNG to `src/logs/qr_codes/` and logs to Sheets. Its `finally` block disables the ROI for `ROI_DISABLE_AFTER_CAPTURE_S`.

**Google services fail soft.** If init fails, the error is logged, `drive`/`sheets` stay `None` and the exhibit still runs, but captures are not uploaded. The Sheets ID comes from `GOOGLE_SHEETS_SPREADSHEET_ID` or the first line of `keys/sheet_id.txt` (an ID or a full URL).

**Configuration:** all tunables are in `src/constant.py`, imported as `cfg`.

**Logging:** `log.py` uses the stdlib `RotatingFileHandler` (`src/logs/log.txt`, 1 MB per file, `BACKUP_COUNT` files in total) and also logs to the console.
