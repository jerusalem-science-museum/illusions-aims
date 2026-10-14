# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A museum exhibit (Ames room / "illusions-aims") that runs on a Raspberry Pi 4 with a Pi camera (OV5647 / Camera Module v1 without an IR-cut filter; a USB webcam still works with `--camera usb`). It shows a live camera feed fullscreen. When a visitor covers a small Region Of Interest (ROI) in the frame, it runs a countdown and flash, then captures a photo. The photo is composited into a branded template, uploaded to Google Drive, and turned into a QR code that links to it. Events can also be logged to a Google Sheet.

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

Pi camera tools: `python3 main.py --calibrate-camera` finds the fixed exposure, gain and white balance for `constant.py` in the room's lighting (white balance from a gray card, since AWB is fooled by infrared). `python3 picam_check.py` checks the tuning, the YUV conversion against the ISP's RGB, speed, and that nothing adapts to the scene.

Runtime requirements: picamera2 from apt (the venv uses `--system-site-packages`; numpy and OpenCV also come from apt, since pip's numpy 2.x breaks picamera2), and for usable speed on the Pi, a GPU-capable SDL2 renderer (the `vc4-kms-v3d` driver; the app logs a warning when it falls back to software rendering). Python deps are in `requirements.txt` (the root and `src/` copies differ slightly; `setup.sh` installs from the one next to it). The app exits on q or Esc, when the window is closed, or on SIGINT/SIGTERM.

To check rendering speed or output without the camera, call `Exhibit(None, None).render(frame, t)` with a synthetic 640×384 BGRA frame. To see the composited screen without a display, run `Display` with `SDL_VIDEODRIVER=dummy` and save `display.renderer.to_surface()`.

## Architecture

**The display composites on the GPU** (`display.py`, pygame's SDL2 renderer). Each frame, only the camera image (camera resolution, BGRA) is uploaded. The GPU crops and scales it into the template window, draws the template over it with its alpha, then the QR textures, and scales the 1920×1080 canvas to the screen. The template and each QR are uploaded once. The ROI box, countdown, flash and debug panel are drawn by OpenCV on the camera-resolution image before upload. This replaced an ffplay pipe, which used over a core of the Pi 3B on software scaling.

**Images are BGR everywhere** (OpenCV's native order): camera frames, the template, the GPU uploads and the saved JPEG. The Pi camera's YUV420 is converted straight to BGRA for uploads (and to BGR for the photo) by `camera.yuv_to`; the camera is configured as BT.601 limited range (`Smpte170m`), which is what OpenCV's I420 conversion assumes, and picamera2's full-range default would distort the colors. A USB frame is converted BGR→BGRA; BGRA bytes are SDL's ARGB8888, so SDL copies them without converting. Keep it that way. Draw colors are `(B, G, R)`.

**Modules:**
- `camera.py`: `PiCamera` and `UsbCamera` share one interface (`latest`, `bgra(raw)`, `photo()`, `status()`, `release()`); each keeps only the latest frame using a background thread.
  - `PiCamera` runs one picamera2 configuration with two YUV420 streams of every frame: `lores` (640 wide, preview + ROI) and `main` (1152 wide, copied only for a photo). It pins the full-view 1296×972 sensor mode, and the ISP crops the sensor to the template window's aspect (`ScalerCrop`) and mirrors it (`Transform`, software fallback).
  - Everything is fixed: exposure, gain, white balance, frame rate and processing controls (`PICAM_*`), plus a NoIR tuning file patched to disable adaptive contrast (`ce_enable`) and adaptive lens shading (`alsc n_iter`), an optional fixed color matrix (`PICAM_CCM`), and `IR_FILTER`: since there's no IR-cut filter, `ir_matrix` pulls red toward the green/blue average in front of the ISP's color matrix (neutrals unchanged, no per-frame cost).
- `main.py`:
  - `Exhibit` holds the trigger → countdown → flash → capture sequence. `render()` takes the camera's BGRA frame (already mirrored) and returns it with overlays, its crop for the window, and the debug panel. It does no I/O.
- `display.py`: `Display` owns the window and the GPU textures, draws one frame per `show()`, and handles q/Esc.
  - `main()` is the loop, paced to `PREVIEW_FPS`.
- `render.py`: `Template` loads `MOCKUP_PNG` once at the canvas size, both straight (`bgra`, for the GPU) and premultiplied (for the photo). The picture window is the bounding box of its transparent area.
  - `crop()` is the center crop of the camera frame with the window's aspect ratio (no stretching), shared by the live view and the photo, so the preview matches the uploaded photo.
  - `photo()` renders the capture on the CPU (`fit()` resizes the crop, `blend()` puts it under the template) and crops it to the gold frame.
  - The file also has the countdown, flash, ROI box and QR helpers.
- `roi_detector.py`: computes the ROI rectangle and patch (`ROI_*` are fractions of the camera frame, on the mirrored image) and contains `ROITriggerDetector`.
  - Baseline: a reference ROI patch and per-pixel noise, averaged over `BASELINE_SECONDS`.
  - Trigger: the fraction of changed pixels reaches `CHANGED_FRAC_THRESHOLD`, or the gray NCC with the reference drops below `NCC_THRESHOLD` (only when the reference is textured), for `HOLD_SECONDS`.
  - Also handles the adaptive baseline, the grace period, the cooldown and the disable window after a capture.
- `qr_queue.py`: the newest QR goes in the big slot and older ones slide through the 3 archive cells (`QR_SLOT_*`). The queue clears after `QR_QUEUE_RESET_S`. Each QR is resized once on push (in the capture thread); `placements()` gives the display each QR's animated position and size.
- `google_upload.py`:
  - `DriveUploader` uploads JPEG bytes from memory over one shared, locked httplib2 connection, with a 30 s timeout and a 45 s keepalive. The "anyone with the link" permission is set asynchronously.
  - `SheetsLogger` appends rows from its own worker thread.

**Per-frame CPU work stays at camera resolution** (one YUV→BGRA conversion, ROI, overlays). All scaling and compositing is on the GPU. Keep it that way: per-frame numpy work at screen size, and ffplay's software scaling, are what made the Pi slow.

**Capture:** when the flash ends, `_capture` runs on a thread and takes `camera.photo()` (the Pi camera's next `main` frame). It renders `Template.photo`, encodes the JPEG and uploads it, pushes the QR, saves the QR PNG to `src/logs/qr_codes/` and logs to Sheets. Its `finally` block disables the ROI for `ROI_DISABLE_AFTER_CAPTURE_S`.

**Google services fail soft.** If init fails, the error is logged, `drive`/`sheets` stay `None` and the exhibit still runs, but captures are not uploaded. The Sheets ID comes from `GOOGLE_SHEETS_SPREADSHEET_ID` or the first line of `keys/sheet_id.txt` (an ID or a full URL).

**Configuration:** all tunables are in `src/constant.py`, imported as `cfg`.

**Logging:** `log.py` uses the stdlib `RotatingFileHandler` (`src/logs/log.txt`, 1 MB per file, `BACKUP_COUNT` files in total) and also logs to the console.
