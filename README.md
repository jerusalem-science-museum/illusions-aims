# AIMS Room Camera Exhibit

A live camera feed is shown fullscreen inside a branded template. When a visitor covers a small Region Of Interest (ROI) in the frame, the app counts down, flashes and takes a photo. The photo is placed in the template's gold frame and uploaded to Google Drive. A QR code linking to it then appears on screen. Each capture can also be logged to a Google Sheet.

It runs on a **Raspberry Pi 4** with a **Pi camera** (OV5647 / Camera Module v1 NoIR, with no IR-cut filter).

## Installation

On a fresh Raspberry Pi OS (Bookworm, desktop):

1. **Clone the repo:**
   ```bash
   git clone <repo-url> illusions-aims
   cd illusions-aims
   ```
2. **Add the keys.** Download the `keys` folder from the [Museum's OneDrive](https://madaorgil-my.sharepoint.com/shared?id=%2Fsites%2FMakeMada%2FShared%20Documents%2F2%2E%20%D7%AA%D7%A2%D7%A8%D7%95%D7%9B%D7%95%D7%AA%2F%D7%90%D7%A9%D7%9C%D7%99%D7%95%D7%AA%2F2%2E%20%D7%91%D7%99%D7%AA%20%D7%9E%D7%9C%D7%90%D7%9B%D7%94%2F2%2E%20%D7%9E%D7%95%D7%A6%D7%92%D7%99%D7%9D%2F%D7%90%D7%99%D7%99%D7%9E%D7%A1%2F6%2E%20%D7%A7%D7%95%D7%93&listurl=https%3A%2F%2Fmadaorgil%2Esharepoint%2Ecom%2Fsites%2FMakeMada%2FShared%20Documents&viewid=b5506206%2Dcce1%2D4963%2D8af9%2D671538374454) and put it at the **repo root** (next to `src/`, not inside it). The keys are gitignored. Without them the exhibit still runs, but photos are not uploaded and no QR appears.
3. **Run the setup script** from `src/`:
   ```bash
   cd src
   chmod +x setup.sh && ./setup.sh
   ```
   It asks for your sudo password and an AnyDesk unattended-access password at the start, then runs on its own. It does the following:
   - installs the apt packages: Python, `picamera2`/`libcamera`, numpy, OpenCV and `rpicam-apps`
   - installs AnyDesk for remote access and prints its ID at the end. **Write the ID down.**
   - creates `src/.venv` with access to the system packages, and installs `src/requirements.txt` into it
   - switches the display to X11 (AnyDesk needs it), enables desktop autologin and turns off screen blanking
   - adds an autostart entry (`~/.config/autostart/kiosk.desktop`) that runs `run.sh` on boot
4. **Reboot:** `sudo reboot`. The exhibit starts by itself fullscreen.
5. **Calibrate the camera in the room's lighting.** See [Camera](#camera) below. Do this again whenever the lighting changes.

> Don't install numpy or OpenCV with pip on the Pi. pip's numpy 2.x breaks the apt modules that picamera2 depends on. `setup.sh` handles this already.

## Running

```bash
cd src
./run.sh                 # activates .venv and runs main.py with any arguments you pass
python3 main.py --help   # all options
```

| Option | What it does |
|---|---|
| `--windowed` | Run in a window instead of fullscreen |
| `--offline` | Don't use Google. Captures are saved to `src/logs/captures/` and the QR encodes the file path |
| `--debug` | Shows a panel with the ROI stats, and writes `logs/roi_stats_*.csv` and `logs/roi_triggers/*.png` |
| `--camera picam\|usb` | Choose the camera (default: `CAM_BACKEND` in `constant.py`, which is `picam`) |
| `--cam N` | USB camera index |
| `--calibrate-camera` | Pi camera: find the fixed exposure, gain and white balance, then exit |

To quit, press **q** or **Esc**, or press Ctrl+C in the terminal.

## Camera

**The exhibit uses the Pi camera.** The USB webcam path is still supported, as a fallback and for developing on a PC.

### Pi camera (default)

Everything is fixed: exposure, gain, white balance, frame rate and color processing. Because nothing adapts to the scene, a visitor walking past doesn't shift the image, and only covering the ROI triggers a capture. The ISP crops the sensor to the template window's shape and mirrors the image, so the CPU only has to convert one small frame each time. Photos come from a second stream of the same frame at 1152 px wide.

The camera has no IR-cut filter, so skin, fabrics and wood come out too red. `IR_FILTER` / `IR_FILTER_STRENGTH` reduce this in the ISP's color matrix, at no per-frame cost.

**Calibration**, on site:
```bash
cd src && source .venv/bin/activate
python3 main.py --calibrate-camera
```
A preview window opens and walks you through it:
1. **Auto exposure settles** for a few seconds on the normal scene. Don't hold the card up yet.
2. **Hold a gray or white card** (matte paper works) so it covers the whole green box in the middle, and press **Enter**. The window shows the box's live color values and warns you if the card is too bright (clipped).
3. **Keep the card in place** while the white balance is adjusted. This takes a few seconds.

Press **Esc** at any point to cancel without saving. Run it from the Pi's desktop or over AnyDesk, since the window needs a display. Over plain SSH it falls back to a terminal prompt with no preview.

 The exposure, gain and white balance are saved to `src/camera_calibration.json`, and the exhibit uses them from its next start. There's nothing to copy.

The file belongs to that Pi and that room, so it's gitignored and a `git pull` won't touch it. If the file is missing or broken, the app logs a warning and falls back to the `PICAM_EXPOSURE_US` / `PICAM_ANALOGUE_GAIN` / `PICAM_COLOUR_GAINS` defaults in `constant.py`. You can also edit the JSON by hand.

**Checking the setup:** `python3 picam_check.py` checks the following:
- that the adaptive stages in the tuning file are off
- that the YUV→BGR conversion matches the ISP's own colors
- the conversion speed
- that exposure and white balance stay put while you wave a hand in front of the camera

### USB webcam

```bash
./run.sh --camera usb            # or set CAM_BACKEND = "usb" in constant.py
```
The USB settings are `CAMERA_RESOLUTION`, `CAM_INDEX` and `CAM_LOCK_AUTO`, in `constant.py`. Before relying on the USB camera in the exhibit, note these limits:
- The photo is the latest preview frame (640×480 by default), so it is much lower resolution than the Pi camera's photo.
- Auto exposure and white balance are not locked by default (`CAM_LOCK_AUTO = False`), so lighting changes can trigger the ROI.
- The `ROI_*` position was tuned for the Pi camera's framing. Re-check it with `--debug`.

## Configuration

All settings are in [src/constant.py](src/constant.py), grouped by section. The ones you're most likely to change on site:
- **ROI position and sensitivity:** `ROI_X/Y/W/H` (fractions of the mirrored frame, as seen on screen), `CHANGED_FRAC_THRESHOLD`, `HOLD_SECONDS`. Use `--debug` to see the live values.
- **Camera:** `IR_FILTER_STRENGTH`, `PICAM_SATURATION`, `PICAM_ZOOM`
- **Capture:** `COUNTDOWN_SECONDS`, `FLASH_DURATION_S`, `QR_QUEUE_RESET_S`
- **Google:** `GOOGLE_DRIVE_FOLDER_ID`, `GOOGLE_SHEETS_WORKSHEET_NAME` (the tab name, e.g. `pi04`; it's created if it's missing), `ENABLE_SHEETS_LOG`

## Logs

`src/logs/log.txt` is a rotating log, 1 MB per file. Every 10 s it records the frame rate, the camera's exposure and gains, and the CPU temperature. If the Pi has no GPU renderer it logs a warning; the app then falls back to software rendering, which is too slow. Saved QR codes go to `src/logs/qr_codes/`.

## Project files

```
src/                 the application (all work happens here)
  main.py            main loop, trigger → countdown → flash → capture sequence, upload, QR, Sheets
  camera.py          PiCamera and UsbCamera (one shared interface), tuning, calibration
  display.py         fullscreen GPU compositing (pygame / SDL2)
  render.py          template, photo rendering, countdown, flash, ROI box, QR images
  roi_detector.py    ROI trigger detection
  roi_debug.py       --debug panel and stats
  qr_queue.py        the animated QR queue in the template's QR slots
  google_upload.py   Google Drive upload and Sheets logging
  constant.py        all settings
  log.py             rotating log file
  picam_check.py     Pi camera diagnostics
  setup.sh, run.sh   Pi installation and launcher
pic/                 template and overlay images (thumbnail1080.png is the active template)
keys/                Google credentials (gitignored, from OneDrive)
legacy/              old version, kept for reference only
```

How it works: each frame, the CPU converts one camera-resolution image and draws the overlays on it. The GPU scales it into the template window and composites the template and QR codes. See [CLAUDE.md](CLAUDE.md) for the architecture.
