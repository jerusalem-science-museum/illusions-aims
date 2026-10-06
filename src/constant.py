import os

# =========================================================
# CONFIGURATION SETTINGS
# =========================================================
# Images are BGR (OpenCV order) everywhere in the app, so colors below are (Blue, Green, Red).

# ---------------------------------------------------------
# SCREEN / CAMERA
# ---------------------------------------------------------
PREVIEW_W = 1920              # Output frame size sent to ffplay (ffplay then scales it fullscreen)
PREVIEW_H = 1080
PREVIEW_FPS = 25
CAMERA_RESOLUTION = (640, 480)
CAM_INDEX = 0                 # Hardware camera index (0 is usually the USB webcam)
# Lock auto exposure / white balance / focus so scene changes don't shift the ROI color.
CAM_LOCK_AUTO = False
CAM_EXPOSURE = None           # Manual exposure (V4L2 units, see `v4l2-ctl -L`); None = freeze the value auto picked
CAM_GAIN = None               # None = freeze the current gain
CAM_LOCK_SETTLE_S = 2.0       # Let auto exposure settle this long before freezing it

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
# keys/ and pic/ live at the repo root (parent of src/)
BASIC_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
KEYS_PATH = os.path.join(BASIC_PATH, 'keys')
PIC_DIR = os.path.join(BASIC_PATH, "pic")
# RGBA template; the picture window is its transparent area. Should be PREVIEW_W x PREVIEW_H.
MOCKUP_PNG = os.path.join(PIC_DIR, "thumbnail1080.png")

# ---------------------------------------------------------
# GOOGLE DRIVE (Service Account)
# ---------------------------------------------------------
GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'logger-176517.json')
GOOGLE_DRIVE_FOLDER_ID = None     # Drive folder to upload into (None = service account root)
GOOGLE_DRIVE_MAKE_PUBLIC = True   # Anyone with the link can view (needed for the QR to work)

# URL shortening for the QR link
ENABLE_URL_SHORTENER = False
SHORTENER_BACKEND = "tinyurl"

# ---------------------------------------------------------
# GOOGLE SHEETS (Optional remote event logging)
# ---------------------------------------------------------
ENABLE_SHEETS_LOG = True
GOOGLE_SHEETS_SPREADSHEET_ID = None  # e.g., "1AbC..."; if None, read from the file below
GOOGLE_SHEETS_SPREADSHEET_ID_FILE = os.path.join(KEYS_PATH, "sheet_id.txt")  # Sheet ID or URL (1st line)
GOOGLE_SHEETS_WORKSHEET_NAME = "pi04"  # Tab inside the spreadsheet (created if missing)
SHEETS_LOG_QR_EVENTS = False           # Also log QR_OK / QR_ERROR rows

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
LOG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
MAX_SIZE_PER_LOG_FILE = 1 * 1024 * 1024  # 1MB per file
BACKUP_COUNT = 10                        # Max log files kept (log.txt + 9 rotated)

# ---------------------------------------------------------
# IMAGE FLIPPING (Mirroring)
# ---------------------------------------------------------
FLIP_PREVIEW = True   # Mirrors the live feed so visitors see themselves naturally
FLIP_CAPTURE = True   # Mirrors the saved photo accordingly
FLIP_MODE = "h"       # "h" = horizontal, "v" = vertical, "hv" = both

# ---------------------------------------------------------
# CAPTURING UX (Countdown & Flash)
# ---------------------------------------------------------
COUNTDOWN_SECONDS = 3
FLASH_DURATION_S = 0.12
FLASH_COLOR = "white"  # "white" or "black"

# ---------------------------------------------------------
# ROI TRIGGER (Region of Interest)
# ---------------------------------------------------------
# Box in camera pixels, on the mirrored image (as seen on screen)
ROI_W = 40
ROI_H = 40
ROI_X = 480
ROI_Y = 80

BASELINE_SECONDS = 2.0        # Time to average the empty-ROI color at startup
TRIGGER_DIST_THRESHOLD = 20.0 # Min RGB distance from baseline to count as a change. Lower = more sensitive.
NOISE_SIGMA_MULT = 6.0        # Effective threshold = max(TRIGGER_DIST_THRESHOLD, this * measured baseline noise)
BASELINE_ADAPT_SECONDS = 5.0  # Baseline slowly follows lighting drift (time constant); 0 disables
HOLD_GRACE_SECONDS = 0.2      # Brief dips below threshold don't reset the hold timer
BRIGHTNESS_COMP = False       # If True, discount uniform gain shifts (can hide shadows/covering)
HOLD_SECONDS = 0.5            # How long the ROI must stay covered to trigger
COOLDOWN_SECONDS = 2.0        # No new trigger for this long after a trigger
ROI_DISABLE_AFTER_CAPTURE_S = 1.0  # No new trigger for this long after a capture finishes

DRAW_ROI_RECT = True  # Draw the ROI box (green = ready, red = not ready)

# ---------------------------------------------------------
# QR MOVING QUEUE (big slot + 3-cell "archive" box in the template)
# ---------------------------------------------------------
# Slot rects as fractions (x0, y0, x1, y1) of the template, measured from pic/thumbnail1080.png
# (1920x1080): the archive box is x=1492-1893, y=872-1004 with white dividers at x=1624 and
# x=1757; the big slot sits in the orange column between the Hebrew title and the line at y=750.
# Keys: BIG = newest QR (no border); A1 = archive right cell, A2 = middle, A3 = left.
_TPL_W, _TPL_H = 1920, 1080
QR_SLOT_RECTS_FRAC = {
    'BIG': (1567 / _TPL_W, 385 / _TPL_H, 1835 / _TPL_W, 745 / _TPL_H),
    'A1': (1758 / _TPL_W, 873 / _TPL_H, 1893 / _TPL_W, 1004 / _TPL_H),
    'A2': (1625 / _TPL_W, 873 / _TPL_H, 1757 / _TPL_W, 1004 / _TPL_H),
    'A3': (1493 / _TPL_W, 873 / _TPL_H, 1624 / _TPL_W, 1004 / _TPL_H),
}
# Fill order, newest first. A QR enters BIG, then moves right to left through the archive (Hebrew RTL)
QR_SLOT_ORDER = ['BIG', 'A1', 'A2', 'A3']
QR_SLOT_PADDING_PX = 4   # gap between a QR and the divider lines / slot edge (px at render size)
QR_ANIM_S = 0.18         # duration of the slide between slots
QR_QUEUE_RESET_S = 60    # seconds without a new capture before all QRs are cleared
