import os
import sys

# =========================================================
# CONFIG
# =========================================================

# ---------------------------------------------------------
# UI / LAYOUT CONSTANTS (editable)
# ---------------------------------------------------------
# LIVE preview overlays:
# - If True: show decorative frame+logo on the live preview (screen).
# - If False: live preview stays clean; overlays can still be burned into the saved/uploaded photo.
PREVIEW_APPLY_OVERLAYS = False

# Saved/uploaded photo overlays:
CAPTURE_APPLY_OVERLAYS = True

# Letterbox background color (areas where the camera image does NOT fill the screen)
# BGR format (OpenCV): (Blue, Green, Red)
LETTERBOX_BG_BGR = (0, 0, 0)

# How the camera image is positioned inside the display area (when letterboxing happens)
# Options: 'c' (center), 'tl','tr','bl','br'
CAMERA_ANCHOR = 'c'
CAMERA_ANCHOR_MARGIN_PX = 0
PREVIEW_W = 640
PREVIEW_H = 480
CAMERA_RESOLUTION = (640, 480)
FRAME_WIDTH, FRAME_HEIGHT = CAMERA_RESOLUTION
CAM_INDEX = 0


# Logo sizing mode (applied to SAVED/UPLOADED photo when CAPTURE_APPLY_OVERLAYS=True)
# - 'scale': use LOGO_SCALE (fraction of image width)
# - 'pixels': use LOGO_TARGET_W_PX (absolute pixels in the image you overlay on)
LOGO_SIZE_MODE = 'scale'
LOGO_TARGET_W_PX = None       # e.g. 420 (pixels). None means disabled for 'pixels' mode.


# ---------- GOOGLE DRIVE (service account) ----------
CAPTURE_DIR = "captures"
BASIC_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(BASIC_PATH)

KEYS_PATH_CANDIDATES = [
    os.path.join(BASIC_PATH, 'keys', 'arad'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'keys', 'arad'),
]

KEYS_PATH = next((p for p in KEYS_PATH_CANDIDATES if os.path.isdir(p)), KEYS_PATH_CANDIDATES[0])
GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'logger-176517.json')

GOOGLE_DRIVE_FOLDER_ID = None
GOOGLE_DRIVE_MAKE_PUBLIC = True

ENABLE_URL_SHORTENER = False
SHORTENER_BACKEND = "tinyurl"

# ---------- LOGGING ----------
# logging values
LOG_FOLDER = os.path.join(os.path.dirname(__file__), "logs")  # get the path of the logs folder
MAX_SIZE_PER_LOG_FILE = 1 * 1024 * 1024  # 1MB
BACKUP_COUNT = 10  # max number of log files, if all 10 are full, the first one will be deleted, rotating the rest

# ---------- GOOGLE SHEETS (optional log) ----------
# Met un Spreadsheet ID pour activer la journalisation.
ENABLE_SHEETS_LOG = True
GOOGLE_SHEETS_SPREADSHEET_ID = None   # ex: "1AbC..." (dans l'URL du Google Sheet)
GOOGLE_SHEETS_SPREADSHEET_ID_FILE = os.path.join(KEYS_PATH,"sheet_id.txt") #  contient soit l'ID, soit l'URL du Sheet (1ère ligne)
GOOGLE_SHEETS_WORKSHEET_NAME = "pi04"  # the name of the tab in google sheet

# Si False: on ne log PAS les évènements QR_OK / QR_ERROR dans Google Sheets
SHEETS_LOG_QR_EVENTS = False


# ---------- PNG OVERLAYS ----------
PIC_DIR =  os.path.join(BASIC_PATH, "pic")
FRAME_PNG = os.path.join(PIC_DIR, "frame.png")
LOGO_PNG  = os.path.join(PIC_DIR, "logo.png")

LOGO_SCALE = 0.5  # fraction of the image width used for logo width
# Logo position options:
# - set LOGO_ANCHOR to one of: 'br','tr','bl','tl','c' (bottom-right, top-right, ...)
# - OR set LOGO_POS_X / LOGO_POS_Y:
#     * None => auto based on LOGO_ANCHOR + margins
#     * 0.0..1.0 => percentage of available space (0.0=left/top, 1.0=right/bottom)
#     * >= 1 => pixels (absolute)
LOGO_ANCHOR = 'br'
LOGO_POS_X = 220
LOGO_POS_Y = 350
LOGO_MARGIN_X = 20
LOGO_MARGIN_Y = 20

# ---------- FLIP ----------
FLIP_PREVIEW = True
FLIP_CAPTURE = True
FLIP_MODE = "h"          # "h"=horizontal, "v"=vertical, "hv"=les deux

# ---------- UX CAPTURE ----------
COUNTDOWN_SECONDS = 3
FLASH_DURATION_S = 0.12
FLASH_COLOR = "white"

# ---------- DÉCLENCHEMENT PAR ROI ----------
ROI_W = 20
ROI_H = 20
ROI_X = 600
ROI_Y = 10

BASELINE_SECONDS = 2.0
TRIGGER_DIST_THRESHOLD = 50.0
HOLD_SECONDS = 0.3
COOLDOWN_SECONDS = 2.0

# Désactive le ROI juste après l’envoi du QR
ROI_DISABLE_AFTER_CAPTURE_S = 10.0

DRAW_ROI_RECT = True

# ---------- QR HISTORY ----------
# QR strip sizing/placement
QR_SIZE_MODE = 'fixed'         # 'auto' or 'fixed'
QR_FIXED_SIZE_PX = 150        # used only if QR_SIZE_MODE == 'fixed'
QR_SIZE_MIN = 80
QR_SIZE_MAX = 320

QR_STRIP_ALIGN = 'center'     # 'center' or 'left' or 'right'
QR_STRIP_MARGIN_PX = 0

# QR strip background (Tk color name or hex)
QR_BAR_BG = '#000000'
QR_HISTORY = 4
QR_SIZE = QR_FIXED_SIZE_PX  # legacy alias; prefer QR_FIXED_SIZE_PX / QR_SIZE_MODE
QR_GAP = 10
QR_ANIM_STEPS = 12
QR_ANIM_DELAY_MS = 15
