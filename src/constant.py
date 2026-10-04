import os
import sys

# =========================================================
# CONFIGURATION SETTINGS
# =========================================================

# ---------------------------------------------------------
# UI / LAYOUT CONSTANTS (Editable)
# ---------------------------------------------------------

# LIVE preview overlays:
# - If True: Show decorative frame and logo on the live preview screen.
# - If False: Live preview stays clean; overlays can still be burned into the saved/uploaded photo.
PREVIEW_APPLY_OVERLAYS = True
USE_CUSTOM_TEMPLATE_MODE = True

# Saved/uploaded photo overlays:
# - If True: Burns the frame and logo into the final saved image.
CAPTURE_APPLY_OVERLAYS = True

# Letterbox background color (areas where the camera image does NOT fill the screen)
# BGR format (OpenCV): (Blue, Green, Red) -> (0, 0, 0) is pure Black
LETTERBOX_BG_BGR = (0, 0, 0)

# How the camera image is positioned inside the display area (when letterboxing happens)
# Options: 'c' (center), 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right)
CAMERA_ANCHOR = 'c'
CAMERA_ANCHOR_MARGIN_PX = 0

# Screen preview and camera resolution dimensions
PREVIEW_W = 1920
PREVIEW_H = 1080
CAMERA_RESOLUTION = (640, 480)
FRAME_WIDTH, FRAME_HEIGHT = CAMERA_RESOLUTION

# Hardware camera index (0 is usually the default built-in/USB webcam)
CAM_INDEX = 0

# Logo sizing mode (applied to SAVED/UPLOADED photo when CAPTURE_APPLY_OVERLAYS=True)
# - 'scale': Use LOGO_SCALE (fraction of image width)
# - 'pixels': Use LOGO_TARGET_W_PX (absolute pixel width)
LOGO_SIZE_MODE = 'scale'
LOGO_TARGET_W_PX = None       # e.g., 420 (pixels). None means disabled for 'pixels' mode.


# ---------------------------------------------------------
# GOOGLE DRIVE & CLOUD STORAGE (Service Account)
# ---------------------------------------------------------
CAPTURE_DIR = "captures"
BASIC_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(BASIC_PATH)

# Path to local Google Cloud credentials and keys
KEYS_PATH = os.path.join(BASIC_PATH, 'keys')
GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'logger-176517.json')

# Google Drive folder ID where captured museum photos will be uploaded
GOOGLE_DRIVE_FOLDER_ID = None
GOOGLE_DRIVE_MAKE_PUBLIC = True

# URL Shortening settings for generated QR code links
ENABLE_URL_SHORTENER = False
SHORTENER_BACKEND = "tinyurl"


# ---------------------------------------------------------
# LOGGING (Local Rotation Logs)
# ---------------------------------------------------------
LOG_FOLDER = os.path.join(os.path.dirname(__file__), "logs")  # Target logs directory
MAX_SIZE_PER_LOG_FILE = 1 * 1024 * 1024                       # 1MB per file limit
BACKUP_COUNT = 10  # Max log files tracking. If all 10 are full, the oldest is overwritten.


# ---------------------------------------------------------
# GOOGLE SHEETS (Optional Remote Event Logging)
# ---------------------------------------------------------
# Set a Spreadsheet ID to enable remote cloud logging.
ENABLE_SHEETS_LOG = True
GOOGLE_SHEETS_SPREADSHEET_ID = None  # e.g., "1AbC..." (found in the Google Sheet URL)
GOOGLE_SHEETS_SPREADSHEET_ID_FILE = os.path.join(KEYS_PATH, "sheet_id.txt") # Contains Sheet ID or URL (1st line)
GOOGLE_SHEETS_WORKSHEET_NAME = "pi04"  # The specific tab/worksheet name inside the Google Sheet

# If False: Does NOT log minor QR_OK / QR_ERROR system events to Google Sheets to save API quota
SHEETS_LOG_QR_EVENTS = False

# ---- Custom Template Frame Settings ----

# Orange color detection bounds (HSV) for the frame interior masking
ORANGE_MASK_LOW = [5, 140, 140]
ORANGE_MASK_HIGH = [22, 255, 255]

# OPTIONAL: Coordinates for dynamic QR overlay if needed in the future
# Based on your template width/height proportions (Right side area)
DYNAMIC_QR_TARGET_W_PX = 120
DYNAMIC_QR_ANCHOR_X = 0.85  # Relative position from left (85% W)
DYNAMIC_QR_ANCHOR_Y = 0.10  # Relative position from top (10% H)

# ---------------------------------------------------------
# PNG GRAPHIC OVERLAYS
# ---------------------------------------------------------
PIC_DIR = os.path.join(BASIC_PATH, "pic")
FRAME_PNG = os.path.join(PIC_DIR, "frame.png")
LOGO_PNG = os.path.join(PIC_DIR, "logo.png")
MOCKUP_PNG = os.path.join(PIC_DIR, "thumbnail1080.png")  # RGBA; picture window is transparent

# Saved/uploaded photo: template is rendered at this width, then cropped to the gold frame only
CAPTURE_TEMPLATE_W = 1920

LOGO_SCALE = 0.5  # Fraction of the image width used for logo width (e.g., 0.5 = 50% width)

# Logo positioning rules:
# - Set LOGO_ANCHOR to one of: 'br','tr','bl','tl','c' (bottom-right, top-right, etc.)
# - OR override with LOGO_POS_X / LOGO_POS_Y:
#     * None => Calculated automatically based on LOGO_ANCHOR + margins
#     * 0.0..1.0 => Percentage of screen space (0.0=left/top, 1.0=right/bottom)
#     * >= 1 => Absolute positions in pixels
LOGO_ANCHOR = 'br'
LOGO_POS_X = 155
LOGO_POS_Y = 350
LOGO_MARGIN_X = 20
LOGO_MARGIN_Y = 20

# ---------------------------------------------------------
# IMAGE FLIPPING (Mirroring for Museum Mirror Exhibits)
# ---------------------------------------------------------
FLIP_PREVIEW = True   # Mirrors the screen live feed so visitors see themselves naturally
FLIP_CAPTURE = True   # Mirrors the final saved photo accordingly
FLIP_MODE = "h"       # Mirror direction: "h" = horizontal, "v" = vertical, "hv" = both


# ---------------------------------------------------------
# CAPTURING UX (Countdown & Visual Feedback)
# ---------------------------------------------------------
COUNTDOWN_SECONDS = 3     # Length of countdown before taking a picture
FLASH_DURATION_S = 0.12   # Duration of the white screen flash overlay effect
FLASH_COLOR = "white"     # Flash screen color option


# ---------------------------------------------------------
# MOTION DETECTION VIA ROI (Region of Interest)
# ---------------------------------------------------------
# The boundary and size parameters of the detection box
ROI_W = 40   # Width of the motion detection bounding box
ROI_H = 40   # Height of the motion detection bounding box
ROI_X = 480  # Horizontal position (Pixels from Left) -> Placed in the Upper-Right corner
ROI_Y = 80   # Vertical position (Pixels from Top) -> Placed in the Upper-Right corner

BASELINE_SECONDS = 2.0        # Time window in seconds to capture and calculate background average
TRIGGER_DIST_THRESHOLD = 20.0 # Min RGB distance from baseline to count as a change (floor; was 70). Lower = more sensitive.
NOISE_SIGMA_MULT = 6.0        # Effective threshold = max(TRIGGER_DIST_THRESHOLD, this * measured baseline noise)
BASELINE_ADAPT_SECONDS = 5.0  # Baseline slowly follows lighting drift (time constant); 0 disables
HOLD_GRACE_SECONDS = 0.2      # Brief dips below threshold don't reset the hold timer
BRIGHTNESS_COMP = False       # If True, discount uniform gain shifts (can hide shadows/covering; off by default)
HOLD_SECONDS = 0.5            # Time in seconds visitor must hold their position/QR to trigger capture
COOLDOWN_SECONDS = 2.0        # Safety lock window right after a trigger event happens

# Disable motion detector completely for X seconds after a successful capture
# Gives visitors time to view/scan their QR code without triggering the camera again
ROI_DISABLE_AFTER_CAPTURE_S = 1.0

DRAW_ROI_RECT = True  # If True, renders the green/red helper rectangle boundary on the screen

# ---------------------------------------------------------
# VISITOR QR HISTORY DISPLAY (UI Strip Layout)
# ---------------------------------------------------------
# QR display code layout options
QR_SIZE_MODE = 'fixed'         # Scaling behavior: 'auto' or 'fixed'
QR_FIXED_SIZE_PX = 150        # Width/Height of the QR code graphic (if size mode is 'fixed')
QR_SIZE_MIN = 80
QR_SIZE_MAX = 320

QR_STRIP_ALIGN = 'center'     # Alignment of the QR badge layout on the monitor: 'center', 'left', 'right'
QR_STRIP_MARGIN_PX = 0

# QR history background presentation properties
QR_BAR_BG = '#000000'         # Background canvas color for the QR code container (Hex format)
QR_HISTORY = 3                # Max number of previous QR codes displayed simultaneously on screen
QR_SIZE = QR_FIXED_SIZE_PX    # Legacy alias variable map; tracks target layout sizes
QR_GAP = 10                   # Distance space padding between adjacent UI badges
QR_ANIM_STEPS = 12            # Number of slide frames for the QR appearance animation
QR_ANIM_DELAY_MS = 15         # Frame step execution speed delay for slide transitions

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
QR_SLOT_PADDING_PX = 4        # gap between a QR and the divider lines / slot edge (px at render size)
QR_QUEUE_RESET_S = 60        # seconds without a new capture before all QRs are cleared