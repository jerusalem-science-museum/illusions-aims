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
PREVIEW_W = 640
PREVIEW_H = 480
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
KEYS_PATH = os.path.join(BASIC_PATH, 'keys', 'arad')
GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'logger-176517.json')

# KEYS_PATH = os.path.join(BASIC_PATH, 'keys', 'nathan')
# GOOGLE_SERVICE_ACCOUNT_JSON = os.path.join(KEYS_PATH, 'cameraqr-489411.json')

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


# ---------------------------------------------------------
# PNG GRAPHIC OVERLAYS
# ---------------------------------------------------------
PIC_DIR = os.path.join(BASIC_PATH, "pic")
FRAME_PNG = os.path.join(PIC_DIR, "frame.png")
LOGO_PNG = os.path.join(PIC_DIR, "logo.png")

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
TRIGGER_DIST_THRESHOLD = 70.0 # Sensitivity threshold. Higher means less sensitive to ambient noise.
HOLD_SECONDS = 0.5            # Time in seconds visitor must hold their position/QR to trigger capture
COOLDOWN_SECONDS = 2.0        # Safety lock window right after a trigger event happens

# Disable motion detector completely for X seconds after a successful capture
# Gives visitors time to view/scan their QR code without triggering the camera again
ROI_DISABLE_AFTER_CAPTURE_S = 10.0

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