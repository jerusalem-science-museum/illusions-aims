import os
import sys
import threading
import datetime
import time
import queue
import logging
import re
import tempfile
from typing import Optional

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from google.oauth2 import service_account
from google.auth.transport.requests import Request


# =========================================================
# CONFIG
# =========================================================
CAPTURE_DIR = "captures"

PREVIEW_W = 640
PREVIEW_H = 480

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

# QR strip sizing/placement
QR_SIZE_MODE = 'fixed'         # 'auto' or 'fixed'
QR_FIXED_SIZE_PX = 150        # used only if QR_SIZE_MODE == 'fixed'
QR_SIZE_MIN = 80
QR_SIZE_MAX = 320

QR_STRIP_ALIGN = 'center'     # 'center' or 'left' or 'right'
QR_STRIP_MARGIN_PX = 0

# QR strip background (Tk color name or hex)
QR_BAR_BG = '#000000'

# Logo sizing mode (applied to SAVED/UPLOADED photo when CAPTURE_APPLY_OVERLAYS=True)
# - 'scale': use LOGO_SCALE (fraction of image width)
# - 'pixels': use LOGO_TARGET_W_PX (absolute pixels in the image you overlay on)
LOGO_SIZE_MODE = 'scale'
LOGO_TARGET_W_PX = None       # e.g. 420 (pixels). None means disabled for 'pixels' mode.

CAM_INDEX = 0
CAMERA_RESOLUTION = (640, 480)
FRAME_WIDTH, FRAME_HEIGHT = CAMERA_RESOLUTION

# ---------- GOOGLE DRIVE (service account) ----------
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
LOG_LEVEL = "INFO"          # "DEBUG" / "INFO" / "WARNING" / "ERROR"
LOG_FILE = None              # ex: "camera_qr.log" (None => console only)

# ---------- GOOGLE SHEETS (optional log) ----------
# Met un Spreadsheet ID pour activer la journalisation.
ENABLE_SHEETS_LOG = True
GOOGLE_SHEETS_SPREADSHEET_ID = None   # ex: "1AbC..." (dans l'URL du Google Sheet)
GOOGLE_SHEETS_SPREADSHEET_ID_FILE = os.path.join(KEYS_PATH,"sheet_id.txt")  # contient soit l'ID, soit l'URL du Sheet (1ère ligne)
GOOGLE_SHEETS_WORKSHEET_NAME = "pi04"

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
QR_HISTORY = 4
QR_SIZE = QR_FIXED_SIZE_PX  # legacy alias; prefer QR_FIXED_SIZE_PX / QR_SIZE_MODE
QR_GAP = 10
QR_ANIM_STEPS = 12
QR_ANIM_DELAY_MS = 15


def _read_first_nonempty_line(path: str) -> str:
    """Return first non-empty, non-comment line from a txt file. Empty string if missing."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                return s
    except FileNotFoundError:
        return ''
    except Exception:
        return ''
    return ''

def extract_spreadsheet_id(value: str) -> str:
    """Accepts either a raw Spreadsheet ID or a full Google Sheets URL and returns the ID."""
    if not value:
        return ''
    v = value.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", v)
    if m:
        return m.group(1)
    return v


# =========================================================
# LOGGING SETUP
# =========================================================
def setup_logging():
    """Configure les logs console (et optionnellement fichier)."""
    level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        try:
            from logging.handlers import RotatingFileHandler
            handlers.append(RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"))
        except Exception:
            # fallback: pas de fichier
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
)

# =========================================================
# OUTILS: flip
# =========================================================
def apply_flip(img_bgr, mode: str):
    if mode == "h":
        return cv2.flip(img_bgr, 1)
    if mode == "v":
        return cv2.flip(img_bgr, 0)
    if mode == "hv":
        return cv2.flip(img_bgr, -1)
    return img_bgr


# =========================================================
# ÉTAPE 1) appliquer frame + logo sur l'image CAPTURE
# =========================================================
def overlay_rgba(dst_bgr, src_rgba, x, y):
    if src_rgba is None:
        return dst_bgr

    h, w = src_rgba.shape[:2]
    H, W = dst_bgr.shape[:2]

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return dst_bgr

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = dst_bgr[y1:y2, x1:x2]
    src = src_rgba[sy1:sy2, sx1:sx2]

    if src.shape[2] == 3:
        roi[:] = src
        dst_bgr[y1:y2, x1:x2] = roi
        return dst_bgr

    src_rgb = src[:, :, :3].astype(np.float32)
    alpha = (src[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    roi_f = roi.astype(np.float32)

    out = alpha * src_rgb + (1.0 - alpha) * roi_f
    dst_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return dst_bgr


def apply_frame_and_logo(frame_bgr):
    """
    Add the decorative frame + logo overlays on top of an image.

    Notes:
    - Works on any resolution (camera capture, preview, etc.)
    - Logo size is relative to the image width (LOGO_SCALE)
    - Logo position can be automatic (LOGO_ANCHOR) or manual (LOGO_POS_X/Y)
    - Uses simple caching so we don't re-decode PNG files every frame.
    """
    H, W = frame_bgr.shape[:2]

    # ---- Lazy caches (per-process) ----
    # resized overlays are cached by output size to avoid repeated PNG decode + resize.
    if not hasattr(apply_frame_and_logo, "_frame_cache"):
        apply_frame_and_logo._frame_cache = {}  # (W,H)->rgba
        apply_frame_and_logo._logo_orig = None  # original rgba
        apply_frame_and_logo._logo_cache = {}   # target_w->rgba

    # -------- frame overlay (full image) --------
    frame_rgba = apply_frame_and_logo._frame_cache.get((W, H))
    if frame_rgba is None:
        src_rgba = cv2.imread(FRAME_PNG, cv2.IMREAD_UNCHANGED)
        if src_rgba is not None:
            frame_rgba = cv2.resize(src_rgba, (W, H), interpolation=cv2.INTER_AREA)
            apply_frame_and_logo._frame_cache[(W, H)] = frame_rgba

    if frame_rgba is not None:
        frame_bgr = overlay_rgba(frame_bgr, frame_rgba, 0, 0)

    # -------- logo overlay --------
    if apply_frame_and_logo._logo_orig is None:
        apply_frame_and_logo._logo_orig = cv2.imread(LOGO_PNG, cv2.IMREAD_UNCHANGED)

    logo_rgba = apply_frame_and_logo._logo_orig
    if logo_rgba is None:
        return frame_bgr

    # Compute logo size (keep aspect ratio)
    # Compute logo width
    if str(LOGO_SIZE_MODE).lower() == 'pixels' and LOGO_TARGET_W_PX is not None:
        target_w = max(1, int(LOGO_TARGET_W_PX))
    else:
        # default: scale with image width
        target_w = max(1, int(W * float(LOGO_SCALE)))
    target_w = min(target_w, W)  # safety
    cached = apply_frame_and_logo._logo_cache.get(target_w)
    if cached is None:
        ratio = target_w / max(1, logo_rgba.shape[1])
        target_h = max(1, int(logo_rgba.shape[0] * ratio))
        target_h = min(target_h, H)
        cached = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        apply_frame_and_logo._logo_cache[target_w] = cached
    else:
        target_h = cached.shape[0]

    # Positioning:
    # 1) If LOGO_POS_X/Y are None => anchor logic
    # 2) If 0..1 => relative percentage of available space
    # 3) Else => pixel coordinates
    avail_x = max(0, W - target_w)
    avail_y = max(0, H - target_h)

    def _pos(val, avail, margin, anchor_side):
        if val is None:
            if anchor_side == "min":
                return int(margin)
            if anchor_side == "center":
                return int(avail / 2)
            return int(avail - margin)
        try:
            v = float(val)
        except Exception:
            v = 0.0
        if 0.0 <= v <= 1.0:
            return int(v * avail)
        return int(v)

    anchor = str(LOGO_ANCHOR).lower().strip()
    if anchor in ("br", "rb"):
        ax, ay = "max", "max"
    elif anchor in ("tr", "rt"):
        ax, ay = "max", "min"
    elif anchor in ("bl", "lb"):
        ax, ay = "min", "max"
    elif anchor in ("tl", "lt"):
        ax, ay = "min", "min"
    else:  # center
        ax, ay = "center", "center"

    x = _pos(LOGO_POS_X, avail_x, LOGO_MARGIN_X, ax)
    y = _pos(LOGO_POS_Y, avail_y, LOGO_MARGIN_Y, ay)

    # Clamp inside the image
    x = max(0, min(x, avail_x))
    y = max(0, min(y, avail_y))

    frame_bgr = overlay_rgba(frame_bgr, cached, x, y)
    return frame_bgr


def apply_frame_and_logo_in_rect(canvas_bgr, x: int, y: int, w: int, h: int):
    """
    Apply the overlays only inside a rectangle of the canvas (useful when the preview is letterboxed).
    """
    Hc, Wc = canvas_bgr.shape[:2]
    x = max(0, min(int(x), Wc - 1))
    y = max(0, min(int(y), Hc - 1))
    w = max(1, min(int(w), Wc - x))
    h = max(1, min(int(h), Hc - y))

    sub = canvas_bgr[y:y+h, x:x+w]
    sub2 = apply_frame_and_logo(sub)
    canvas_bgr[y:y+h, x:x+w] = sub2
    return canvas_bgr


# =========================================================
# GOOGLE DRIVE UPLOADER
# =========================================================
class GoogleDriveUploaderOAuth:
    """
    Upload vers Google Drive via OAuth "installed app" (compte Google perso).
    1) utilise oauth_client.json (le fichier téléchargé "installed")
    2) génère token.json au premier lancement (ouvre un navigateur)
    """

    def __init__(self, oauth_client_json: str, token_json: str = "token.json", folder_id: Optional[str] = None):
        self.oauth_client_json = oauth_client_json
        self.token_json = token_json
        self.folder_id = folder_id

        self._drive = None
        self._init_drive()

    def _init_drive(self):
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
        except Exception as e:
            raise RuntimeError(
                "Librairies Google manquantes. Installe:\n"
                "  pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2\n"
                f"Détail: {e}"
            )

        if not os.path.isfile(self.oauth_client_json):
            raise FileNotFoundError(f"OAuth client JSON introuvable: {self.oauth_client_json}")

        scopes = ["https://www.googleapis.com/auth/drive.file"]

        creds = None
        if os.path.exists(self.token_json):
            creds = Credentials.from_authorized_user_file(self.token_json, scopes=scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.oauth_client_json, scopes=scopes)
                creds = flow.run_local_server(port=0)
            with open(self.token_json, "w", encoding="utf-8") as f:
                f.write(creds.to_json())

        self._drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    def upload_and_get_url(self, filepath: str) -> str:
        from googleapiclient.http import MediaFileUpload

        filename = os.path.basename(filepath)
        metadata = {"name": filename}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]

        media = MediaFileUpload(filepath, mimetype="image/jpeg", resumable=True)
        created = self._drive.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]

        # lien consultable
        info = self._drive.files().get(fileId=file_id, fields="webViewLink").execute()
        return info["webViewLink"]

class GoogleDriveUploader:
    def __init__(
        self,
        service_account_json: str,
        folder_id: Optional[str] = None,
        make_public: bool = True,
        enable_shortener: bool = False,
        shortener_backend: str = "tinyurl",
    ):
        self.service_account_json = service_account_json
        self.folder_id = folder_id
        self.make_public = make_public

        self.enable_shortener = enable_shortener
        self.shortener_backend = shortener_backend
        self._shortener = None
        self._drive = None

        self._init_drive()
        self._init_shortener()

    def _init_drive(self):
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except Exception as e:
            raise RuntimeError(
                "Librairies Google manquantes. Installe:\n"
                "  pip install google-api-python-client google-auth-httplib2 google-auth\n"
                f"Détail: {e}"
            )

        if not os.path.isfile(self.service_account_json):
            raise FileNotFoundError(f"Service account JSON introuvable: {self.service_account_json}")

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        creds = service_account.Credentials.from_service_account_file(self.service_account_json, scopes=scopes)
        self._drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _init_shortener(self):
        if not self.enable_shortener:
            return
        try:
            import pyshorteners
            self._shortener = pyshorteners.Shortener()
        except Exception:
            self._shortener = None

    def upload_and_get_url(self, filepath: str) -> str:
        from googleapiclient.http import MediaFileUpload

        filename = os.path.basename(filepath)
        metadata = {"name": filename}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]

        media = MediaFileUpload(filepath, mimetype="image/jpeg", resumable=True)
        created = self._drive.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]

        if self.make_public:
            try:
                self._drive.permissions().create(
                    fileId=file_id,
                    body={"type": "anyone", "role": "reader"},
                    fields="id",
                ).execute()
            except Exception:
                pass

        info = self._drive.files().get(fileId=file_id, fields="webViewLink,webContentLink").execute()
        url = info.get("webViewLink") or info.get("webContentLink") or f"https://drive.google.com/file/d/{file_id}/view"

        if self._shortener is not None:
            try:
                short_fn = getattr(self._shortener, self.shortener_backend).short
                url = short_fn(url)
            except Exception:
                pass

        return url


class GoogleSheetsLogger:
    """Petit logger Google Sheets (append une ligne par évènement).

    Points importants:
    - spreadsheet_id doit être l'ID (pas l'URL entière). Si tu passes l'URL, on extrait l'ID automatiquement.
    - worksheet_name doit exister; si l'onglet n'existe pas, on le crée.
    - Si le nom d'onglet contient des espaces, on met des quotes A1 ('Feuille 1'!A1).
    """

    def __init__(self, service_account_json: str, spreadsheet_id: str, worksheet_name: str = "logs"):
        self.service_account_json = service_account_json
        self.spreadsheet_id = self._normalize_spreadsheet_id(spreadsheet_id)
        self.worksheet_name = (worksheet_name or "logs").strip()
        self._svc = None
        self._init_sheets()

    @staticmethod
    def _normalize_spreadsheet_id(value: str) -> str:
        if not value:
            return value
        v = str(value).strip()
        # Si l'utilisateur colle l'URL complète, on extrait l'ID entre /d/ et /edit
        if "docs.google.com" in v and "/spreadsheets/d/" in v:
            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", v)
            if m:
                return m.group(1)
        return v

    def _a1_range(self) -> str:
        title = self.worksheet_name
        # A1 notation: si le nom d'onglet a des espaces/symboles, il faut des quotes.
        if any(ch in title for ch in [" ", "!", ":", "'"]):
            title = title.replace("'", "''")
            return f"'{title}'!A1"
        return f"{title}!A1"

    def _init_sheets(self):
        try:
            from googleapiclient.discovery import build
        except Exception as e:
            raise RuntimeError(
                "Librairies Google manquantes pour Sheets. Installe:\n"
                "  pip install google-api-python-client google-auth-httplib2 google-auth\n"
                f"Détail: {e}"
            )

        if not os.path.isfile(self.service_account_json):
            raise FileNotFoundError(f"Service account JSON introuvable: {self.service_account_json}")

        if not self.spreadsheet_id:
            raise ValueError("Spreadsheet ID vide. Mets uniquement l'ID (entre /d/ et /edit).")

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = service_account.Credentials.from_service_account_file(self.service_account_json, scopes=scopes)
        self._svc = build("sheets", "v4", credentials=creds, cache_discovery=False)

        # Vérifie / crée l'onglet
        self._ensure_worksheet_exists()

    def _ensure_worksheet_exists(self):
        """Crée l'onglet worksheet_name s'il n'existe pas."""
        from googleapiclient.errors import HttpError

        try:
            meta = self._svc.spreadsheets().get(spreadsheetId=self.spreadsheet_id).execute()
            sheets = meta.get("sheets", []) or []
            titles = {s.get("properties", {}).get("title") for s in sheets}
            if self.worksheet_name not in titles:
                req = {
                    "requests": [
                        {"addSheet": {"properties": {"title": self.worksheet_name}}}
                    ]
                }
                self._svc.spreadsheets().batchUpdate(spreadsheetId=self.spreadsheet_id, body=req).execute()
        except HttpError as e:
            # On remonte un message plus lisible (ça aide énormément pour diagnostiquer)
            detail = ""
            try:
                detail = e.content.decode("utf-8", errors="ignore")
            except Exception:
                detail = str(e)
            raise RuntimeError(
                "Google Sheets: impossible de lire/créer l'onglet. Vérifie:\n"
                "- Spreadsheet ID correct (pas l'URL)\n"
                "- le Sheet est partagé avec l'email du service account (Editor)\n"
                "- Google Sheets API activée dans Google Cloud\n"
                f"\nDétail HTTP: {detail}"
            ) from e

    def append_row(self, values: list):
        """Ajoute une ligne dans l'onglet worksheet_name."""
        from googleapiclient.errors import HttpError

        body = {"values": [values]}
        rng = self._a1_range()

        try:
            self._svc.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=rng,
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body,
            ).execute()
        except HttpError as e:
            detail = ""
            try:
                detail = e.content.decode("utf-8", errors="ignore")
            except Exception:
                detail = str(e)
            raise RuntimeError(
                f"Google Sheets append error (range={rng}). "
                f"Vérifie que l'onglet existe et que le nom est exact. Détail HTTP: {detail}"
            ) from e


class CaptureStorage:
    """Capture une frame, l'upload sur Drive, puis supprime le fichier local.

    Note: il y a forcément un fichier temporaire sur disque car googleapiclient MediaFileUpload
    attend un chemin local. Le fichier est supprimé immédiatement après upload.
    """

    def __init__(self, uploader: GoogleDriveUploader):
        self.uploader = uploader

    def save_frame_and_upload(self, frame_bgr) -> tuple[str, str]:
        # microseconds -> évite collisions et garantit une URL/QR par photo
        ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")
        filename = f"capture_{ts}.jpg"

        tmp_path = os.path.join(tempfile.gettempdir(), filename)

        ok = cv2.imwrite(tmp_path, frame_bgr)
        if not ok:
            raise IOError(f"Impossible d'écrire l'image temporaire: {tmp_path}")

        try:
            url = self.uploader.upload_and_get_url(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return filename, url

    def close(self):
        pass

# =========================================================
# URL -> QR code
# =========================================================
def make_qr_image(url: str, size: int = 260) -> Image.Image:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size))


# =========================================================
# ROI mean
# =========================================================
def get_roi_mean_bgr(preview_bgr, roi_x=ROI_X, roi_y=ROI_Y, roi_w=ROI_W, roi_h=ROI_H) -> np.ndarray:
    H, W = preview_bgr.shape[:2]
    x1 = max(0, min(W - 1, int(roi_x)))
    y1 = max(0, min(H - 1, int(roi_y)))
    x2 = max(0, min(W, x1 + int(roi_w)))
    y2 = max(0, min(H, y1 + int(roi_h)))

    roi = preview_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return roi.reshape(-1, 3).mean(axis=0).astype(np.float32)


# =========================================================
# APP
# =========================================================
class CameraAppGUI:
    """
    Full-screen Tkinter app:
    - Live camera preview (scaled to window, keeps aspect ratio)
    - ROI trigger (ROI defined in CAMERA_RESOLUTION coordinates, auto-mapped to preview)
    - Countdown + flash + capture
    - Upload capture to Google Drive, generate QR, show QR history strip
    """

    def __init__(self):
        self._logger = logging.getLogger('camera_qr_google_drive')

        # -----------------------------
        # Runtime state / thread safety
        # -----------------------------
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._last_frame_bgr = None
        self._running = True
        self._sequence_running = False

        # UI-safe queue (call Tk from the Tk thread only)
        self._uiq = queue.Queue()

        # UX capture
        self._countdown_text = None
        self._flash_until = 0.0
        self._flash_color = FLASH_COLOR

        # Baseline for ROI trigger
        self._baseline_start = time.monotonic()
        self._baseline_samples = []
        self._baseline_mean = None

        # Hold timer for ROI trigger
        self._roi_active_since = None

        # Cooldown / disable ROI after capture
        self._cooldown_until = 0.0
        self._roi_disabled_until = 0.0

        # Layout (dynamic)
        # preview_w/preview_h = processing size (keep small for speed)
        self.preview_w = PREVIEW_W
        self.preview_h = PREVIEW_H
        # display_w/display_h = actual on-screen area (fullscreen/window)
        self.display_w = PREVIEW_W
        self.display_h = PREVIEW_H
        self.qr_size = int(QR_FIXED_SIZE_PX)
        self.qr_gap = QR_GAP
        self.qr_canvas_h = self.qr_size + 2 * int(QR_GAP)
        self._layout_pending = False
        self._last_layout = None

        # QR history UI state
        self._qr_items = []
        self._qr_animating = False

        # -----------------------------
        # Init camera and services
        # -----------------------------
        self._init_camera()
        self._init_services()

        # -----------------------------
        # Tkinter UI
        # -----------------------------
        self.root = tk.Tk()
        self.root.title("Camera → Drive → QR")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Fullscreen by default + Escape toggles
        self._is_fullscreen = True
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.toggle_fullscreen())

        # Main container (no padding, fullscreen)
        main = ttk.Frame(self.root, padding=0)
        main.pack(fill="both", expand=True)

        # Preview fills all space above QR strip
        self.preview_label = ttk.Label(main)
        self.preview_label.pack(fill="both", expand=True)

        # QR strip (bottom)
        self.qr_canvas = tk.Canvas(main, highlightthickness=0, bg=QR_BAR_BG)
        self.qr_canvas.pack(side="bottom", fill="x")

        self._tk_preview_img = None

        # Recompute layout when window size changes (debounced)
        self.root.bind("<Configure>", self._on_configure)

        # loops
        self.root.after(10, self.update_preview)
        self.root.after(25, self._process_ui_queue)

    # =====================================================
    # Init helpers
    # =====================================================
    def _init_camera(self):
        """Open the camera with a platform-appropriate backend."""
        backend = None
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        else:
            backend = getattr(cv2, "CAP_V4L2", None)

        if backend is None:
            self.cap = cv2.VideoCapture(CAM_INDEX)
        else:
            self.cap = cv2.VideoCapture(CAM_INDEX, backend)

        # Keep latency low when supported
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam. Change CAM_INDEX (0/1/2).")

        # Request camera resolution (may be adjusted by driver)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] Requested: {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]} | Actual: {actual_w}x{actual_h}")

    def _init_services(self):
        """Init Google Drive uploader + optional Google Sheets logger."""
        # Google uploader + storage
        self.storage = None
        try:
            uploader = GoogleDriveUploader(
                service_account_json=GOOGLE_SERVICE_ACCOUNT_JSON,
                folder_id=GOOGLE_DRIVE_FOLDER_ID,
                make_public=GOOGLE_DRIVE_MAKE_PUBLIC,
                enable_shortener=ENABLE_URL_SHORTENER,
                shortener_backend=SHORTENER_BACKEND,
            )
            self.storage = CaptureStorage(uploader)
            self._logger.info("Google Drive uploader initialisé.")
        except Exception:
            self._logger.exception("Google Drive init failed")
            self.storage = None

        # Google Sheets logger (optional)
        self.sheets_logger = None
        if ENABLE_SHEETS_LOG:
            sheet_id = GOOGLE_SHEETS_SPREADSHEET_ID
            if not sheet_id:
                sheet_id = _read_first_nonempty_line(GOOGLE_SHEETS_SPREADSHEET_ID_FILE)
                if sheet_id:
                    self._logger.info("Spreadsheet ID lu depuis %s", GOOGLE_SHEETS_SPREADSHEET_ID_FILE)

            sheet_id = extract_spreadsheet_id(sheet_id)

            if sheet_id:
                try:
                    self.sheets_logger = GoogleSheetsLogger(
                        service_account_json=GOOGLE_SERVICE_ACCOUNT_JSON,
                        spreadsheet_id=sheet_id,
                        worksheet_name=GOOGLE_SHEETS_WORKSHEET_NAME,
                    )
                    self._logger.info("Google Sheets logger initialisé.")
                except Exception:
                    self._logger.exception("Google Sheets init failed")
                    self.sheets_logger = None
            else:
                self._logger.warning(
                    "Google Sheets log activé mais aucun Sheet ID trouvé (ni GOOGLE_SHEETS_SPREADSHEET_ID ni %s).",
                    GOOGLE_SHEETS_SPREADSHEET_ID_FILE,
                )

    # =====================================================
    # Fullscreen toggle
    # =====================================================
    def toggle_fullscreen(self):
        """Toggle fullscreen on/off (Escape key)."""
        self._is_fullscreen = not self._is_fullscreen
        self.root.attributes("-fullscreen", self._is_fullscreen)

    # =====================================================
    # UI queue
    # =====================================================
    def ui(self, fn):
        """Schedule a UI update from a worker thread."""
        self._uiq.put(fn)

    def _process_ui_queue(self):
        if not self._running:
            return
        try:
            while True:
                fn = self._uiq.get_nowait()
                try:
                    fn()
                except Exception:
                    self._logger.debug("UI callback failed", exc_info=True)
        except queue.Empty:
            pass
        self.root.after(25, self._process_ui_queue)

    # =====================================================
    # Layout / resize handling
    # =====================================================
    def _on_configure(self, event):
        """Debounce layout recomputation on window resize."""
        if self._layout_pending:
            return
        self._layout_pending = True
        self.root.after(120, self._apply_layout)

    def _apply_layout(self):
        """Compute preview and QR sizes from current window size."""
        self._layout_pending = False
        if not self._running:
            return

        win_w = max(320, int(self.root.winfo_width()))
        win_h = max(240, int(self.root.winfo_height()))

        # QR size adapts to window width (fits QR_HISTORY items + gaps) OR can be fixed from constants
        gap = int(self.qr_gap)
        if str(QR_SIZE_MODE).lower() == 'fixed':
            qr_size = int(QR_FIXED_SIZE_PX)
        else:
            max_qr = int((win_w - (QR_HISTORY + 1) * gap) / max(1, QR_HISTORY))
            qr_size = int(max(QR_SIZE_MIN, min(QR_SIZE_MAX, max_qr)))

        qr_canvas_h = qr_size + 2 * gap
        preview_h = max(200, win_h - qr_canvas_h)

        layout = (win_w, win_h, qr_size, qr_canvas_h, preview_h)
        if self._last_layout == layout:
            return
        self._last_layout = layout

        # Apply new layout
        self.qr_size = qr_size
        self.qr_canvas_h = qr_canvas_h
        # Display area (fullscreen/window): this is how large we *render* the preview.
        self.display_w = win_w
        self.display_h = preview_h
        # Processing preview size: keep it small for speed (especially on Raspberry Pi).
        # ROI + detection are computed on this size, then we scale/letterbox for display.
        self.preview_w = PREVIEW_W
        self.preview_h = PREVIEW_H

        self.qr_canvas.config(width=win_w, height=qr_canvas_h, bg=QR_BAR_BG)

        # If size changed, reset the QR strip to avoid stretched items
        self._reset_qr_strip()

    def _reset_qr_strip(self):
        """Clear QR strip and history (called on resize)."""
        try:
            self.qr_canvas.delete("all")
        except Exception:
            pass
        self._qr_items.clear()
        self._qr_animating = False

    # =====================================================
    # QR strip helpers (dynamic sizing)
    # =====================================================
    def _qr_target_centers(self):
        """Return the target X centers for each QR slot + common Y center."""
        w = max(1, int(self.qr_canvas.winfo_width() or self.preview_w))
        gap = int(self.qr_gap)
        size = int(self.qr_size)

        # Placement of the strip: center/left/right + optional margin
        total = QR_HISTORY * size + (QR_HISTORY + 1) * gap
        align = str(QR_STRIP_ALIGN).lower()
        margin = int(QR_STRIP_MARGIN_PX)
        if align == 'left':
            left = max(0, margin)
        elif align == 'right':
            left = max(0, int(w - total - margin))
        else:
            # center
            left = max(0, int((w - total) / 2) + margin)

        centers = []
        for i in range(QR_HISTORY):
            x_left = left + gap + i * (size + gap)
            centers.append(x_left + size // 2)

        cy = gap + size // 2
        return centers, cy

    def _animate_qr_to_targets(self, start_positions, target_positions, steps=QR_ANIM_STEPS, delay=QR_ANIM_DELAY_MS, on_done=None):
        """Smoothly animate QR items to their new positions."""
        self._qr_animating = True

        def step(k):
            t = (k / steps)
            _, cy = self._qr_target_centers()
            for idx, item in enumerate(self._qr_items):
                x0 = start_positions[idx]
                x1 = target_positions[idx]
                x = x0 + (x1 - x0) * t
                self.qr_canvas.coords(item["id"], x, cy)

            if k < steps:
                self.root.after(delay, lambda: step(k + 1))
            else:
                self._qr_animating = False
                if on_done:
                    on_done()

        step(0)

    def push_qr_to_history(self, qr_pil_img: Image.Image):
        """Insert a new QR in the strip (animated), keeping last QR_HISTORY."""
        size = int(self.qr_size)
        gap = int(self.qr_gap)

        qr_pil_img = qr_pil_img.resize((size, size))
        qr_tk = ImageTk.PhotoImage(qr_pil_img)

        centers, cy = self._qr_target_centers()
        start_x_new = -size // 2
        new_id = self.qr_canvas.create_image(start_x_new, cy, image=qr_tk)
        self._qr_items.insert(0, {"id": new_id, "img": qr_tk})

        to_remove = None
        if len(self._qr_items) > QR_HISTORY:
            to_remove = self._qr_items[-1]

        start_positions = [self.qr_canvas.coords(it["id"])[0] for it in self._qr_items]
        target_positions = []
        for i in range(len(self._qr_items)):
            if i < QR_HISTORY:
                target_positions.append(centers[i])
            else:
                # move out to the right before deleting
                target_positions.append(self.preview_w + size)

        def cleanup():
            nonlocal to_remove
            if to_remove is not None:
                try:
                    self.qr_canvas.delete(to_remove["id"])
                except Exception:
                    pass
                try:
                    self._qr_items.remove(to_remove)
                except ValueError:
                    pass

            # Snap remaining to targets
            centers2, cy2 = self._qr_target_centers()
            for i, it in enumerate(self._qr_items[:QR_HISTORY]):
                self.qr_canvas.coords(it["id"], centers2[i], cy2)

        if self._qr_animating:
            cleanup()
            return

        self._animate_qr_to_targets(start_positions, target_positions, on_done=cleanup)

    # =====================================================
    # ROI mapping (camera -> preview)
    # =====================================================
    def _roi_cam_to_preview(self, frame_w: int, frame_h: int):
        """
        ROI_* values are defined in CAMERA coordinates (FRAME_WIDTH/FRAME_HEIGHT).
        This maps them to the current preview size (preview_w/preview_h).
        """
        if frame_w <= 0 or frame_h <= 0:
            return 0, 0, 1, 1

        sx = self.preview_w / frame_w
        sy = self.preview_h / frame_h

        # Clamp ROI in camera coordinates, then scale to preview coordinates
        rx = max(0, min(frame_w - 1, int(ROI_X)))
        ry = max(0, min(frame_h - 1, int(ROI_Y)))
        rw = max(1, min(frame_w - rx, int(ROI_W)))
        rh = max(1, min(frame_h - ry, int(ROI_H)))

        x = int(rx * sx)
        y = int(ry * sy)
        w = max(1, int(rw * sx))
        h = max(1, int(rh * sy))
        return x, y, w, h

    # =====================================================
    # Preview loop
    # =====================================================
    def update_preview(self):
        """Main preview loop (Tk thread)."""
        if not self._running:
            return

        ok, frame = self.cap.read()
        if ok and frame is not None:
            frame_h, frame_w = frame.shape[:2]

            with self._frame_lock:
                self._last_frame_bgr = frame

            # Resize frame to current preview size
            interp = cv2.INTER_AREA if (self.preview_w <= frame_w and self.preview_h <= frame_h) else cv2.INTER_LINEAR
            small = cv2.resize(frame, (self.preview_w, self.preview_h), interpolation=interp)

            if FLIP_PREVIEW:
                small = apply_flip(small, FLIP_MODE)

            now = time.monotonic()

            # Map ROI from camera coordinates to preview coordinates
            roi_x, roi_y, roi_w, roi_h = self._roi_cam_to_preview(frame_w, frame_h)
            roi_mean = get_roi_mean_bgr(small, roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h)

            # Baseline collection (silent)
            if self._baseline_mean is None:
                elapsed = now - self._baseline_start
                self._baseline_samples.append(roi_mean)
                if elapsed >= BASELINE_SECONDS and len(self._baseline_samples) > 0:
                    arr = np.stack(self._baseline_samples, axis=0)
                    self._baseline_mean = arr.mean(axis=0)
            else:
                roi_enabled = (
                    (now >= self._cooldown_until) and
                    (not self._sequence_running) and
                    (now >= self._roi_disabled_until)
                )

                if not roi_enabled:
                    self._roi_active_since = None
                else:
                    dist = float(np.linalg.norm(roi_mean - self._baseline_mean))
                    if dist >= TRIGGER_DIST_THRESHOLD:
                        if self._roi_active_since is None:
                            self._roi_active_since = now

                        held = now - self._roi_active_since
                        if held >= HOLD_SECONDS:
                            self._roi_active_since = None
                            self._cooldown_until = now + COOLDOWN_SECONDS
                            self.start_countdown_then_capture(COUNTDOWN_SECONDS)
                    else:
                        self._roi_active_since = None

            # ROI rectangle: green when active, red otherwise
            if DRAW_ROI_RECT:
                roi_is_active = (
                    self._baseline_mean is not None
                    and now >= self._roi_disabled_until
                    and not self._sequence_running
                    and now >= self._cooldown_until
                )
                color = (0, 255, 0) if roi_is_active else (0, 0, 255)
                x1 = int(roi_x)
                y1 = int(roi_y)
                x2 = int(roi_x + roi_w)
                y2 = int(roi_y + roi_h)
                cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)

            # flash overlay
            if now < self._flash_until:
                small[:] = 0 if self._flash_color.lower() == "black" else 255

            # countdown overlay
            if self._countdown_text is not None:
                text = str(self._countdown_text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(1.0, min(self.preview_w, self.preview_h) / 250.0)
                thickness = max(2, int(scale * 2.5))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                x = int((self.preview_w - tw) / 2)
                y = int((self.preview_h + th) / 2)
                cv2.putText(small, text, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
                cv2.putText(small, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            # Optional live preview overlays (decorative frame/logo)
            if PREVIEW_APPLY_OVERLAYS:
                small = apply_frame_and_logo(small)

            # Letterbox into the on-screen area while keeping aspect ratio
            disp_w = max(1, int(self.display_w))
            disp_h = max(1, int(self.display_h))
            src_h, src_w = small.shape[:2]
            scale = min(disp_w / max(1, src_w), disp_h / max(1, src_h))
            fit_w = max(1, int(src_w * scale))
            fit_h = max(1, int(src_h * scale))
            # Anchor positioning inside the display area
            margin = int(CAMERA_ANCHOR_MARGIN_PX)
            anchor = str(CAMERA_ANCHOR).lower()
            if anchor == 'tl':
                x0, y0 = margin, margin
            elif anchor == 'tr':
                x0, y0 = disp_w - fit_w - margin, margin
            elif anchor == 'bl':
                x0, y0 = margin, disp_h - fit_h - margin
            elif anchor == 'br':
                x0, y0 = disp_w - fit_w - margin, disp_h - fit_h - margin
            else:
                # center
                x0, y0 = (disp_w - fit_w) // 2, (disp_h - fit_h) // 2

            display_bgr = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            display_bgr[:] = np.array(LETTERBOX_BG_BGR, dtype=np.uint8)
            interp2 = cv2.INTER_AREA if (fit_w <= src_w and fit_h <= src_h) else cv2.INTER_LINEAR
            resized = cv2.resize(small, (fit_w, fit_h), interpolation=interp2)
            display_bgr[y0:y0+fit_h, x0:x0+fit_w] = resized

            # Convert for Tkinter display
            rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self._tk_preview_img = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self._tk_preview_img)

        # ~30 FPS
        self.root.after(33, self.update_preview)

    # =====================================================
    # Countdown + flash + capture
    # =====================================================
    def start_countdown_then_capture(self, seconds: int = 3):
        """Run a countdown, flash, then capture in a worker thread."""
        if self._sequence_running:
            return
        if not self._capture_lock.acquire(blocking=False):
            return

        self._sequence_running = True

        def show_number(n: int):
            self._countdown_text = str(n)

        def clear_countdown():
            self._countdown_text = None

        def do_flash():
            clear_countdown()
            self._flash_until = time.monotonic() + FLASH_DURATION_S

        def start_capture_thread():
            threading.Thread(target=self.capture_flow, daemon=True).start()

        for i in range(seconds, 0, -1):
            delay_ms = (seconds - i) * 1000
            self.root.after(delay_ms, lambda n=i: show_number(n))

        self.root.after(seconds * 1000, do_flash)
        self.root.after(int(seconds * 1000 + FLASH_DURATION_S * 1000), start_capture_thread)

    def capture_flow(self):
        """Capture the last camera frame, burn overlays into the SAVED photo, upload to Drive, generate QR, update UI.

        Note: the LIVE preview stays clean; the decorative overlays (frame/logo) are applied ONLY to the photo that is saved/uploaded.
        """
        try:
            with self._frame_lock:
                frame = None if self._last_frame_bgr is None else self._last_frame_bgr.copy()
            if frame is None:
                return

            if FLIP_CAPTURE:
                frame = apply_flip(frame, FLIP_MODE)
            # IMPORTANT: burn overlays (frame/logo) into the SAVED/UPLOADED photo.
            # The LIVE preview remains clean.
            if CAPTURE_APPLY_OVERLAYS:
                frame = apply_frame_and_logo(frame)

            if self.storage is None:
                self._logger.error("Capture demandée mais Google Drive n'est pas initialisé (storage=None).")
                return

            try:
                local_path, url = self.storage.save_frame_and_upload(frame)
            except Exception:
                self._logger.exception("Upload vers Drive a échoué")
                if self.sheets_logger is not None:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "UPLOAD_ERROR", "", "", "exception"])
                    except Exception:
                        self._logger.exception("Sheets append_row failed (UPLOAD_ERROR)")
                return

            if not url:
                self._logger.error("Upload OK mais URL vide (url='').")
                if self.sheets_logger is not None:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "URL_EMPTY", local_path, "", ""])
                    except Exception:
                        self._logger.exception("Sheets append_row failed (URL_EMPTY)")
                return

            self._logger.info("URL reçue: %s", url)
            if self.sheets_logger is not None:
                try:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.sheets_logger.append_row([ts, "UPLOAD_OK", local_path, url, ""])
                except Exception:
                    self._logger.exception("Sheets append_row failed (UPLOAD_OK)")

            # Generate QR with the current UI QR size (keeps it readable)
            try:
                qr_img = make_qr_image(url, size=int(self.qr_size))
            except Exception:
                self._logger.exception("Conversion URL -> QR a échoué")
                if self.sheets_logger is not None and SHEETS_LOG_QR_EVENTS:
                    try:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.sheets_logger.append_row([ts, "QR_ERROR", local_path, url, "exception"])
                    except Exception:
                        self._logger.exception("Sheets append_row failed (QR_ERROR)")
                return

            if self.sheets_logger is not None and SHEETS_LOG_QR_EVENTS:
                try:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.sheets_logger.append_row([ts, "QR_OK", local_path, url, ""])
                except Exception:
                    self._logger.exception("Sheets append_row failed (QR_OK)")

            # Disable ROI temporarily after capture (avoid retrigger)
            self._roi_disabled_until = time.monotonic() + ROI_DISABLE_AFTER_CAPTURE_S

            # Push QR to history (UI thread)
            self.ui(lambda: self.push_qr_to_history(qr_img))

        finally:
            self._sequence_running = False
            try:
                self._capture_lock.release()
            except Exception:
                pass

    # =====================================================
    # Shutdown
    # =====================================================
    def on_close(self):
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            if self.storage:
                self.storage.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self._logger.info("Entrée dans mainloop (la fenêtre doit rester ouverte).")
        self.root.mainloop()
        self._logger.info("Sortie de mainloop (fenêtre fermée).")

if __name__ == "__main__":
    setup_logging()
    logging.getLogger("camera_qr_google_drive").info("Démarrage de l'application Camera → Drive → QR")
    CameraAppGUI().run()