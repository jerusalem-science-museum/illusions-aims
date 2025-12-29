import os
import datetime
import re
import tempfile
from typing import Optional
import cv2
from google.oauth2 import service_account


# =========================================================
# GOOGLE DRIVE UPLOADER
# =========================================================
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
