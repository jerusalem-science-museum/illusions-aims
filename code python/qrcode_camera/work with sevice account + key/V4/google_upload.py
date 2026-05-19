import os
import datetime
import re
import tempfile
from typing import Optional
import cv2
from google.oauth2 import service_account
import pyshorteners


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
            print(f"[ERROR] Missing Google libraries: {e}")
            raise RuntimeError(
                "Missing Google libraries. Install:\n"
                "  pip install google-api-python-client google-auth-httplib2 google-auth"
            ) from e

        if not os.path.isfile(self.service_account_json):
            print(f"[ERROR] Service account JSON file not found: {self.service_account_json}")
            raise FileNotFoundError(
                f"Service account JSON file not found: {self.service_account_json}"
            )

        try:
            scopes = ["https://www.googleapis.com/auth/drive.file"]
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_json,
                scopes=scopes
            )
            self._drive = build("drive", "v3", credentials=creds, cache_discovery=False)
            print("[INFO] Google Drive initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Google Drive with the provided key: {e}")
            raise RuntimeError(
                "Failed to initialize Google Drive. "
                "Check that the service account JSON is valid and that the Drive API is enabled."
            ) from e

    def _init_shortener(self):
        if not self.enable_shortener:
            return
        try:
            self._shortener = pyshorteners.Shortener()
            print("[INFO] URL shortener initialized.")
        except Exception as e:
            print(f"[WARNING] Failed to initialize URL shortener: {e}")
            self._shortener = None

    def upload_and_get_url(self, filepath: str) -> str:
        from googleapiclient.http import MediaFileUpload

        if self._drive is None:
            print("[ERROR] Google Drive is not initialized.")
            raise RuntimeError("Google Drive is not initialized.")

        filename = os.path.basename(filepath)
        metadata = {"name": filename}
        if self.folder_id:
            metadata["parents"] = [self.folder_id]

        try:
            media = MediaFileUpload(filepath, mimetype="image/jpeg", resumable=True)
            created = self._drive.files().create(
                body=metadata,
                media_body=media,
                fields="id"
            ).execute()
            file_id = created["id"]
            print(f"[INFO] File uploaded to Drive. file_id={file_id}")
        except Exception as e:
            print(f"[ERROR] Google Drive upload failed for {filepath}: {e}")
            raise RuntimeError(f"Google Drive upload failed: {e}") from e

        if self.make_public:
            try:
                self._drive.permissions().create(
                    fileId=file_id,
                    body={"type": "anyone", "role": "reader"},
                    fields="id",
                ).execute()
                print("[INFO] Public permission added.")
            except Exception as e:
                print(f"[WARNING] Failed to make file public: {e}")

        try:
            info = self._drive.files().get(
                fileId=file_id,
                fields="webViewLink,webContentLink"
            ).execute()

            url = (
                info.get("webViewLink")
                or info.get("webContentLink")
                or f"https://drive.google.com/file/d/{file_id}/view"
            )
        except Exception as e:
            print(f"[WARNING] Failed to retrieve webViewLink/webContentLink: {e}")
            url = f"https://drive.google.com/file/d/{file_id}/view"

        if self._shortener is not None:
            try:
                short_fn = getattr(self._shortener, self.shortener_backend).short
                url = short_fn(url)
            except Exception as e:
                print(f"[WARNING] URL shortener failed: {e}")

        return url


class GoogleSheetsLogger:
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
        if "docs.google.com" in v and "/spreadsheets/d/" in v:
            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", v)
            if m:
                return m.group(1)
        return v

    def _a1_range(self) -> str:
        title = self.worksheet_name
        if any(ch in title for ch in [" ", "!", ":", "'"]):
            title = title.replace("'", "''")
            return f"'{title}'!A1"
        return f"{title}!A1"

    def _init_sheets(self):
        try:
            from googleapiclient.discovery import build
        except Exception as e:
            print(f"[ERROR] Missing Google Sheets libraries: {e}")
            raise RuntimeError(
                "Missing Google libraries for Sheets. Install:\n"
                "  pip install google-api-python-client google-auth-httplib2 google-auth"
            ) from e

        if not os.path.isfile(self.service_account_json):
            print(f"[ERROR] Service account JSON file not found: {self.service_account_json}")
            raise FileNotFoundError(
                f"Service account JSON file not found: {self.service_account_json}"
            )

        if not self.spreadsheet_id:
            print("[ERROR] Spreadsheet ID is empty.")
            raise ValueError("Spreadsheet ID is empty. Use only the ID between /d/ and /edit.")

        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_json,
                scopes=scopes
            )
            self._svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
            print("[INFO] Google Sheets initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Google Sheets with the provided key: {e}")
            raise RuntimeError(
                "Failed to initialize Google Sheets. "
                "Check that the service account JSON is valid and that the Sheets API is enabled."
            ) from e

        self._ensure_worksheet_exists()

    def _ensure_worksheet_exists(self):
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
                self._svc.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=req
                ).execute()
                print(f"[INFO] Worksheet created: {self.worksheet_name}")
        except HttpError as e:
            detail = ""
            try:
                detail = e.content.decode("utf-8", errors="ignore")
            except Exception:
                detail = str(e)

            print(f"[ERROR] Failed to read/create Google Sheets worksheet: {detail}")
            raise RuntimeError(
                "Google Sheets: failed to read/create worksheet.\n"
                "- Check spreadsheet ID\n"
                "- Share the spreadsheet with the service account\n"
                "- Make sure Google Sheets API is enabled\n"
                f"HTTP detail: {detail}"
            ) from e

    def append_row(self, values: list):
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
            print(f"[INFO] Row added to Google Sheets: {values}")
        except HttpError as e:
            detail = ""
            try:
                detail = e.content.decode("utf-8", errors="ignore")
            except Exception:
                detail = str(e)

            print(f"[ERROR] Google Sheets append error: {detail}")
            raise RuntimeError(
                f"Google Sheets append error (range={rng}). HTTP detail: {detail}"
            ) from e


class CaptureStorage:
    def __init__(self, uploader: GoogleDriveUploader):
        self.uploader = uploader

    def save_frame_and_upload(self, frame_bgr) -> tuple[str, str]:
        ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S__%f")
        filename = f"capture_{ts}.jpg"
        tmp_path = os.path.join(tempfile.gettempdir(), filename)

        ok = cv2.imwrite(tmp_path, frame_bgr)
        if not ok:
            print(f"[ERROR] Failed to write temporary image: {tmp_path}")
            raise IOError(f"Failed to write temporary image: {tmp_path}")

        try:
            url = self.uploader.upload_and_get_url(tmp_path)
        except Exception as e:
            print(f"[ERROR] Image upload failed: {e}")
            raise
        finally:
            try:
                os.remove(tmp_path)
                print(f"[INFO] Temporary file deleted: {tmp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to delete temporary file: {e}")

        return filename, url

    def close(self):
        pass


def _read_first_nonempty_line(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                return s
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return ''
    except Exception as e:
        print(f"[ERROR] Failed to read file ({path}): {e}")
        return ''
    return ''


def extract_spreadsheet_id(value: str) -> str:
    if not value:
        return ''
    v = value.strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", v)
    if m:
        return m.group(1)
    return v