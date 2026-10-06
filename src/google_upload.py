"""Google Drive upload and Google Sheets logging through a service account.

Google libraries are imported lazily so a missing package only disables these features.
"""
import io
import queue
import re
import threading
import time

from log import get_logger

log = get_logger()

# Socket timeout for every Google call. httplib2's default is "none", so a stalled
# connection would hold the client lock forever. Must exceed a real photo upload.
HTTP_TIMEOUT_S = 30


def _build_service(json_path: str, scope: str, name: str, version: str):
    import httplib2
    from google.oauth2 import service_account
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(json_path, scopes=[scope])
    http = AuthorizedHttp(creds, http=httplib2.Http(timeout=HTTP_TIMEOUT_S))
    return build(name, version, http=http, cache_discovery=False)


class DriveUploader:
    def __init__(self, json_path: str, folder_id=None, make_public=True, shortener_backend=None):
        self._svc = _build_service(json_path, "https://www.googleapis.com/auth/drive.file", "drive", "v3")
        # httplib2 is not thread-safe: one shared service, every call under this lock.
        self._lock = threading.Lock()
        self.folder_id = folder_id
        self.make_public = make_public

        self._shorten = None
        if shortener_backend:
            try:
                import pyshorteners
                self._shorten = getattr(pyshorteners.Shortener(timeout=5), shortener_backend).short
            except Exception as e:
                log.warning("URL shortener unavailable: %s", e)

        threading.Thread(target=self._keepalive, daemon=True).start()

    def _keepalive(self):
        # Cheap call every 45 s: keeps the OAuth token fresh and the TLS connection open,
        # so an upload after a quiet period doesn't pay for a token fetch + handshake.
        while True:
            try:
                with self._lock:
                    self._svc.about().get(fields="kind").execute()
            except Exception as e:
                log.warning("Drive keepalive failed: %s", e)
            time.sleep(45)

    def _set_public(self, file_id: str):
        for attempt in range(3):
            try:
                with self._lock:
                    self._svc.permissions().create(
                        fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id"
                    ).execute()
                return
            except Exception as e:
                log.warning("Failed to make file public (try %d): %s", attempt + 1, e)
                time.sleep(1)

    def upload_jpeg(self, data: bytes, name: str) -> str:
        """Upload JPEG bytes; returns the file's view URL (shortened if enabled)."""
        from googleapiclient.http import MediaIoBaseUpload

        body = {"name": name}
        if self.folder_id:
            body["parents"] = [self.folder_id]
        # Simple (single-request) upload; resumable adds extra round trips.
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype="image/jpeg", resumable=False)
        with self._lock:
            created = self._svc.files().create(
                body=body, media_body=media, fields="id,webViewLink"
            ).execute(num_retries=2)
        file_id = created["id"]
        url = created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"

        if self.make_public:
            # Off the capture path: the QR shows while the permission is being set.
            threading.Thread(target=self._set_public, args=(file_id,), daemon=True).start()

        if self._shorten is not None:
            try:
                url = self._shorten(url)
            except Exception as e:
                log.warning("URL shortener failed: %s", e)
        return url


class SheetsLogger:
    """Appends rows to one worksheet from a background thread, off the capture path."""

    def __init__(self, json_path: str, spreadsheet_id: str, worksheet: str):
        self._svc = _build_service(json_path, "https://www.googleapis.com/auth/spreadsheets", "sheets", "v4")
        self.spreadsheet_id = spreadsheet_id
        self.worksheet = worksheet
        self._ensure_worksheet()
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _ensure_worksheet(self):
        meta = self._svc.spreadsheets().get(
            spreadsheetId=self.spreadsheet_id, fields="sheets.properties.title"
        ).execute()
        titles = {s["properties"]["title"] for s in meta.get("sheets", [])}
        if self.worksheet not in titles:
            self._svc.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": self.worksheet}}}]},
            ).execute()
            log.info("Worksheet created: %s", self.worksheet)

    def append(self, row: list):
        self._q.put(row)

    def _worker(self):
        # Only this thread uses the service, so no lock is needed.
        rng = "'" + self.worksheet.replace("'", "''") + "'!A1"
        while True:
            row = self._q.get()
            try:
                self._svc.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=rng,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [row]},
                ).execute()
            except Exception as e:
                log.error("Google Sheets append failed: %s", e)


def resolve_spreadsheet_id(value, path: str) -> str:
    """
    Spreadsheet ID from value, or else from the first non-empty, non-# line of path.
    Either may be a bare ID or a full sheet URL.
    """
    if not value:
        try:
            with open(path, encoding="utf-8") as f:
                value = next((s for s in (line.strip() for line in f) if s and not s.startswith("#")), "")
        except OSError as e:
            log.error("Cannot read spreadsheet ID file %s: %s", path, e)
            return ""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
    return m.group(1) if m else value.strip()
