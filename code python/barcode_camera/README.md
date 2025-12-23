# Google Drive OAuth (Desktop App) Setup — Full Checklist

This README explains how to create a **fresh Gmail account**, set up a **new Google Cloud project**, configure **OAuth**, and generate the files you need so your Python app can upload photos to **your personal Google Drive** (without service accounts / quota issues).

> Why OAuth?  
> Service Accounts often fail for personal Drive uploads (quota/ownership limitations). OAuth logs in as a real Google user, so files go into *that user's* Drive.

---

## 0) Create a new Gmail account (recommended)

1. Create a new Gmail address (example: `myproject.camera@gmail.com`).
2. Log in to that account in your browser.
3. Use this account for **all** steps below (Cloud Console + Drive).

---

## A) Create a Google Cloud project

1. Open **Google Cloud Console**.
2. Top bar → project selector → **New Project**.
3. Name: e.g. `barcode-camera`
4. Click **Create**.
5. Make sure the new project is selected in the top bar.

---

## B) Enable the Google Drive API

1. In Cloud Console: **APIs & Services → Library**
2. Search **Google Drive API**
3. Open it and click **Enable**

(You do **not** need “API Keys API” for this use case.)

---

## C) Configure OAuth consent screen

In new UI you may see “Google Auth Platform” pages (Overview/Branding/Audience…).
That’s normal.

1. Go to: **APIs & Services → OAuth consent screen**
   - or **Google Auth Platform → Audience / Branding** (depending on UI)

2. **Branding**
   - App name: `Barcode Camera`
   - User support email: your new Gmail
   - Developer contact email: your new Gmail
   - Save

3. **Audience**
   - Choose **External** (for normal Gmail accounts)
   - Publishing status: keep **Testing** (recommended)
   - Add **Test users**:
     - Add the Gmail you will log in with (usually the same new Gmail)
   - Save

4. **Data Access / Scopes**
   - Add scopes:
     - `.../auth/drive.file`  (recommended: upload files your app creates)
   - Save

> If your app stays in **Testing**, only “Test users” can log in.  
> That’s fine for your project.

---

## D) Create OAuth Client ID (Desktop App) and download `oauth_client.json`

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Application type: **Desktop app**
4. Name: `Desktopapp` (or any name)
5. Click **Create**
6. Click **Download JSON**
7. Save it as:
   - `oauth_client.json` (recommended filename)

✅ This file contains keys like: `installed`, `client_id`, `client_secret`, `auth_uri`, `token_uri`.

---

## E) Put the OAuth file in your project

Recommended structure:

```
code python/barcode_camera/
  keys/
    oauth_client.json
```

Then in your Python code, set:

```python
OAUTH_CLIENT_JSON = os.path.join(KEYS_PATH, "oauth_client.json")
```

---

## F) First login to generate `token.json`

Your app should run an OAuth flow once and create a token cache file.

Typical behavior:
- On first run, it opens a browser page → you log in → you grant permission
- It saves `token.json` locally (so next runs do NOT ask again)

Recommended token location:

```
code python/barcode_camera/keys/token.json
```

And in code:

```python
OAUTH_TOKEN_JSON = os.path.join(KEYS_PATH, "token.json")
```

If you previously got:
- **403 access_denied**
then either:
- you are not a “Test user”, or
- consent screen is not fully configured, or
- you are logged into a different Google account in the browser.

Fix:
1. Add your email in **Audience → Test users**
2. Try again in an incognito window with the correct Gmail.

---

## G) Remove the service account JSON (if you switch to OAuth)

You should **not** use service account credentials for personal Drive uploads in this setup.

So:
- remove `GOOGLE_SERVICE_ACCOUNT_JSON` usage in your code
- use `oauth_client.json` + `token.json` instead

---

## H) Git safety: ignore secrets but keep folders

### 1) Ignore all files inside `keys/` but keep the folder

In your **main `.gitignore`** (repo root):

```gitignore
# Ignore everything in keys/
code python/barcode_camera/keys/*

# But keep a placeholder file
!code python/barcode_camera/keys/.gitkeep
```

Then create the placeholder:

```bash
mkdir -p "code python/barcode_camera/keys"
touch "code python/barcode_camera/keys/.gitkeep"
git add "code python/barcode_camera/keys/.gitkeep" .gitignore
git commit -m "Keep keys folder, ignore secrets"
```

✅ Result: the folder exists in git, but your JSON/token files are not committed.

---

## I) If a secret was already committed: remove it from history

If GitHub blocks your push due to a leaked file, you must rewrite history and force-push.

**Recommended tool: `git filter-repo`**

### 1) Install (one-time)
- Linux:
  ```bash
  sudo apt install git-filter-repo
  ```
- Or via pip:
  ```bash
  pip install git-filter-repo
  ```

### 2) Remove the file from all history
Example:
```bash
git filter-repo --path "code python/barcode_camera/keys/logger_api_key.txt" --invert-paths
```

Then:
```bash
git add -A
git commit -m "Remove leaked secret from history" || true
git push --force
```

> If you see: “Refusing to destructively overwrite repo history since this does not look like a fresh clone”  
> clone a fresh copy and run filter-repo there, or run:
> ```bash
> git filter-repo --force --path "..." --invert-paths
> ```
> (Be careful: this rewrites history.)

---

## J) What to do with the two JSON files you downloaded

You may have downloaded **two different types** of JSON:

1) **Service Account key JSON**  
   - has: `client_email`, `private_key`, `token_uri`, etc  
   - used for server-to-server auth  
   - **NOT recommended** for personal Drive uploads

2) **OAuth client JSON (Desktop app)**  
   - has: `installed` object with `client_id`, `client_secret`, etc  
   - **THIS is the one you need for OAuth**

✅ For your app (personal Drive): keep only:
- `oauth_client.json`
- `token.json` (generated after first login)

---

## Quick troubleshooting

### “Access blocked / app not verified” / 403 access_denied
- Add your Gmail under **Audience → Test users**
- Ensure consent screen fields are filled (Branding)
- Try incognito / log in with the correct account

### “Service Accounts do not have storage quota”
- That’s a service account limitation for personal Drive
- Use OAuth instead (this README)

---

## Final checklist

- [ ] New Gmail account created
- [ ] New Cloud Project created
- [ ] Drive API enabled
- [ ] OAuth consent screen configured (Branding + Audience + Scopes)
- [ ] OAuth client (Desktop app) created
- [ ] `oauth_client.json` placed in `keys/`
- [ ] First run generates `token.json`
- [ ] `.gitignore` prevents pushing secrets
