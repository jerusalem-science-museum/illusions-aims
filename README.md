# Camera QR Application

## Overview
This project captures images from a camera, detects activity in a small ROI, starts a countdown, takes a picture, uploads it with the Google helpers used in the code, generates a QR code from the returned URL, and displays recent QR codes in the application window.

Frames are drawn with OpenCV and shown fullscreen through `ffplay` (from ffmpeg). QR codes are generated with `qrcode` + Pillow.

## Project files

- keys are located in [Museum's OneDrive](https://madaorgil-my.sharepoint.com/shared?id=%2Fsites%2FMakeMada%2FShared%20Documents%2F2%2E%20%D7%AA%D7%A2%D7%A8%D7%95%D7%9B%D7%95%D7%AA%2F%D7%90%D7%A9%D7%9C%D7%99%D7%95%D7%AA%2F2%2E%20%D7%91%D7%99%D7%AA%20%D7%9E%D7%9C%D7%90%D7%9B%D7%94%2F2%2E%20%D7%9E%D7%95%D7%A6%D7%92%D7%99%D7%9D%2F%D7%90%D7%99%D7%99%D7%9E%D7%A1%2F6%2E%20%D7%A7%D7%95%D7%93&listurl=https%3A%2F%2Fmadaorgil%2Esharepoint%2Ecom%2Fsites%2FMakeMada%2FShared%20Documents&viewid=b5506206%2Dcce1%2D4963%2D8af9%2D671538374454)

- `main.py`  
  Main application: camera reader, ffplay display, main loop, countdown and capture flow (upload, QR, Sheets logging).

- `render.py`  
  Drawing: the branded template with the camera in its window, the saved photo, countdown, flash, ROI box and QR images.

- `qr_queue.py`  
  The moving QR queue drawn in the template's QR slots.

- `roi_detector.py`  
  ROI logic and trigger detection based on color change over time.

- `google_upload.py`  
  Upload helper for Google Drive and optional Google Sheets logging.

- `constant.py`  
  Main configuration file: camera settings, ROI settings, file paths, overlays, Google settings, logging settings, and QR history settings.

- `log.py`  
  Rotating log file configuration.

## Python version

Recommended: **Python 3.10+**

## Required libraries

Create a file named `requirements.txt` with the following content:

```txt
opencv-python
numpy
Pillow
qrcode[pil]
google-api-python-client
google-auth
google-auth-httplib2
pyshorteners