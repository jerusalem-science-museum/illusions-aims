# Camera QR Application

## Overview
This project captures images from a camera, detects activity in a small ROI, starts a countdown, takes a picture, uploads it with the Google helpers used in the code, generates a QR code from the returned URL, and displays recent QR codes in the application window.

The GUI is built with Tkinter, image processing is done with OpenCV, and QR generation is done with `qrcode` + Pillow.

## Project files

- `main.py`  
  Main application. Handles camera preview, GUI, countdown, capture flow, upload flow, QR display, and optional Google Sheets logging.

- `graphics.py`  
  UI and image helpers: flip, overlay handling, QR image creation, QR strip animation, layout helpers, ROI manager, countdown controller.

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