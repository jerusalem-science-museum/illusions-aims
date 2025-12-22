import logging
import os.path

from infra.core.ansi import Ansi
from infra.run.common import *


# LOGOR_LEVEL = logging.DEBUG

GSM_UART = {'url': 'loop://' if IS_WINDOWS else '/dev/ttyGsmUart', 'baudrate': 115200, 'timeout': 1}

CAMERA_EFFECTS = ['none', 'negative', 'solarize', 'sketch', 'emboss', 'hatch', 'watercolor', 'colorswap', 'posterise', 'cartoon']
# CAMERA_FPS = 30
# CAMERA_RESOLUTION = (1680, 1050)
# CAMERA_RESOLUTION = (1640, 922)
# CAMERA_RESOLUTION = (1440, 1080)
CAMERA_RESOLUTION = (1640, 1232)
# CAMERA_RESOLUTION = (1327, 996)
# CAMERA_RESOLUTION = (3280, 2464)
# CAMERA_CROP = (100, 0, 1328, 996)
# CAMERA_CROP = (100, 0, 200, 100)
CAMERA_CROP = (180, 80, 410, 230) # (170, 80, 380, 210)
# SCREEN_RESOLUTION = (1680, 1050)
# SCREEN_RESOLUTION = (1280, 720)

SEP_SCALE = 0.005

PICTURES_HEIGHT_SCALE = 0.15
PICTURES_PATH = '/tmp/%s.jpg'
PICTURES_DATETIME_FORMAT = DATETIME_FORMAT.replace('-', '_').replace(':', '_').replace(' ', '_')

# FRAME_PATH, FRAME_POS, FRAME_SIZE = os.path.join(BASIC_PATH, 'sms_camera', 'res', 'frame1.png'), (105, 105), (536, 452)
FRAME_PATH, FRAME_POS, FRAME_SIZE = os.path.join(BASIC_PATH, 'sms_camera', 'res', 'frame2.png'), (215, 223), (1200, 980)
LOGO3_PATH = os.path.join(BASIC_PATH, 'sms_camera', 'res', 'logo3.png')

GSM_SEND_SMS_FORMAT = 'התמונה שלך:\n%s'
REBOOT_FORMAT = 'system will reboot in a minute'
GSM_DATA_FORMAT = 'csq: %s, vbat: %s, temperature: %s'

KEYS_PATH = os.path.join(BASIC_PATH, 'keys')
# SHORT_URL_ARGS = {'engine': 'Google', 'api_key': open(os.path.join(KEYS_PATH, 'google_api_key.txt'), 'r').read(), 'timeout': 1} # {'engine': 'Tinyurl', 'timeout': 1}
SHORT_URL_ARGS = {'engine': 'Google', 'api_key': open(os.path.join(KEYS_PATH, 'old/logger_api_key.txt'), 'r').read(), 'timeout': 1} # {'engine': 'Tinyurl', 'timeout': 1}
SERVICE_ACCOUNT_PATH = os.path.join(KEYS_PATH, 'old/logger-995ad2d4b91d.json')
# SERVICE_ACCOUNT_PATH = os.path.join(KEYS_PATH, 'google_service_account.json')
DRIVE_SMS_CAMERA_FOLDER = '0B2AIf2iKCHvpYkxMM0NUNFE0bG8'
DRIVE_UPLOAD_TIMEOUT = 50
SHEET_SMS_LOG_NAME = 'sms_log'
WORKSHEET_SMS_NAME = '0'
