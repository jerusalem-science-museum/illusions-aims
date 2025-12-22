import os.path

from infra.core.ansi import Ansi
from infra.run.common import *


# LOGOR_LEVEL = logging.DEBUG

CAMERA_EFFECTS = [
    'none', 'negative', 'solarize', 'sketch', 'emboss', 'hatch',
    'watercolor', 'colorswap', 'posterise', 'cartoon'
]

# CAMERA_FPS = 30
CAMERA_RESOLUTION = (1640, 1232)
CAMERA_CROP = (180, 80, 410, 230)

SEP_SCALE = 0.005

PICTURES_HEIGHT_SCALE = 0.15
PICTURES_PATH = '/tmp/%s.jpg'
PICTURES_DATETIME_FORMAT = DATETIME_FORMAT.replace('-', '_').replace(':', '_').replace(' ', '_')

FRAME_PATH = os.path.join(BASIC_PATH, 'sms_camera', 'res', 'frame2.png')
FRAME_POS = (215, 223)
FRAME_SIZE = (1200, 980)
LOGO3_PATH = os.path.join(BASIC_PATH, 'sms_camera', 'res', 'logo3.png')

KEYS_PATH = os.path.join(BASIC_PATH, 'keys')
SHORT_URL_ARGS = {
    'engine': 'Google',
    'api_key': open(os.path.join(KEYS_PATH, 'old/logger_api_key.txt'), 'r').read(),
    'timeout': 1
}

SERVICE_ACCOUNT_PATH = os.path.join(KEYS_PATH, 'old/logger-995ad2d4b91d.json')
DRIVE_SMS_CAMERA_FOLDER = '0B2AIf2iKCHvpYkxMM0NUNFE0bG8'
DRIVE_UPLOAD_TIMEOUT = 50
SHEET_SMS_LOG_NAME = 'sms_log'
WORKSHEET_SMS_NAME = '0'

# ---- GPIO Button (pour déclencher la photo) ----
BUTTON_GPIO_PIN = 17
BUTTON_BOUNCE_TIME = 0.1

