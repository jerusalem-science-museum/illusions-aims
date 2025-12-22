import os
import time
import logging
import datetime
import platform
from PIL import Image, ImageDraw
from picamera import PiCamera
from pyshorteners import Shortener
import serial
import serial.threaded


from infra.app import app
from infra.old_modules.sim800 import sim800
from infra.modules.google.drive import drive
from infra.modules.google.sheets import sheets
from sms_camera.src.camera_gsm_to_url import constants


class CameraGsmToUrl(app.App):
    _logger = logging.getLogger('camera_gsm_to_url')


    def __init__(self):
        app.App.__init__(self, constants, spam_loggers=sheets.Sheets.SPAM_LOGGERS +
            ('PIL.PngImagePlugin', 'urllib3.connectionpool'))
        self._modules.extend((sim800, drive, sheets))
        # google drive uploader
        try:
            self.drive = drive.Drive(constants.SERVICE_ACCOUNT_PATH)
        except:
            self._logger.exception('self.drive')
            self.drive = None
        # google sheets logger
        try:
            self.sheets = sheets.Sheets(constants.SERVICE_ACCOUNT_PATH)
        except:
            self._logger.exception('self.sheet')
            self.sheets = None
        # url shorter
        try:
            self.short_url = Shortener(**constants.SHORT_URL_ARGS).short
        except:
            self._logger.exception('self.short_url')
            self.short_url = None
        # google sheets name is the hostname
        constants.WORKSHEET_SMS_NAME = platform.node()
        # last pictures overlays
        self.pictures = []
        # attach to raspberrypi camera
        self.camera = PiCamera()
        # configure camera
        if hasattr(constants, 'CAMERA_FPS'):
            self.camera.framerate = constants.CAMERA_FPS
        if hasattr(constants, 'CAMERA_RESOLUTION'):
            self.camera.resolution = constants.CAMERA_RESOLUTION
        if not hasattr(constants, 'CAMERA_CROP'):
            constants.CAMERA_CROP = (0, 0, 0, 0)
        if not hasattr(constants, 'SCREEN_RESOLUTION'):
            constants.SCREEN_RESOLUTION = tuple(map(int, os.popen(r'tvservice -s | grep -oP "\d\d+x\d\d+"', 'r').read().strip().split('x')))
        # calc draw stuff and draw preview
        self.calc_and_draw_preview()
        # draw pictures overlays
        self.draw_pictures()
        # log display parameters
        self._logger.info('\nmain window: %s sep: %s\ncamera resolution: %sx%s @ %s fps', 
            (self.left, self.top, self.width, self.height), self.sep,
            self.camera.resolution.width, self.camera.resolution.height, self.camera.framerate)
        self._logger.debug('\nfirst picture window: %s count: %s', 
            (self.pictures_l, self.pictures_t, self.pictures_w, self.pictures_h), self.pictures_c)
        # gsm module
        try:
            self._gsm_uart = serial.serial_for_url(**constants.GSM_UART)
            self._gsm_reader = serial.threaded.ReaderThread(self._gsm_uart, sim800.Sim800)
            self._gsm_reader.start()
            self.gsm = self._gsm_reader.connect()[1]
        except:
            self._logger.exception('gsm')
            self._logger.error('gsm problem: check DC power is stable..\n'
                'new modules needs to be configure with:\n'
                'ATE0;+CMGF=1;+CNMI=2,2,0,0,0;+CSCS="UCS2";+CSMP=17,167,0,8;+CSAS;+IPR=115200\n'
                'AT&W')
            self.gsm = None
        else:
            self.gsm.status_changed = self.gsm_status_changed
            self.gsm.sms_recived = self.gsm_sms_recived

    def gsm_status_changed(self):
        self._logger.info('gsm_status_changed: %s', self.gsm.status)
        if self.gsm.status == 'ALIVE':
            self._logger.info(constants.GSM_DATA_FORMAT,
                self.gsm.get_csq(), self.gsm.get_vbat(), self.gsm.get_temperature())
        elif self.gsm.status == 'TIMEOUT':
            self._logger.warning('gsm did not respond')

    def gsm_sms_recived(self, number, send_time, text):
        # normalize sms text, number and send_time
        text = text.encode(errors='replace').decode().strip().replace('\n', ' ').replace('\t', ' ').replace('\r', '')
        normalize_number = self.gsm.normalize_phone_number(number)
        send_time = send_time.strftime(constants.DATETIME_FORMAT)
        self._logger.info('AT: %s FROM: %s MESSAGES: %s', send_time, normalize_number, text)
        if text == 'REBOOT':
            # self._logger.info(constants.REBOOT_FORMAT)
            self.send_sms(number, constants.REBOOT_FORMAT, False)
            os.system('shutdown -r')
        elif text == 'GSM DATA':
            try:
                t = constants.GSM_DATA_FORMAT % (
                    self.gsm.get_csq(), self.gsm.get_vbat(), self.gsm.get_temperature())
            except:
                self._logger.warning('cant read gsm data')
                return
            # self._logger.info(t)
            self.send_sms(number, t.replace(', ', '\n'), False)
        else:
            url = self.capture_and_share(number)
            # log to worksheet
            if self.sheets is not None:
                try:
                    self.sheets.append_worksheet_table(constants.SHEET_SMS_LOG_NAME, constants.WORKSHEET_SMS_NAME,
                        send_time, normalize_number, text, url)
                except:
                    self._logger.error('capture_and_share: append_worksheet_table failed')

    def capture_and_share(self, number):
        try:
            path = self.take_picture()
        except:
            self._logger.error('capture_and_share: take_picture failed')
            return ''
        try:
            url = self.upload_picture(path)
        except:
            self._logger.exception('capture_and_share: upload_picture failed')
            return ''
        try:
            self.send_sms(number, constants.GSM_SEND_SMS_FORMAT % (url,))
        except:
            self._logger.error('capture_and_share: send_sms failed')
        return url

    def calc_and_draw_preview(self):
        self.left, self.top = 0, 0
        self.width, self.height = constants.SCREEN_RESOLUTION
        crop_l, crop_t, crop_r, crop_b = constants.CAMERA_CROP
        camera_w, camera_h = self.camera.resolution.width, self.camera.resolution.height
        croped_camera_w, croped_camera_h = camera_w - crop_l - crop_r, camera_h - crop_t - crop_b
        view_scale, camera_scale = croped_camera_w / croped_camera_h, camera_w / camera_h
        self.sep = int(min(self.width, self.height) * constants.SEP_SCALE)
        # calc pictures count, size and positions
        self.pictures_h = int((self.height - 3 * self.sep) * constants.PICTURES_HEIGHT_SCALE)
        self.pictures_w = int(view_scale * self.pictures_h)
        self.pictures_c = int((self.width - self.sep) / (self.pictures_w + self.sep))
        self.pictures_l = int((self.width - self.pictures_c * (self.pictures_w + self.sep) - self.sep) / 2) + self.sep + self.left
        self.pictures_t = self.sep + self.top
        # calc final view size and position
        view_t = self.sep + self.pictures_h + self.sep
        view_h = self.height - view_t - self.sep
        view_w = int(view_scale * view_h)
        max_view_w = self.width - self.sep * 2
        if view_w > max_view_w:
            view_w = max_view_w
            view_h = int(1 / view_scale * view_w)
        view_l = int((self.width - view_w) / 2) + self.left
        # calc camera preview size and position
        preview_scale = lambda x: int(x * view_w / camera_w)
        offset_l = preview_scale(crop_l)
        offset_t = preview_scale(crop_t)
        offset_r = preview_scale(crop_r)
        offset_b = preview_scale(crop_b)
        preview_l = view_l - offset_l
        preview_t = view_t - offset_t
        preview_w = view_w + preview_scale(crop_l + crop_r)
        # preview_h = view_h + preview_scale(crop_t + crop_b)
        # preview_w = view_w + offset_l + offset_r
        # preview_h = view_h + offset_t + offset_b
        preview_h = int(1 / camera_scale * preview_w)
        # draw camera preview
        self.camera.start_preview()
        self.camera.preview.fullscreen = False
        self.camera.preview.window = [preview_l, preview_t, preview_w, preview_h]
        # draw preview crop mask (black rectangle with transparent rectangle inside)
        img = Image.new('RGBA', (preview_w, preview_h), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle((offset_l, offset_t, preview_w - offset_r, preview_h - offset_b), (0, 0, 0, 0))
        self.preview_crop = self._image_to_overlay(img, layer=self.camera.preview.layer + 1)
        self.preview_crop.window = self.camera.preview.window

    def draw_pictures(self):
        # draw pictures overlays
        for i, p in enumerate(self.pictures):
            p.window = (self.pictures_l + i * (self.pictures_w + self.sep), self.pictures_t, self.pictures_w, self.pictures_h)

    def take_picture(self):
        # capture picture and return its path
        self.camera.preview.alpha = 100
        time.sleep(0.7)
        path = constants.PICTURES_PATH % (datetime.datetime.now().strftime(constants.PICTURES_DATETIME_FORMAT),)
        self._logger.info('take_picture: %s', path)
        self.camera.capture(path)
        self.camera.preview.alpha = 255
        # crop sides and re-save image
        img = Image.open(path, 'r')
        l, t, r, b = 140, 80, 300, 170 # constants.CAMERA_CROP
        img = img.crop((l, t, img.width - r, img.height - b))
        img.save(path)
        self.add_picture(path)
        # add frame and re-save image
        draw = Image.open(constants.LOGO3_PATH, 'r').convert('RGBA')
        img_w, img_h = img.size
        draw_w, draw_h = draw.size
        img.paste(draw, (int((img_w - draw_w) / 2), img_h - draw_h), draw)
        frame = Image.open(constants.FRAME_PATH, 'r').convert('RGBA')
        img = img.resize(constants.FRAME_SIZE)
        frame.paste(img, constants.FRAME_POS)
        img = frame
        img = img.convert('RGB')
        img.save(path)

        return path

    def add_picture(self, path):
        # add the given picture to pictures list
        self.pictures.append(self._image_path_to_overlay(
            path, resize=(self.pictures_w, self.pictures_h), transparent=False, layer=self.preview_crop.layer + 1))
        if len(self.pictures) > self.pictures_c:
            self.pictures[0].close()
            self.pictures.pop(0)
        self.draw_pictures()

    def set_effect(self, index):
        # set camera image effect by its index
        self._logger.info('set_effect: %s', index)
        self.camera.image_effect = constants.CAMERA_EFFECTS[index % len(constants.CAMERA_EFFECTS)]

    def upload_picture(self, path):
        if self.drive is None:
            raise IOError('can\t upload: self.drive is None')
        self._logger.info('upload_picture: %s', path)
        file_id = self.drive.upload_file(path, share=True, delete=True,
            parent_directory=constants.DRIVE_SMS_CAMERA_FOLDER, timeout=constants.DRIVE_UPLOAD_TIMEOUT)
        url = self.drive.VIEW_FILE_URL % (file_id,)
        if self.short_url is not None:
            url = self.short_url(self.drive.VIEW_FILE_URL % (file_id,))
        self._logger.debug('upload_picture url: %s', url)
        return url

    def _image_path_to_overlay(self, image_path, resize=None, *args, **kwargs):
        # open image and resize it if needed
        img = Image.open(image_path)
        if resize is not None:
            img = img.resize(resize)

        return self._image_to_overlay(img, *args, **kwargs)

    def _image_to_overlay(self, img, layer=0, alpha=255, fullscreen=False, transparent=True):
        # create required size (32, 16) padding for the image 
        pad = Image.new('RGBA' if transparent else 'RGB', [((n + m - 1) // m) * m for n, m in zip(img.size, (32, 16))])
        # paste the original image into the padding
        pad.paste(img)
        # crearw image overlay, and return it with its size
        overlay = self.camera.add_overlay(pad.tobytes(), img.size, layer=layer, alpha=alpha, fullscreen=fullscreen)
        overlay.width, overlay.height = img.size
        return overlay

    def send_sms(self, number, text, raise_exception=True):
        try:
            if number.replace('+', '').isdigit():
                self.gsm.send_sms(number, text)
        except:
            if raise_exception:
                raise

    def smart_reload(self, reason=None):
        self.reload()
        return

    def __exit__(self):
        try:
            self._gsm_reader.close()
        except:
            pass
        try:
            self.camera.close()
        except:
            pass
        app.App.__exit__(self)
