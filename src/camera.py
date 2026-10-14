"""
Cameras behind one interface. Each is read on a background thread that keeps only the latest frame:
    latest      newest raw frame (None until the first one); frames: how many were read so far
    bgra(raw)   a new BGRA image of raw for the preview and the ROI, mirrored per FLIP_PREVIEW
    photo()     a BGR frame for the uploaded photo, mirrored per FLIP_CAPTURE (None if none came)
    status()    the camera's current exposure and gains, for the FPS log ("" if unknown)
    release()

PiCamera (picamera2) runs a single configuration with two YUV420 streams made from the same frames:
    lores   small, every frame: preview + ROI
    main    photo size, copied only when a photo is requested
The ISP crops the sensor to the template window's aspect (ScalerCrop), scales both streams and mirrors
them (Transform), so no pixel is captured only to be thrown away and the CPU never flips or crops.
Both streams are converted by yuv_to(), so the preview and the photo have identical colors.

Exposure, gain, white balance and color processing are fixed (see constant.py), so the image of an
unchanged scene never changes and only a visitor can trigger the ROI. The tuning file is patched to
turn off the two adaptive stages that remain when AE/AWB are off (see _tuning()).
"""
import threading
import time
from typing import Optional

import cv2
import numpy as np

import constant as cfg
from log import get_logger

log = get_logger()

_FLIPS = {"h": (1, 0), "v": (0, 1), "hv": (1, 1)}  # FLIP_MODE -> (hflip, vflip)
_FLIP_CODES = {(1, 0): 1, (0, 1): 0, (1, 1): -1}   # (hflip, vflip) -> cv2.flip code


def flip(img: np.ndarray) -> np.ndarray:
    hv = _FLIPS.get(cfg.FLIP_MODE.lower().strip())
    return img if hv is None else cv2.flip(img, _FLIP_CODES[hv])


def yuv_to(raw: np.ndarray, size, code: int) -> np.ndarray:
    """
    Convert a picamera2 YUV420 (I420) array to BGR/BGRA (code: cv2.COLOR_YUV2BGR(A)_I420).
    raw is (h * 3/2, stride): rows can be padded past the width. Padded I420 is a valid I420 image of
    width stride (the chroma planes have stride / 2), so it is converted whole and the padding cut off.

    OpenCV's I420 conversion is BT.601 limited range (Y 16-235), which is libcamera's Smpte170m, the
    colour space PiCamera configures. picamera2's default for still/preview is sYCC (full range): that
    would come out with crushed blacks, clipped whites and oversaturated colors.
    """
    w, h = size
    out = cv2.cvtColor(raw[:h * 3 // 2], code)
    return out if out.shape[1] == w else np.ascontiguousarray(out[:, :w])


def _algo(tuning: dict, name: str) -> Optional[dict]:
    """The parameters of one algorithm in a libcamera tuning file (version 1 or 2), or None."""
    if tuning.get("version", 1) == 1:
        return tuning.get(name)
    return next((a[name] for a in tuning.get("algorithms", []) if name in a), None)


def _tuning() -> dict:
    """
    The camera's tuning file with the adaptive stages turned off, so the image only depends on the scene:
      rpi.contrast ce_enable=0  adaptive contrast picks a new tone curve per frame from the histogram,
                                so a visitor walking in would shift every pixel, the ROI included
      rpi.alsc n_iter=0         adaptive lens-shading correction slowly drifts the colors per region;
                                0 keeps only the calibrated tables
      rpi.ccm (PICAM_CCM)       a fixed color matrix, for a camera without an IR-cut filter
    """
    from picamera2 import Picamera2

    tuning = Picamera2.load_tuning_file(cfg.PICAM_TUNING_FILE)
    for name, key, value in (("rpi.contrast", "ce_enable", 0), ("rpi.alsc", "n_iter", 0)):
        params = _algo(tuning, name)
        if params is None:
            log.warning("Tuning file %s has no %s; can't set %s.", cfg.PICAM_TUNING_FILE, name, key)
        else:
            params[key] = value
    if cfg.PICAM_CCM is not None:
        m = [float(v) for v in np.asarray(cfg.PICAM_CCM).ravel()]
        if len(m) != 9:
            raise ValueError("PICAM_CCM must be 3x3 (9 values).")
        ccm = _algo(tuning, "rpi.ccm")
        if ccm is None:
            log.warning("Tuning file %s has no rpi.ccm; PICAM_CCM ignored.", cfg.PICAM_TUNING_FILE)
        else:
            # The ISP interpolates between matrices by color temperature; the same one at both ends is fixed
            ccm["ccms"] = [{"ct": 2000, "ccm": m}, {"ct": 10000, "ccm": m}]
    return tuning


def flicker_period_us() -> float:
    """Lights flicker at twice the mains frequency; exposures that are multiples of this don't band."""
    return 1e6 / (2 * cfg.MAINS_HZ)


class PiCamera:
    def __init__(self, aspect: float, auto: bool = False, main_format: str = "YUV420"):
        """
        aspect: width / height of the template window; the sensor is cropped to it.
        auto: let AE/AWB run (only for --calibrate-camera); otherwise everything is fixed.
        main_format: "YUV420", or "RGB888" (BGR in memory) for picam_check's color test.
        """
        from picamera2 import Picamera2

        self.cam = Picamera2(tuning=_tuning())
        self.auto = auto
        self.main_format = main_format
        self.main_size, self.lores_size = self._configure(aspect)
        self.cam.set_controls(self._controls())
        self.cam.start()

        self.latest: Optional[np.ndarray] = None
        self.metadata: dict = {}
        self.frames = 0
        self._photo_lock = threading.Lock()   # one photo request at a time
        self._photo_wanted = threading.Event()
        self._photo_ready = threading.Event()
        self._photo = None                    # (lores, main) raw arrays of one request
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    @staticmethod
    def _size(width: int, aspect: float):
        """Stream size: width as given (a multiple of 64 keeps rows unpadded), height a multiple of 16."""
        return width, max(16, round(width / aspect / 16) * 16)

    def _configure(self, aspect: float):
        from libcamera import ColorSpace, Transform

        main_size = self._size(cfg.PICAM_MAIN_W, aspect)
        lores_size = self._size(cfg.PICAM_LORES_W, aspect)
        hv = _FLIPS.get(cfg.FLIP_MODE.lower().strip(), (0, 0))
        hw_flip = cfg.FLIP_PREVIEW and cfg.FLIP_CAPTURE and hv != (0, 0)

        def make(transform):
            config = self.cam.create_video_configuration(
                main={"size": main_size, "format": self.main_format},
                lores={"size": lores_size, "format": "YUV420"},  # lores must be YUV420 on a Pi 4 or older
                # Pinned: picamera2 would otherwise pick the mode closest to the main size, which on the
                # OV5647 is 1920x1080, a center crop of the sensor (narrower view).
                sensor={"output_size": tuple(cfg.PICAM_SENSOR_MODE), "bit_depth": 10},
                transform=transform,
                colour_space=ColorSpace.Smpte170m(),  # what OpenCV's I420 conversion expects (see yuv_to)
                buffer_count=4,
                display=None, encode=None,
            )
            self.cam.align_configuration(config)
            self.cam.configure(config)
            return self.cam.camera_configuration()

        self._flip_preview, self._flip_capture = cfg.FLIP_PREVIEW, cfg.FLIP_CAPTURE
        config = None
        if hw_flip:
            try:
                config = make(Transform(hflip=hv[0], vflip=hv[1]))
                t = config["transform"]
                if (int(t.hflip), int(t.vflip)) == hv:
                    self._flip_preview = self._flip_capture = False
                else:
                    log.warning("The camera ignored the mirror transform; mirroring in software.")
                    config = None
            except Exception:
                log.warning("The camera can't mirror (%s); mirroring in software.", cfg.FLIP_MODE, exc_info=True)
                config = None
        if config is None:
            config = make(Transform())

        main, lores = config["main"], config["lores"]
        self._crop = self._scaler_crop(main["size"][0] / main["size"][1])
        log.info("Pi camera: sensor mode %s, main %s %s (stride %s), lores %s %s (stride %s), %s, "
                 "transform %s, crop %s, %s",
                 config.get("sensor"), main["size"], main["format"], main.get("stride"),
                 lores["size"], lores["format"], lores.get("stride"),
                 config.get("colour_space"), config["transform"], self._crop,
                 "auto exposure (calibration)" if self.auto else "fixed exposure")
        return tuple(main["size"]), tuple(lores["size"])

    def _scaler_crop(self, aspect: float):
        """Center crop of the sensor mode's area with the streams' aspect, shrunk by PICAM_ZOOM."""
        try:
            x, y, w, h = self.cam.camera_controls["ScalerCrop"][1]  # largest crop for this mode
        except (KeyError, TypeError, ValueError):
            x, y, w, h = self.cam.camera_properties["ScalerCropMaximum"]
        if w / h > aspect:
            cw, ch = h * aspect, h
        else:
            cw, ch = w, w / aspect
        zoom = max(1.0, cfg.PICAM_ZOOM)
        cw, ch = round(cw / zoom), round(ch / zoom)
        return x + (w - cw) // 2, y + (h - ch) // 2, cw, ch

    def _available(self, controls: dict) -> dict:
        """Drop controls this camera / libcamera version doesn't have (names changed across versions)."""
        known = self.cam.camera_controls
        missing = sorted(k for k in controls if k not in known)
        if missing:
            log.debug("Camera controls not available here: %s", missing)
        return {k: v for k, v in controls.items() if k in known}

    def _controls(self) -> dict:
        frame_us = int(1e6 / cfg.PREVIEW_FPS)
        controls = {
            "ScalerCrop": self._crop,
            "FrameDurationLimits": (frame_us, frame_us),  # fixed frame rate
            "Brightness": 0.0, "Contrast": 1.0, "Sharpness": 1.0, "Saturation": cfg.PICAM_SATURATION,
            "NoiseReductionMode": 1,  # draft.NoiseReductionModeEnum.Fast, the same for preview and photo
        }
        if self.auto:
            controls.update({"AeEnable": True, "AwbEnable": True, "ExposureTimeMode": 0, "AnalogueGainMode": 0,
                             # AE only picks exposures that are multiples of the flicker period
                             "AeFlickerMode": 1, "AeFlickerPeriod": int(flicker_period_us())})
        else:
            controls.update(self.manual(cfg.PICAM_EXPOSURE_US, cfg.PICAM_ANALOGUE_GAIN, cfg.PICAM_COLOUR_GAINS))
        return self._available(controls)

    @staticmethod
    def manual(exposure_us: float, gain: float, colour_gains) -> dict:
        """Controls that fix exposure, gain and white balance (both the old and the new libcamera names)."""
        frame_us = 1e6 / cfg.PREVIEW_FPS
        if exposure_us > frame_us:
            log.warning("Exposure %d us is longer than a frame at %d fps; using %d us.",
                        exposure_us, cfg.PREVIEW_FPS, frame_us)
            exposure_us = frame_us
        return {
            "AeEnable": False, "ExposureTimeMode": 1, "AnalogueGainMode": 1,  # 1 = manual (libcamera >= 0.5)
            "ExposureTime": int(exposure_us), "AnalogueGain": float(gain),
            "AwbEnable": False, "ColourGains": (float(colour_gains[0]), float(colour_gains[1])),
        }

    def set_controls(self, controls: dict) -> None:
        self.cam.set_controls(self._available(controls))

    def _reader(self):
        fails = 0
        while self.running:
            try:
                req = self.cam.capture_request()
            except Exception:
                fails += 1
                if fails in (1, 30, 120):
                    log.warning("Camera capture failed (%d). Retrying...", fails, exc_info=fails == 1)
                time.sleep(0.01)
                continue
            fails = 0
            try:  # copy what's needed and hand the buffer straight back: held buffers stall the camera
                lores = req.make_array("lores")
                self.metadata = req.get_metadata()
                if self._photo_wanted.is_set():
                    self._photo = (lores, req.make_array("main"))
                    self._photo_wanted.clear()
                    self._photo_ready.set()
            finally:
                req.release()
            self.latest = lores
            self.frames += 1

    def bgra(self, raw: np.ndarray) -> np.ndarray:
        img = yuv_to(raw, self.lores_size, cv2.COLOR_YUV2BGRA_I420)
        return flip(img) if self._flip_preview else img

    def photo_raw(self, timeout: float = 1.0):
        """(lores, main) raw arrays from the next frame, or None."""
        with self._photo_lock:
            self._photo = None
            self._photo_ready.clear()
            self._photo_wanted.set()
            if not self._photo_ready.wait(timeout):
                self._photo_wanted.clear()
                return None
            return self._photo

    def photo(self) -> Optional[np.ndarray]:
        raw = self.photo_raw()
        if raw is None:
            return None
        main = raw[1]
        img = main if self.main_format == "RGB888" else yuv_to(main, self.main_size, cv2.COLOR_YUV2BGR_I420)
        return flip(img) if self._flip_capture else img

    def status(self) -> str:
        md = self.metadata
        if "ExposureTime" not in md:
            return ""
        gains = md.get("ColourGains", (0, 0))
        return " | cam exp %d us, gain %.2f x %.2f, wb %.2f/%.2f" % (
            md["ExposureTime"], md.get("AnalogueGain", 0), md.get("DigitalGain", 0), gains[0], gains[1])

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cam.stop()
        self.cam.close()


class UsbCamera:
    """A USB webcam through OpenCV (V4L2 on the Pi), e.g. for development on a PC."""

    def __init__(self, index: int):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open the camera on index {index}.")

        # The reader thread drains the queue as fast as frames arrive, so extra buffers add no latency.
        # With only 1, the driver has nowhere to put the next frame while the previous one is being
        # decoded, and on a Pi that drops every other frame (30 -> 15 fps).
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAM_BUFFERS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, cfg.PREVIEW_FPS)  # last: V4L2 resets the frame rate when the size changes
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        log.info("USB camera opened: %dx%d @ %s fps, format %s",
                 self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 self.cap.get(cv2.CAP_PROP_FPS), fourcc.to_bytes(4, "little").decode("ascii", "replace"))
        if cfg.CAM_LOCK_AUTO:
            self._lock_auto()

        # cap.read() returns a new array each time and nobody mutates it, so no copies are needed.
        self.latest: Optional[np.ndarray] = None
        self.frames = 0
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _lock_auto(self):
        """Freezes exposure, gain, white balance and focus so scene changes can't trigger the ROI.

        Auto exposure runs briefly first so the frozen value suits the room's lighting.
        Support varies by camera and driver, so every step is best effort.
        """
        cap = self.cap
        end = time.time() + cfg.CAM_LOCK_SETTLE_S
        while time.time() < end:
            cap.read()

        exposure = cfg.CAM_EXPOSURE if cfg.CAM_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = cfg.CAM_GAIN if cfg.CAM_GAIN is not None else cap.get(cv2.CAP_PROP_GAIN)
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)

        # V4L2 uses 1 for manual exposure, DirectShow uses 0.25.
        manual = 1 if hasattr(cv2, "CAP_V4L2") and cap.getBackendName() == "V4L2" else 0.25
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if wb > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        log.info("Camera locked: auto_exposure=%s exposure=%s gain=%s wb=%s",
                 cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_EXPOSURE),
                 cap.get(cv2.CAP_PROP_GAIN), cap.get(cv2.CAP_PROP_WB_TEMPERATURE))

    def _reader(self):
        fails = 0
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                fails = 0
                self.latest = frame
                self.frames += 1
            else:
                fails += 1
                if fails in (1, 30, 120):
                    log.warning("Camera read failed (%d). Retrying...", fails)
                time.sleep(0.01)

    def bgra(self, raw: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(flip(raw) if cfg.FLIP_PREVIEW else raw, cv2.COLOR_BGR2BGRA)

    def photo(self) -> Optional[np.ndarray]:
        frame = self.latest
        if frame is None:
            return None
        return flip(frame) if cfg.FLIP_CAPTURE else frame

    def status(self) -> str:
        return ""

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def open_camera(backend: str, index: int, aspect: float):
    if backend == "picam":
        return PiCamera(aspect)
    if backend == "usb":
        return UsbCamera(index)
    raise ValueError(f"Unknown camera backend {backend!r} (use 'picam' or 'usb').")


def calibrate(aspect: float) -> None:
    """
    --calibrate-camera: find the fixed settings for constant.py, in the room's real lighting.
      1. Auto exposure settles (only flicker-safe exposures), then exposure and gain are frozen
         (exposure rounded to a multiple of the flicker period, gain compensating).
      2. White balance on a gray/white card held over the center of the view: the red/blue gains are
         adjusted until the card is neutral. Done on the card rather than by AWB, because without
         an IR-cut filter AWB is fooled by the infrared.
    """
    cam = PiCamera(aspect, auto=True)
    try:
        log.info("Calibration: letting auto exposure settle for %.0f s...", cfg.CAM_CALIBRATE_SETTLE_S)
        time.sleep(cfg.CAM_CALIBRATE_SETTLE_S)
        md = cam.metadata
        exp, gain = md["ExposureTime"], md["AnalogueGain"] * md.get("DigitalGain", 1.0)
        period = flicker_period_us()
        longest = int(1e6 / cfg.PREVIEW_FPS / period) * period
        if longest < period:
            log.warning("At %d fps a frame is shorter than the %d us flicker period; banding is possible.",
                        cfg.PREVIEW_FPS, period)
            fixed_exp = exp
        else:
            fixed_exp = min(longest, max(period, round(exp / period) * period))
        gain = max(1.0, gain * exp / fixed_exp)
        r, b = md.get("ColourGains", (1.5, 1.5))
        log.info("Auto picked exposure %d us, gain %.2f, AWB gains %.2f/%.2f; fixing exposure %d us, gain %.2f.",
                 exp, md["AnalogueGain"], r, b, fixed_exp, gain)
        cam.set_controls(PiCamera.manual(fixed_exp, gain, (r, b)))

        input("Hold a gray or white card (matte paper works) so it fills the center of the view, "
              "then press Enter...")
        for _ in range(12):
            time.sleep(0.5)  # new controls take a few frames to reach the output
            img = cam.bgra(cam.latest)
            h, w = img.shape[:2]
            patch = img[2 * h // 5:3 * h // 5, 2 * w // 5:3 * w // 5, :3]
            mb, mg, mr = patch.reshape(-1, 3).mean(axis=0)
            log.info("Card BGR %.0f %.0f %.0f, gains %.3f/%.3f", mb, mg, mr, r, b)
            if max(mb, mg, mr) >= 250:
                log.warning("The card is clipped (too bright); lower the light or PICAM_EXPOSURE_US after.")
            if min(mb, mg, mr) < 5:
                log.error("The card is too dark to measure.")
                break
            if abs(mg / mr - 1) < 0.01 and abs(mg / mb - 1) < 0.01:
                break
            r = min(8.0, max(0.3, r * mg / mr))
            b = min(8.0, max(0.3, b * mg / mb))
            cam.set_controls({"ColourGains": (r, b)})

        text = ("PICAM_EXPOSURE_US = %d\nPICAM_ANALOGUE_GAIN = %.2f\nPICAM_COLOUR_GAINS = (%.3f, %.3f)"
                % (fixed_exp, gain, r, b))
        log.info("Calibration done. Put these in constant.py:\n%s", text)
    finally:
        cam.release()
