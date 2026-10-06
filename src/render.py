"""Drawing: the branded template, countdown/flash, ROI box and QR images. All images are BGR."""
from typing import Tuple

import cv2
import numpy as np
import qrcode

import constant as cfg
from log import get_logger

log = get_logger()

_FLIP_CODES = {"h": 1, "v": 0, "hv": -1}


def flip(img: np.ndarray) -> np.ndarray:
    code = _FLIP_CODES.get(cfg.FLIP_MODE.lower().strip())
    return img if code is None else cv2.flip(img, code)


class Template:
    """
    The RGBA template with a transparent picture window, loaded once at the screen size.

    Stored premultiplied, so putting video in the window is one blend of the window area:
        out = template * alpha + video * (1 - alpha)
    The camera image is center-cropped to the window's aspect, so it is never stretched,
    and the live preview shows exactly what the saved photo will contain.
    """

    def __init__(self, path: str, w: int, h: int):
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            log.error("Template %s missing or without alpha; showing the plain camera image.", path)
            rgba = np.zeros((h, w, 4), np.uint8)  # fully transparent: the window is the whole screen
        b, g, r, a = cv2.split(rgba)
        pre = cv2.merge([cv2.multiply(c, a, scale=1 / 255) for c in (b, g, r)] + [a])
        if pre.shape[:2] != (h, w):
            pre = cv2.resize(pre, (w, h), interpolation=cv2.INTER_AREA)
        self.bgr = np.ascontiguousarray(pre[:, :, :3])  # premultiplied template
        alpha = pre[:, :, 3]

        # Window = bounding box of the transparent area
        hole = alpha < 128
        cols = np.flatnonzero(hole.any(axis=0))
        rows = np.flatnonzero(hole.any(axis=1))
        if len(cols) == 0:
            log.warning("Template %s has no transparent window.", path)
            cols, rows = np.array([0, w - 1]), np.array([0, h - 1])
        x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        self.win = (x0, y0, x1, y1)
        self.win_size = (x1 - x0, y1 - y0)
        self._win_bgr = self.bgr[y0:y1, x0:x1].copy()
        self._win_inv_alpha = cv2.merge([255 - alpha[y0:y1, x0:x1]] * 3)

        self.photo_box = self._find_gold_frame(w, h)

    def _find_gold_frame(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Box of the gold frame = largest connected region differing from the corner background color."""
        sw = 1600
        sh = max(1, round(sw * h / w))
        small = cv2.resize(self.bgr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.int16)
        diff = (np.abs(small - small[2, 2]).sum(axis=2) > 40).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
        if n <= 1:
            return 0, 0, w, h
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        return (int(x * w / sw), int(y * h / sh),
                min(w, int(np.ceil((x + bw) * w / sw))), min(h, int(np.ceil((y + bh) * h / sh))))

    def fit(self, frame: np.ndarray):
        """
        Center-crop frame to the window's aspect and resize it to the window size.
        Returns (video, (crop_x, crop_y, scale)) so camera coordinates can be mapped onto video.
        """
        bw, bh = self.win_size
        fh, fw = frame.shape[:2]
        if fw * bh > fh * bw:  # frame is wider than the window
            cw, ch = fh * bw // bh, fh
        else:
            cw, ch = fw, fw * bh // bw
        cx, cy = (fw - cw) // 2, (fh - ch) // 2
        interp = cv2.INTER_LINEAR if bw >= cw else cv2.INTER_AREA
        video = cv2.resize(frame[cy:cy + ch, cx:cx + cw], (bw, bh), interpolation=interp)
        return video, (cx, cy, bw / cw)

    def blend(self, dst: np.ndarray, video: np.ndarray) -> None:
        """Write video (window-sized) into dst's window, under the template."""
        x0, y0, x1, y1 = self.win
        under = cv2.multiply(video, self._win_inv_alpha, scale=1 / 255)
        dst[y0:y1, x0:x1] = cv2.add(self._win_bgr, under)

    def photo(self, frame: np.ndarray) -> np.ndarray:
        """The image saved/uploaded after a capture: the gold frame with the photo inside."""
        out = self.bgr.copy()
        self.blend(out, self.fit(frame)[0])
        x0, y0, x1, y1 = self.photo_box
        return np.ascontiguousarray(out[y0:y1, x0:x1])


def draw_roi(video: np.ndarray, rect, mapping, active: bool) -> None:
    """Draw the ROI box (camera coordinates) onto the window-sized video: green = ready, red = not."""
    cx, cy, s = mapping
    x0, y0, x1, y1 = rect
    p0 = (round((x0 - cx) * s), round((y0 - cy) * s))
    p1 = (round((x1 - cx) * s), round((y1 - cy) * s))
    cv2.rectangle(video, p0, p1, (0, 255, 0) if active else (0, 0, 255), 2)


def draw_countdown(img: np.ndarray, n: int) -> None:
    text = str(n)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, min(w, h) / 250.0)
    thickness = max(2, int(scale * 2.5))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def flash(img: np.ndarray) -> None:
    img[:] = 0 if cfg.FLASH_COLOR.lower() == "black" else 255


def make_qr(url: str) -> np.ndarray:
    """QR code for url as a black-on-white 3-channel image (channel order doesn't matter)."""
    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(url)
    qr.make(fit=True)
    return np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"), dtype=np.uint8)
