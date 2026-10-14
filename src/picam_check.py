"""
Checks the Pi camera setup on the Pi itself (run from src/ with the venv active, nothing else using the camera):
    python3 picam_check.py

1. Tuning: the adaptive stages are off, and the color matrix in use (a starting point for PICAM_CCM).
2. Color: the ISP's own RGB output (main, RGB888) against our YUV420 -> BGR conversion of the lores
   stream of the same frame. If the colour space were wrong, the conversion would be off by tens of
   levels; it should be within a few. The camera's YUV is also tried as full range, to show which fits.
3. Speed: the per-frame lores conversion and the photo conversion.
4. Fixed settings: for CHECK_S seconds, wave a hand in front of the right half of the view.
   Exposure, gains and white balance must not move, and the left side of the image should not
   change brightness (it would if anything still adapted to the scene).
"""
import time
from pathlib import Path

import cv2
import numpy as np

import constant as cfg
from camera import PiCamera, _algo, _tuning, yuv_to
from render import Template

CHECK_S = 15.0


def full_range_bgr(raw: np.ndarray, size) -> np.ndarray:
    """I420 -> BGR assuming full-range BT.601 (sYCC / JPEG), i.e. what picamera2's default colour space gives."""
    w, h = size
    stride = raw.shape[1]
    y = raw[:h, :w].astype(np.float32)
    chroma = raw[h:h * 3 // 2].ravel()
    n = (h // 2) * (stride // 2)
    u = chroma[:n].reshape(h // 2, stride // 2)[:, :w // 2]
    v = chroma[n:2 * n].reshape(h // 2, stride // 2)[:, :w // 2]
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128
    bgr = np.dstack([y + 1.772 * u, y - 0.344136 * u - 0.714136 * v, y + 1.402 * v])
    return np.clip(bgr, 0, 255).astype(np.uint8)


def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute difference per channel of two slightly blurred images (the two streams are scaled
    separately by the ISP, so edges differ a little)."""
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32)
    return np.abs(a - b).reshape(-1, 3).mean(axis=0)


def check_tuning():
    print("\n== 1. Tuning file", cfg.PICAM_TUNING_FILE)
    tuning = _tuning()
    contrast, alsc, ccm = (_algo(tuning, n) for n in ("rpi.contrast", "rpi.alsc", "rpi.ccm"))
    print("rpi.contrast ce_enable =", contrast and contrast.get("ce_enable"), "(want 0)")
    print("rpi.alsc n_iter =", alsc and alsc.get("n_iter"), "(want 0)")
    if ccm:
        for entry in ccm.get("ccms", []):
            m = np.round(np.array(entry["ccm"]).reshape(3, 3), 3)
            print(f"rpi.ccm at {entry['ct']} K:", m.tolist())


def check_color(aspect: float):
    print("\n== 2. Color conversion (main = ISP RGB, lores = our YUV conversion)")
    cam = PiCamera(aspect, main_format="RGB888")
    try:
        time.sleep(2)
        lores, main = cam.photo_raw()
        isp = cv2.resize(main, cam.lores_size, interpolation=cv2.INTER_AREA)  # RGB888 is BGR in memory
        ours = yuv_to(lores, cam.lores_size, cv2.COLOR_YUV2BGR_I420)
        wrong = full_range_bgr(lores, cam.lores_size)
        y = lores[:cam.lores_size[1], :cam.lores_size[0]]
        d_ours, d_wrong = diff(ours, isp), diff(wrong, isp)
        print("lores stride:", lores.shape[1], "for width", cam.lores_size[0])
        print("Y range: %d..%d (limited range stays within ~16..235)" % (y.min(), y.max()))
        print("mean BGR  ISP %s  ours %s" % (isp.reshape(-1, 3).mean(0).round(1), ours.reshape(-1, 3).mean(0).round(1)))
        print("diff BGR  limited-range (ours) %s   full-range %s" % (d_ours.round(2), d_wrong.round(2)))
        ok = d_ours.max() < 4 and d_ours.mean() < d_wrong.mean()
        print("PASS" if ok else "FAIL: the YUV conversion doesn't match the ISP's colors")
        out = Path(cfg.LOG_FOLDER)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "picam_check_isp.png"), isp)
        cv2.imwrite(str(out / "picam_check_ours.png"), ours)
        print("Saved logs/picam_check_isp.png and logs/picam_check_ours.png to compare by eye.")
    finally:
        cam.release()


def check_speed_and_static(aspect: float):
    cam = PiCamera(aspect)
    try:
        time.sleep(2)
        print("\n== 3. Speed")
        raw = cam.latest
        t = time.perf_counter()
        for _ in range(50):
            cam.bgra(raw)
        print("lores -> BGRA: %.2f ms per frame" % ((time.perf_counter() - t) * 1000 / 50))
        t = time.perf_counter()
        photo = cam.photo()
        print("photo (wait for frame + convert): %.1f ms, %s" % ((time.perf_counter() - t) * 1000,
                                                                photo.shape if photo is not None else None))

        print("\n== 4. Fixed settings: wave a hand over the RIGHT half of the view for %.0f s..." % CHECK_S)
        keys = ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains")
        seen = {k: [] for k in keys}
        left = []
        end = time.time() + CHECK_S
        while time.time() < end:
            md = cam.metadata
            for k in keys:
                if k in md:
                    seen[k].append(np.ravel(md[k]).astype(float))
            img = cam.bgra(cam.latest)
            left.append(img[:, :img.shape[1] // 4, :3].reshape(-1, 3).mean(axis=0))
            time.sleep(0.1)
        ok = True
        for k, values in seen.items():
            if not values:
                print(f"{k}: not reported")
                continue
            v = np.array(values)
            lo, hi = v.min(axis=0), v.max(axis=0)
            moved = np.any(hi - lo > 0.01 * np.maximum(np.abs(lo), 1e-6))
            ok &= not moved
            print(f"{k}: {lo.round(3).tolist()} .. {hi.round(3).tolist()}{'  <-- CHANGED' if moved else ''}")
        spread = np.ptp(np.array(left), axis=0)
        print("left quarter mean BGR spread:", spread.round(2), "(should stay within a few levels)")
        print("PASS" if ok else "FAIL: the camera still adjusts itself")
    finally:
        cam.release()


def main():
    bw, bh = Template(cfg.MOCKUP_PNG, cfg.PREVIEW_W, cfg.PREVIEW_H).win_size
    check_tuning()
    check_color(bw / bh)
    check_speed_and_static(bw / bh)


if __name__ == "__main__":
    main()
