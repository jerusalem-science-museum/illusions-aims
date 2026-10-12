"""
Fullscreen display drawn on the GPU through SDL2's renderer (pygame._sdl2).

Each frame only the camera image (camera resolution) is uploaded. The GPU crops and scales it
into the template's picture window, draws the template on top with its alpha, then the QRs,
and scales the whole canvas to the screen. The template and each QR are uploaded once.

Uploads are BGRA, which SDL calls ARGB8888 (its default texture format on little-endian),
so the bytes are copied as is with no conversion.
"""
import os
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402
from pygame._sdl2.video import Renderer, Texture, Window, get_drivers  # noqa: E402

from log import get_logger  # noqa: E402

log = get_logger()


def _surface(bgra: np.ndarray) -> pygame.Surface:
    """A Surface sharing bgra's memory (keep the array alive while the Surface is used)."""
    return pygame.image.frombuffer(bgra, (bgra.shape[1], bgra.shape[0]), "BGRA")


class Display:
    def __init__(self, template, fullscreen: bool = True):
        """template: render.Template at the canvas size (cfg.PREVIEW_W x cfg.PREVIEW_H)."""
        pygame.init()
        h, w = template.bgra.shape[:2]
        self.window = Window("Ames room", size=(w, h), fullscreen_desktop=fullscreen, resizable=not fullscreen)
        try:
            self.renderer = Renderer(self.window, accelerated=1, vsync=False)
        except Exception:
            log.warning("No GPU renderer available (drivers: %s); using software rendering, which is slow.",
                        [d.name for d in get_drivers()])
            self.renderer = Renderer(self.window, accelerated=0, vsync=False)
        self.renderer.logical_size = (w, h)  # draw in template pixels; SDL scales to the screen
        pygame.mouse.set_visible(False)

        self.template = Texture.from_surface(self.renderer, _surface(np.ascontiguousarray(template.bgra)))
        x0, y0, x1, y1 = template.win
        self.win = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        self._video: Optional[Texture] = None
        self._panel: Optional[Texture] = None
        self._qrs = {}  # id(qr image) -> (image, texture); the image is kept so its id stays unique
        log.info("Display ready: %dx%d canvas, %s.", w, h, "fullscreen" if fullscreen else "windowed")

    @staticmethod
    def _streaming(tex: Optional[Texture], renderer: Renderer, img: np.ndarray) -> Texture:
        """Upload img into tex, (re)creating it when the size changes."""
        size = (img.shape[1], img.shape[0])
        if tex is None or (tex.width, tex.height) != size:
            tex = Texture(renderer, size, streaming=True)
        tex.update(_surface(img))
        return tex

    def show(self, video: np.ndarray, crop, qrs, panel: Optional[np.ndarray] = None) -> bool:
        """
        Draw one frame. video: BGRA camera image; crop: (x, y, w, h) of it shown in the window;
        qrs: [(image, x, y, size)] oldest first; panel: optional BGRA image drawn at the window's corner.
        Returns False once the window is closed (q, Esc or the window's close button).
        """
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                return False

        r = self.renderer
        r.draw_color = (0, 0, 0, 255)
        r.clear()
        self._video = self._streaming(self._video, r, video)
        self._video.draw(srcrect=pygame.Rect(crop), dstrect=self.win)
        if panel is not None:
            self._panel = self._streaming(self._panel, r, panel)
            self._panel.draw(dstrect=(self.win.x, self.win.y))
        self.template.draw()

        live = {}
        for img, x, y, size in qrs:
            entry = self._qrs.get(id(img))
            if entry is None:
                bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                entry = (img, Texture.from_surface(r, _surface(bgra)))
            live[id(img)] = entry
            entry[1].draw(dstrect=(x, y, size, size))
        self._qrs = live  # textures of QRs that left the queue are freed

        r.present()
        return True

    def close(self):
        pygame.quit()
