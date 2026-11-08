"""The 0-1000 grid overlay the VLM reads boxes against, and conversions to and from pixels.

Boxes in the 0-1000 convention are [ymin, xmin, ymax, xmax] (Gemini's); pixel boxes are
core.types.Box (xmin, ymin, xmax, ymax).
"""

from __future__ import annotations

from core.types import Box

STEP = 100
_ALPHA = 0.35  # line opacity


def _bgr(image):
    import cv2
    import numpy as np

    if isinstance(image, (bytes, bytearray)):
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not a decodable image")
        return bgr
    return image


def size(image) -> tuple[int, int]:
    """(width, height) of JPEG bytes or a BGR array."""
    h, w = _bgr(image).shape[:2]
    return w, h


def with_grid(image, step: int = STEP, quality: int = 85) -> bytes:
    """JPEG bytes of image (JPEG bytes or BGR) with thin grid lines every step on the 0-1000 scale,
    labelled along the top and left edges."""
    import cv2

    bgr = _bgr(image).copy()
    h, w = bgr.shape[:2]
    lines = bgr.copy()
    for n in range(step, 1000, step):
        x, y = round(n * w / 1000), round(n * h / 1000)
        cv2.line(lines, (x, 0), (x, h - 1), (0, 255, 255), 1)
        cv2.line(lines, (0, y), (w - 1, y), (0, 255, 255), 1)
    out = cv2.addWeighted(lines, _ALPHA, bgr, 1 - _ALPHA, 0)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.35
    for n in range(step, 1000, step):
        x, y = round(n * w / 1000), round(n * h / 1000)
        for text, org in ((str(n), (x + 2, 10)), (str(n), (2, y - 2))):
            cv2.putText(out, text, org, font, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, text, org, font, scale, (0, 255, 255), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return jpeg.tobytes()


def to_pixels(box_1000, w: int, h: int) -> Box:
    """[ymin, xmin, ymax, xmax] in 0-1000 to a pixel Box, clamped and ordered."""
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) for v in box_1000)
    ymin, ymax = sorted((ymin, ymax))
    xmin, xmax = sorted((xmin, xmax))
    return (xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000)


def to_1000(box: Box, w: int, h: int) -> list[int]:
    """A pixel Box to [ymin, xmin, ymax, xmax] in 0-1000."""
    xmin, ymin, xmax, ymax = box
    return [round(ymin * 1000 / h), round(xmin * 1000 / w), round(ymax * 1000 / h),
            round(xmax * 1000 / w)]
