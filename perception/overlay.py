"""Draw annotations on a frame.

Annotation dicts: {"type": "point"|"box"|"text"|"cross", "uv"|"box"|"text", "color", "label"}.
Convention: predicted tip = hollow circle (point with "hollow": True), detected marker = filled
circle (point), target = cross, VLM box = rectangle with its label.
"""

from __future__ import annotations

COLORS = {
    "green": (0, 220, 0),
    "red": (0, 0, 255),
    "blue": (255, 120, 0),
    "yellow": (0, 230, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "orange": (0, 140, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
DEFAULT_COLOR = {"point": "green", "box": "yellow", "text": "white", "cross": "red"}


def _color(a: dict):
    c = a.get("color") or DEFAULT_COLOR.get(a.get("type"), "white")
    if isinstance(c, str):
        if c.startswith("#") and len(c) == 7:
            return int(c[5:7], 16), int(c[3:5], 16), int(c[1:3], 16)
        return COLORS.get(c, COLORS["white"])
    return tuple(int(x) for x in c)


def _hollow(a: dict) -> bool:
    return bool(
        a.get("hollow")
        or a.get("filled") is False
        or a.get("style") == "hollow"
        or "pred" in str(a.get("label", "")).lower()
    )


def _text(img, text, org, color, s, k):
    import cv2

    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), k + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, k, cv2.LINE_AA)


def draw(bgr, annotations: list[dict] | None):
    """A copy of `bgr` with the annotations drawn; bad annotations are skipped."""
    import cv2

    img = bgr.copy()
    s = max(1.0, img.shape[1] / 640)
    k = max(1, round(s))
    for a in annotations or []:
        try:
            t, color, label = a.get("type"), _color(a), a.get("label") or ""
            if t == "point":
                u, v = a["uv"]
                r = int(7 * s)
                cv2.circle(
                    img, (int(u), int(v)), r, color, 2 * k if _hollow(a) else -1, cv2.LINE_AA
                )
                if label:
                    _text(img, label, (u + r + 3, v - r), color, s, k)
            elif t == "cross":
                u, v = a["uv"]
                cv2.drawMarker(img, (int(u), int(v)), color, cv2.MARKER_CROSS, int(22 * s), 2 * k)
                if label:
                    _text(img, label, (u + 12 * s, v + 18 * s), color, s, k)
            elif t == "box":
                x0, y0, x1, y1 = (int(c) for c in a["box"])
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2 * k)
                if label:
                    _text(img, label, (x0, max(y0 - 6 * s, 14 * s)), color, s, k)
            elif t == "text":
                u, v = a.get("uv") or (10, 22 * s)
                _text(img, str(a.get("text", label)), (u, v), color, s, k)
        except (KeyError, TypeError, ValueError, cv2.error):
            continue
    return img
