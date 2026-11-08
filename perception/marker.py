"""The tape marker on the gripper: HSV threshold, largest blob (or the one nearest a hint), centroid."""

from __future__ import annotations

from core.types import UV

# Yellow tape, measured live on the C270 (H 18-31, S 123-207, V 121-230). OpenCV HSV: H 0-179.
DEFAULTS = {"hsv_low": [16, 110, 110], "hsv_high": [38, 255, 255], "min_area": 80}


def find_blobs(bgr, marker_cfg: dict | None = None) -> list[tuple[UV, int]]:
    """Every blob inside the HSV range with at least `min_area` px, as (centroid, area), largest
    first. Calibration uses this to drop blobs that never move (food, clutter)."""
    import cv2
    import numpy as np

    if bgr is None:
        return []
    m = dict(DEFAULTS)
    if marker_cfg:
        m.update(marker_cfg.get("marker", marker_cfg))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(m["hsv_low"], np.uint8), np.array(m["hsv_high"], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    blobs = [
        ((float(cents[i][0]), float(cents[i][1])), int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= m["min_area"]
    ]
    return sorted(blobs, key=lambda b: -b[1])


def find_marker(bgr, marker_cfg: dict | None = None, near: UV | None = None) -> UV | None:
    """Centroid of the largest blob inside the HSV range with at least `min_area` px, else None.
    With `near` (the predicted pixel), the qualifying blob closest to it wins instead, so yellow
    clutter elsewhere in the frame doesn't steal the match.
    `marker_cfg` is the `marker:` section of cameras.yaml or the whole file."""
    import cv2
    import numpy as np

    if bgr is None:
        return None
    m = dict(DEFAULTS)
    if marker_cfg:
        m.update(marker_cfg.get("marker", marker_cfg))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(m["hsv_low"], np.uint8), np.array(m["hsv_high"], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    ok = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= m["min_area"]]
    if not ok:
        return None
    if near is not None:
        i = min(ok, key=lambda i: (cents[i][0] - near[0]) ** 2 + (cents[i][1] - near[1]) ** 2)
    else:
        i = max(ok, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return float(cents[i][0]), float(cents[i][1])
