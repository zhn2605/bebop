"""Where the mouth is in 3D (base frame, mm): the face camera's mouth pixel, at the depth the side
camera gives (triangulated) or, when the side camera can't see the mouth, the depth the eye spacing
gives (a typical 63 mm apart; people vary by about 8 %, so about 3 cm at 40 cm)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.types import FaceState, Vec3

EYES_MM = 63.0
SIDE_AGREE_MM = 80.0  # the triangulated mouth may sit this far from the eye-spacing one
SIDE_PX = 30.0  # and must reproject this close to both mouth pixels


def locate_mouth(calib, face: FaceState, side: FaceState | None = None,
                 eyes_mm: float = EYES_MM) -> tuple[Vec3, str] | None:
    """(mouth xyz, how) from one frame of each camera, or None when the face camera has no mouth
    or no calibration."""
    if not face.present or face.mouth_uv is None or "face" not in calib.K:
        return None
    est = None
    if face.eyes_px > 0:
        # Turning the head shortens the eye spacing in the image by cos(yaw).
        f = calib.K["face"][0, 0]
        depth = f * eyes_mm * math.cos(math.radians(face.yaw_deg)) / face.eyes_px
        est = calib.at_depth("face", face.mouth_uv, depth)
    if side is not None and side.present and side.mouth_uv is not None and "side" in calib.P:
        uvs = {"face": face.mouth_uv, "side": side.mouth_uv}
        p = calib.triangulate(uvs)
        off = max(math.dist(calib.project(c, p), uv) for c, uv in uvs.items())
        if off <= SIDE_PX and (est is None or math.dist(p, est) <= SIDE_AGREE_MM):
            return p, "face + side cameras"
    return (est, "face camera + eye spacing") if est is not None else None


class MouthTrack:
    """Mouth positions over time; stable() is their median over the last `window_s` when there
    are at least `min_n` of them within `spread_mm` of it (a still head), else None."""

    def __init__(self, window_s: float = 1.0, min_n: int = 4, spread_mm: float = 30.0):
        self.window_s, self.min_n, self.spread_mm = window_s, min_n, spread_mm
        self.samples: deque[tuple[float, Vec3, str]] = deque(maxlen=60)

    def add(self, t: float, xyz: Vec3, how: str) -> None:
        self.samples.append((t, tuple(float(v) for v in xyz), how))

    def stable(self, now: float) -> Vec3 | None:
        pts = [p for t, p, _ in self.samples if now - t <= self.window_s]
        if len(pts) < self.min_n:
            return None
        med = np.median(np.array(pts), axis=0)
        if max(math.dist(p, med) for p in pts) > self.spread_mm:
            return None
        return float(med[0]), float(med[1]), float(med[2])

    @property
    def how(self) -> str:
        return self.samples[-1][2] if self.samples else ""
