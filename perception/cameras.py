"""Camera grabber threads (latest frame only) and the fake world that renders frames in fake mode."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np

from core.types import CAMERAS, Vec3
from perception.calib import Calibration, fake_calibration
from perception.overlay import draw

FAKE_YAML = Path(__file__).resolve().parent.parent / "config" / "fake.yaml"


def _source(src):
    """int index for /dev/videoN (also through /dev/v4l/by-id symlinks), else the value as given."""
    if isinstance(src, str):
        if src.isdigit():
            return int(src)
        m = re.fullmatch(r"/dev/video(\d+)", os.path.realpath(src))
        if m:
            return int(m.group(1))
    return src


def open_capture(src, width: int = 1280, height: int = 720, fps: int = 15):
    """V4L2 capture with MJPG, the requested size, buffer size 1; None when it gives no frame."""
    import cv2

    cap = cv2.VideoCapture(_source(src), cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before the size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):
        if cap.read()[0]:
            return cap
        time.sleep(0.1)
    cap.release()
    return None


class Cameras:
    """`cfg` is cameras.yaml. With `fake`, every name in CAMERAS is rendered by the FakeWorld."""

    def __init__(self, cfg: dict | None, fake: FakeWorld | None = None):
        cfg = cfg or {}
        self.specs: dict[str, dict] = cfg.get("cameras") or {}
        self.fps = float(cfg.get("fps", 15))
        self.fake = fake
        self.names: list[str] = []
        self._frames: dict[str, tuple] = {}
        self._run = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> Cameras:
        self._run.set()
        if self.fake is not None:
            for name in CAMERAS:
                self._spawn(name, self._render, name)
            return self
        for name, spec in self.specs.items():
            spec = spec or {}
            cap = self._open(spec)
            if cap is None:
                print(f"cameras: {name} (source {spec.get('source')!r}) did not open; skipped")
                continue
            self._spawn(name, self._grab, name, cap, spec)
        return self

    def _open(self, spec: dict):
        w, h = spec.get("width", 1280), spec.get("height", 720)
        return open_capture(spec.get("source", 0), w, h, int(self.fps))

    def _spawn(self, name, target, *args):
        self.names.append(name)
        th = threading.Thread(target=target, args=args, name=f"cam-{name}", daemon=True)
        th.start()
        self._threads.append(th)

    def _grab(self, name, cap, spec):
        fails = 0
        while self._run.is_set():
            ok, frame = cap.read() if cap is not None else (False, None)
            if ok:
                self._frames[name] = (frame, time.time())
                fails = 0
                continue
            fails += 1
            if fails == 20:
                print(f"cameras: {name} stopped delivering frames; reopening")
            if fails % 20 == 0:
                if cap is not None:
                    cap.release()
                cap = self._open(spec)
            time.sleep(0.05)
        if cap is not None:
            cap.release()

    def _render(self, name):
        failed = False
        while self._run.is_set():
            try:
                self._frames[name] = (self.fake.render(name), time.time())
            except Exception as e:  # noqa: BLE001  (keep rendering once tip_fn works again)
                if not failed:
                    print(f"cameras: fake {name} render failed: {e!r}")
                failed = True
            time.sleep(1.0 / self.fps)

    def stop(self) -> None:
        self._run.clear()
        for th in self._threads:
            th.join(timeout=2)
        self._threads.clear()

    def latest(self, name: str) -> tuple[np.ndarray, float] | None:
        return self._frames.get(name)

    def jpeg(
        self, name: str, annotations: list[dict] | None = None, quality: int = 70
    ) -> bytes | None:
        import cv2

        got = self.latest(name)
        if got is None:
            return None
        img = draw(got[0], annotations) if annotations else got[0]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes() if ok else None


def _fake_placement() -> dict | None:
    try:
        import yaml

        return (yaml.safe_load(FAKE_YAML.read_text()) or {}).get("cameras")
    except (ImportError, OSError):
        return None


class FakeWorld:
    """Renders 640x480 fake frames: grey table, brown food disc, green marker at the tip, a "fake"
    label; the `face` camera shows a drawn face with an open mouth.

    `tip_fn()` returns the jaw tip (a Pose or an xyz). The food sticks to the tip once the tip comes
    within `food_radius_mm` of it (drawn hanging below and in front of the jaws, clear of the marker), and is gone ("eaten") once the held food is `release_mm` from
    where it was. `held_fn()`, when given, decides instead (food gone once it turns false again).
    `chain_fn()`, when given, returns base, shoulder, elbow, wrist and tip xyz for drawing the arm
    (else a post to the shoulder and one bar to the tip). `reset()` puts the food back.
    """

    TIP_RADIUS_MM = 8.0

    def __init__(
        self,
        tip_fn: Callable[[], object],
        calib: Calibration | None = None,
        food_xyz: Vec3 = (250.0, 0.0, 15.0),
        food_radius_mm: float = 15.0,
        held_fn: Callable[[], bool] | None = None,
        release_mm: float = 200.0,
        chain_fn: Callable[[], list[Vec3]] | None = None,
    ):
        self.tip_fn = tip_fn
        self.calib = calib or fake_calibration(_fake_placement())
        self.food_xyz = tuple(float(c) for c in food_xyz)
        self.food_radius_mm = float(food_radius_mm)
        self.held_fn = held_fn
        self.release_mm = release_mm
        self.chain_fn = chain_fn
        self.size = (640, 480)
        self.held = False
        self.eaten = False

    def reset(self, food_xyz: Vec3 | None = None) -> None:
        if food_xyz is not None:
            self.food_xyz = tuple(float(c) for c in food_xyz)
        self.held = self.eaten = False

    def tip(self) -> Vec3 | None:
        t = self.tip_fn()
        if t is None:
            return None
        return tuple(float(c) for c in getattr(t, "xyz", t))

    def _update(self, tip: Vec3 | None) -> None:
        if self.held_fn is not None:
            held = bool(self.held_fn())
            self.eaten = self.eaten or (self.held and not held)
            self.held = held
            return
        if tip is None or self.eaten:
            return
        d = math.dist(tip, self.food_xyz)
        if not self.held and d <= self.food_radius_mm:
            self.held = True
        elif self.held and d > self.release_mm:
            self.held, self.eaten = False, True

    def render(self, name: str) -> np.ndarray:
        if name == "face":
            return self.render_face()
        import cv2

        w, h = self.size
        img = np.full((h, w, 3), (175, 180, 185), np.uint8)
        tip = self.tip()
        self._update(tip)
        if name in self.calib.P:
            P = self.calib.P[name]
            proj = lambda xyz: self.calib.project(name, xyz)
            corners = [(-200, -400, 0), (700, -400, 0), (700, 400, 0), (-200, 400, 0)]
            if min(P[2] @ np.r_[c, 1.0] for c in corners) > 10:
                cv2.fillConvexPoly(img, np.int32([proj(c) for c in corners]), (125, 125, 125))
                for x in range(-100, 700, 100):
                    cv2.line(
                        img, _pt(proj((x, -400, 0))), _pt(proj((x, 400, 0))), (110, 110, 110), 1
                    )
                for y in range(-400, 500, 100):
                    cv2.line(
                        img, _pt(proj((-200, y, 0))), _pt(proj((700, y, 0))), (110, 110, 110), 1
                    )
            else:
                img[:] = (125, 125, 125)
            if not self.held and not self.eaten:
                self._disc(img, name, self.food_xyz, self.food_radius_mm, (30, 75, 140))
            if tip is not None:
                pts = self.chain_fn() if self.chain_fn is not None else None
                if not pts:
                    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 120.0), tip]
                for a, b in pairwise(pts):
                    cv2.line(img, _pt(proj(a)), _pt(proj(b)), (60, 60, 60), 10, cv2.LINE_AA)
                if self.held:
                    self._disc(img, name, self._held_xyz(tip), self.food_radius_mm, (30, 75, 140))
                self._disc(img, name, tip, self.TIP_RADIUS_MM, (0, 255, 0))
        _label(img, f"fake {name}", h)
        return img

    def _held_xyz(self, tip: Vec3) -> Vec3:
        """Held food centre: a little below the tip and just in front of it (away from the base),
        touching the green marker without covering it, from above and from the side."""
        x, y, z = tip
        rho = math.hypot(x, y) or 1.0
        ahead = self.TIP_RADIUS_MM + self.food_radius_mm
        return (x + x / rho * ahead, y + y / rho * ahead, z - 0.4 * self.food_radius_mm)

    def _disc(self, img, name, xyz, r_mm, color):
        import cv2

        u, v = self.calib.project(name, xyz)
        r_px = max(
            np.hypot(*np.subtract(self.calib.project(name, np.add(xyz, d)), (u, v)))
            for d in r_mm * np.eye(3)
        )
        c, r = (round(u * 16), round(v * 16)), round(max(r_px, 4.0) * 16)  # 4 subpixel bits
        cv2.circle(img, c, r, color, -1, cv2.LINE_AA, 4)

    def render_face(self) -> np.ndarray:
        import cv2

        w, h = self.size
        img = np.full((h, w, 3), (205, 210, 215), np.uint8)
        cx, cy = w // 2, h // 2 - 10
        skin, dark = (140, 175, 215), (40, 45, 55)
        cv2.ellipse(img, (cx, cy + 210), (190, 90), 0, 180, 360, (120, 90, 60), -1)  # shoulders
        cv2.rectangle(img, (cx - 45, cy + 120), (cx + 45, cy + 200), (125, 160, 200), -1)  # neck
        cv2.ellipse(img, (cx, cy - 40), (128, 120), 0, 180, 360, dark, -1)  # hair
        cv2.ellipse(img, (cx, cy), (115, 150), 0, 0, 360, skin, -1)
        for ear in (cx - 112, cx + 112):
            cv2.ellipse(img, (ear, cy + 5), (14, 28), 0, 0, 360, skin, -1)
        for ex in (cx - 45, cx + 45):
            cv2.ellipse(img, (ex, cy - 20), (24, 12), 0, 0, 360, (245, 245, 245), -1)
            cv2.circle(img, (ex, cy - 20), 10, (50, 70, 100), -1)
            cv2.circle(img, (ex, cy - 20), 5, (10, 10, 10), -1)
            cv2.ellipse(img, (ex, cy - 20), (24, 12), 0, 180, 360, (70, 90, 120), 2)
            cv2.ellipse(img, (ex, cy - 48), (28, 8), 0, 180, 360, dark, 5)  # brow
        nose = np.int32([(cx, cy - 15), (cx - 14, cy + 35), (cx + 14, cy + 35)])
        cv2.polylines(img, [nose], False, (100, 130, 175), 3, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + 75), (42, 30), 0, 0, 360, (90, 80, 170), -1)  # lips
        cv2.ellipse(img, (cx, cy + 76), (32, 21), 0, 0, 360, (35, 25, 60), -1)  # open mouth
        cv2.ellipse(img, (cx, cy + 61), (22, 5), 0, 0, 360, (235, 235, 235), -1)  # teeth
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _label(img, "fake face", h)
        return img


def _pt(uv) -> tuple[int, int]:
    return round(float(uv[0])), round(float(uv[1]))


def _label(img, text: str, h: int) -> None:
    import cv2

    cv2.putText(
        img, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
