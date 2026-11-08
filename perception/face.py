"""Face signals from the `face` camera with MediaPipe FaceLandmarker: mouth open, yaw/pitch, looking
away, where the mouth is (pixel) and the eye spacing (px, for its depth), and chewing. Debounced:
mouth_open needs the ratio above threshold for `open_s`, looking_away needs |yaw| above
`away_yaw_deg` or no face for `away_s` (and the same time to clear). Chewing = the lips opened past
`chew_hi` and closed under `chew_lo` at least twice within `chew_s`."""

from __future__ import annotations

import math
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

from core.types import FaceState

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "face_landmarker.task"
DEFAULTS = {
    "mouth_open_ratio": 0.35,
    "open_s": 0.3,
    "away_s": 0.5,
    "away_yaw_deg": 30.0,
    "yaw_sign": 1,
    "pitch_sign": 1,
    "yaw_offset_deg": 0.0,
    "hz": 15,
    "chew_hi": 0.15,
    "chew_lo": 0.07,
    "chew_s": 3.0,
}
LIP_TOP, LIP_BOTTOM, CORNER_L, CORNER_R = 13, 14, 61, 291
IRIS_R, IRIS_L = 468, 473  # iris centres; the landmarker's 478-point output has them


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the FaceLandmarker model on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"face: downloading {MODEL_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


class _Debounce:
    def __init__(self, hold_s: float):
        self.hold_s, self.value, self._since = hold_s, False, None

    def update(self, raw: bool, t: float) -> bool:
        if raw == self.value:
            self._since = None
        elif self._since is None:
            self._since = t
        elif t - self._since >= self.hold_s:
            self.value, self._since = raw, None
        return self.value


class FaceSignals:
    """`cfg` is cameras.yaml (its `face:` section is used) or that section itself. `fake=True` emits
    a constant open-mouth, looking-at-the-arm FaceState without MediaPipe."""

    def __init__(self, cameras, name: str = "face", cfg: dict | None = None, fake: bool = False):
        cfg = cfg or {}
        self.cfg = {**DEFAULTS, **(cfg.get("face") or cfg)}
        self.cameras, self.name, self.fake = cameras, name, fake
        self._state = FaceState(0.0, False, False, 0.0, 0.0, 0.0, False)
        self._open = _Debounce(self.cfg["open_s"])
        self._away = _Debounce(self.cfg["away_s"])
        self._closes: deque[float] = deque(maxlen=20)  # times the lips closed after opening
        self._lips_apart = False
        self._t0 = time.time()
        self._run = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> FaceState:
        if self.fake:  # open 2 s, then closed (chewing) 1.5 s, so a fake bite happens
            now = time.time()
            opened = (now - self._t0) % 3.5 < 2.0
            return FaceState(now, True, opened, 0.5 if opened else 0.05, 0.0, 0.0, False,
                             (320.0, 230.0), 230.0, (320.0, 305.0), 60.0, not opened)
        return self._state

    def start(self) -> FaceSignals:
        if not self.fake and self._thread is None:
            self._run.set()
            self._thread = threading.Thread(target=self._loop, name="face", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _landmarker(self):
        import mediapipe as mp  # noqa: F401  (loads the native library)
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    def _loop(self) -> None:
        try:
            lm = self._landmarker()
        except Exception as e:  # noqa: BLE001  (a broken install must not kill the run)
            print(f"face: FaceLandmarker unavailable ({e}); face signals stay absent")
            return
        last_t, last_ts, period = None, 0, 1.0 / self.cfg["hz"]
        while self._run.is_set():
            got = self.cameras.latest(self.name)
            now = time.time()
            if got is not None and got[1] == last_t:
                time.sleep(period / 2)
                continue
            obs = None
            if got is not None and now - got[1] < 1.0:
                last_t = got[1]
                last_ts = max(int(got[1] * 1000), last_ts + 1)
                obs = self.measure(lm, got[0], last_ts)
            self._state = self._debounce(obs, now)
            time.sleep(period)

    def measure(self, lm, bgr, ts_ms: int) -> dict | None:
        """Raw signals of one frame, or None when no face is found."""
        import cv2
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not r.face_landmarks:
            return None
        pts = [(p.x * w, p.y * h) for p in r.face_landmarks[0]]
        gap = math.dist(pts[LIP_TOP], pts[LIP_BOTTOM])
        width = math.dist(pts[CORNER_L], pts[CORNER_R])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        yaw = pitch = 0.0
        if r.facial_transformation_matrixes:
            R = r.facial_transformation_matrixes[0][:3, :3]
            pitch = math.degrees(math.atan2(-R[1, 2], R[2, 2]))
            yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
        (ux, uy), (lx, ly) = pts[LIP_TOP], pts[LIP_BOTTOM]
        mouth = ((ux + lx) / 2, (uy + ly) / 2)
        eyes = math.dist(pts[IRIS_R], pts[IRIS_L]) if len(pts) > IRIS_L else 0.0
        return {
            "mouth": mouth,
            "eyes": eyes,
            "ratio": gap / width if width > 0 else 0.0,
            "yaw": self.cfg["yaw_sign"] * yaw - self.cfg["yaw_offset_deg"],
            "pitch": self.cfg["pitch_sign"] * pitch,
            "center": ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
            "width": max(xs) - min(xs),
        }

    def _debounce(self, obs: dict | None, t: float) -> FaceState:
        c = self.cfg
        if obs is None:
            away = self._away.update(True, t)
            return FaceState(t, False, self._open.update(False, t), 0.0, 0.0, 0.0, away)
        mouth_open = self._open.update(obs["ratio"] > c["mouth_open_ratio"], t)
        away = self._away.update(abs(obs["yaw"]) > c["away_yaw_deg"], t)
        if obs["ratio"] > c["chew_hi"]:
            self._lips_apart = True
        elif obs["ratio"] < c["chew_lo"] and self._lips_apart:
            self._lips_apart = False
            self._closes.append(t)
        chewing = sum(t - tc <= c["chew_s"] for tc in self._closes) >= 2
        return FaceState(
            t,
            True,
            mouth_open,
            obs["ratio"],
            obs["yaw"],
            obs["pitch"],
            away,
            obs["center"],
            obs["width"],
            obs["mouth"],
            obs["eyes"],
            chewing,
        )
