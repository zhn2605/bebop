"""Camera models fitted from (calibration ball xyz, ball pixel) pairs.

P maps base-frame mm to undistorted pixels: [u v 1]^T ~ P [x y z 1]^T. P is scaled so that its third
row gives the depth in mm (positive in front of the camera). A fitted camera also has K and one
radial distortion term k1: project() applies it, at_height()/triangulate() remove it first.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from core.types import UV, Box, Vec3

MIN_SAMPLES = 6


def fit_camera(xyz: np.ndarray, uv: np.ndarray, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, K, dist) of a pinhole camera with square pixels, its principal point at the image
    centre and one radial distortion term, fitted to N >= 6 non-coplanar 3D-2D pairs. 8 unknowns
    (focal length, k1, pose) instead of DLT's 11, so a short arm sweep can't bend it into a
    camera that can't exist (seen live: a DLT top camera 3.8 m under the table, skew 566)."""
    import cv2

    w, h = size
    obj, img = [xyz.astype(np.float32)], [uv.astype(np.float32)]
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    best = None
    for f in (0.6 * w, 0.9 * w, 1.3 * w, 2.0 * w):  # the solver needs a focal guess; try a few
        K0 = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        try:
            rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, (w, h), K0, np.zeros(5),
                                                       flags=flags)
        except cv2.error:
            continue
        if best is None or rms < best[0]:
            best = (rms, K, dist.ravel()[:2], rv[0], tv[0])
    if best is None:
        raise ValueError("no camera fit converged")
    _, K, dist, rv, tv = best
    P = K @ np.c_[cv2.Rodrigues(rv)[0], tv]
    return P, K, dist


def look_at(pos, target, up, f_px: float, size) -> np.ndarray:
    """P = K[R|t] for a pinhole camera at `pos` looking at `target`; `up` points to the image top."""
    pos, target, up = (np.asarray(a, float) for a in (pos, target, up))
    z = target - pos
    z /= np.linalg.norm(z)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.array([x, y, z])
    K = np.array([[f_px, 0, size[0] / 2], [0, f_px, size[1] / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ pos]


FAKE_PLACEMENT = {
    "size": [640, 480],
    "f_px": 550,
    "top": {"pos": [250, 0, 700], "look_at": [250, 0, 0], "up": [1, 0, 0]},
    "side": {"pos": [250, 600, 150], "look_at": [250, 0, 60], "up": [0, 0, 1]},
}


def fake_calibration(placement: dict | None = None) -> Calibration:
    """The built-in calibration of the fake world: `top` 700 mm above the table looking down,
    `side` 600 mm to the left looking at the workspace. `placement` overrides keys of
    FAKE_PLACEMENT (the `cameras:` section of config/fake.yaml)."""
    pl = {**FAKE_PLACEMENT, **(placement or {})}
    size = pl["size"]
    P = {}
    for name in ("top", "side"):
        c = {**FAKE_PLACEMENT[name], **pl.get(name, {})}
        P[name] = look_at(c["pos"], c["look_at"], c["up"], c.get("f_px", pl["f_px"]), size)
    return Calibration(P, {n: 0.0 for n in P}, {n: list(size) for n in P})


class Calibration:
    def __init__(
        self,
        P: dict[str, np.ndarray],
        rms_px: dict[str, float] | None = None,
        size: dict[str, list[int]] | None = None,
        K: dict[str, np.ndarray] | None = None,
        dist: dict[str, list[float]] | None = None,
    ):
        self.P = {n: np.asarray(p, float) for n, p in P.items()}
        self.rms_px = rms_px or {}
        self.size = size or {}
        self.K = {n: np.asarray(k, float) for n, k in (K or {}).items()}
        self.dist = {n: [float(v) for v in d] for n, d in (dist or {}).items()}

    @property
    def names(self) -> list[str]:
        return list(self.P)

    @classmethod
    def fit(
        cls, samples: list[tuple[Vec3, dict[str, UV]]], size: dict[str, list[int]]
    ) -> Calibration:
        """One camera model per camera over the samples it saw ({cam: [w, h]} image sizes).
        Cameras with fewer than 6 are skipped."""
        names = sorted({n for _, uvs in samples for n in uvs})
        cal = cls({}, {}, dict(size))
        for n in names:
            pairs = [(xyz, uvs[n]) for xyz, uvs in samples if uvs.get(n) is not None]
            if len(pairs) < MIN_SAMPLES:
                print(f"calib: {n} has {len(pairs)} samples, need {MIN_SAMPLES}; skipped")
                continue
            xyz = np.array([p[0] for p in pairs], float)
            uv = np.array([p[1] for p in pairs], float)
            cal.P[n], cal.K[n], dist = fit_camera(xyz, uv, size[n])
            cal.dist[n] = [float(v) for v in dist]
            err = np.array([cal.project(n, p) for p in xyz]) - uv
            cal.rms_px[n] = float(np.sqrt((err ** 2).sum(axis=1).mean()))
        if not cal.P:
            raise ValueError("no camera has enough samples to fit")
        return cal

    def implausible(self, table_z: float) -> list[str]:
        """Why a fitted camera can't be the real one ([] = plausible): it sits under the table,
        its focal length or distortion is far from any webcam, or `top` doesn't look down."""
        out = []
        for n, P in self.P.items():
            centre = -np.linalg.solve(P[:, :3], P[:, 3])
            if centre[2] < table_z:
                out.append(f"{n} camera is {table_z - centre[2]:.0f} mm under the table")
            if n in self.K:
                w = (self.size.get(n) or [2 * self.K[n][0, 2]])[0]
                f = self.K[n][0, 0]
                if not 0.3 * w <= f <= 3.0 * w:
                    out.append(f"{n} focal length {f:.0f} px is not a webcam's at {w} px wide")
                if abs(self.dist[n][0]) > 1.0:
                    out.append(f"{n} lens distortion k1 {self.dist[n][0]:.2f} is far too strong")
            # P's third row (unit length) is the viewing direction; -z is straight down.
            if n == "top" and P[2, 2] > -0.5:
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, -P[2, 2]))))
                out.append(f"top camera looks {tilt:.0f} deg away from straight down")
        return out

    def _distort(self, name: str, uv: np.ndarray) -> np.ndarray:
        """Ideal (pinhole) pixels to the pixels the lens actually produces."""
        if name not in self.dist:
            return uv
        K, (k1, k2) = self.K[name], self.dist[name]
        c, f = K[:2, 2], np.diag(K)[:2]
        x = (uv - c) / f
        r2 = (x ** 2).sum(axis=-1, keepdims=True)
        return c + f * x * (1 + k1 * r2 + k2 * r2 ** 2)

    def _undistort(self, name: str, uv: UV) -> UV:
        if name not in self.dist:
            return uv
        import cv2

        d = np.array([*self.dist[name], 0.0, 0.0])
        out = cv2.undistortPoints(np.array([[uv]], float), self.K[name], d, P=self.K[name])
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def project(self, name: str, xyz: Vec3) -> UV:
        u, v = self._distort(name, _proj(self.P[name], np.asarray(xyz, float)[None]))[0]
        return float(u), float(v)

    def triangulate(self, uvs: dict[str, UV]) -> Vec3:
        """Linear least squares over every calibrated camera with a UV, reweighted by depth once."""
        rows = [(self.P[n], self._undistort(n, uv)) for n, uv in uvs.items()
                if uv is not None and n in self.P]
        if len(rows) < 2:
            raise ValueError(f"triangulate needs two cameras, got {[n for n in uvs]}")
        w = np.ones(len(rows))
        X = np.zeros(3)
        for _ in range(2):
            A, b = [], []
            for (P, (u, v)), wi in zip(rows, w):
                for r in (u * P[2] - P[0], v * P[2] - P[1]):
                    A.append(r[:3] / wi)
                    b.append(-r[3] / wi)
            X = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
            w = np.array([max(P[2] @ np.r_[X, 1.0], 1e-6) for P, _ in rows])
        return float(X[0]), float(X[1]), float(X[2])

    def at_height(self, name: str, uv: UV, z: float) -> Vec3:
        """The point on the plane at height z that projects to uv in one camera."""
        P, (u, v) = self.P[name], self._undistort(name, uv)
        r1, r2 = u * P[2] - P[0], v * P[2] - P[1]
        A = np.array([r1[:2], r2[:2]])
        b = -np.array([r1[2] * z + r1[3], r2[2] * z + r2[3]])
        x, y = np.linalg.solve(A, b)
        return float(x), float(y), float(z)

    def along_ray(self, name: str, uv: UV, other: str, other_uv: UV,
                  z_range: tuple[float, float]) -> tuple[Vec3, float]:
        """The point on `name`'s ray through uv (heights z_range, 1 mm steps) that `other` sees
        closest to other_uv, and how many px from other_uv that is. `name` fixes x and y, `other`
        only picks the height: the side camera judges height well and depth badly."""
        best, best_d = None, math.inf
        for z in np.arange(z_range[0], z_range[1] + 0.5, 1.0):
            p = self.at_height(name, uv, float(z))
            d = math.dist(self.project(other, p), other_uv)
            if d < best_d:
                best, best_d = p, d
        return best, best_d

    def width_mm(self, name: str, box: Box, z: float) -> float:
        """Width of a pixel box (along image u) in mm on the plane at height z, from the local scale
        of `project` around the box center."""
        xmin, ymin, xmax, ymax = box
        c = np.array(self.at_height(name, ((xmin + xmax) / 2, (ymin + ymax) / 2), z))
        uv0 = np.array(self.project(name, c))
        # px per mm of world x and y around the center
        J = np.column_stack([np.array(self.project(name, c + d)) - uv0 for d in np.eye(3)[:2]])
        return float(np.linalg.norm(np.linalg.solve(J, [xmax - xmin, 0.0])))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "P": {n: p.tolist() for n, p in self.P.items()},
            "K": {n: k.tolist() for n, k in self.K.items()},
            "dist": self.dist,
            "rms_px": self.rms_px,
            "size": self.size,
        }
        path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text())
        return cls(data["P"], data.get("rms_px"), data.get("size"), data.get("K"), data.get("dist"))


def _proj(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    h = P @ np.c_[xyz, np.ones(len(xyz))].T
    return (h[:2] / h[2]).T
