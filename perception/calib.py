"""Camera projection matrices fitted by DLT from (tip xyz, marker pixel) pairs.

P maps base-frame mm to pixels: [u v 1]^T ~ P [x y z 1]^T. P is scaled so that its third row gives
the depth in mm (positive in front of the camera).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from core.types import UV, Box, Vec3

MIN_SAMPLES = 6


def _normalizer(pts: np.ndarray) -> np.ndarray:
    """Hartley similarity: centroid to the origin, mean distance sqrt(dim)."""
    dim = pts.shape[1]
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = math.sqrt(dim) / d if d > 0 else 1.0
    T = np.eye(dim + 1)
    T[:dim, :dim] *= s
    T[:dim, dim] = -s * c
    return T


def dlt(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """3x4 P from N >= 6 non-coplanar 3D-2D pairs (normalized DLT)."""
    T3, T2 = _normalizer(xyz), _normalizer(uv)
    X = (T3 @ np.c_[xyz, np.ones(len(xyz))].T).T
    x = (T2 @ np.c_[uv, np.ones(len(uv))].T).T
    A = []
    for Xh, (u, v, _) in zip(X, x):
        A.append(np.r_[Xh, np.zeros(4), -u * Xh])
        A.append(np.r_[np.zeros(4), Xh, -v * Xh])
    _, _, Vt = np.linalg.svd(np.array(A))
    P = np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3
    P /= np.linalg.norm(P[2, :3])
    if np.median(P[2] @ np.c_[xyz, np.ones(len(xyz))].T) < 0:
        P = -P
    return P


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
    ):
        self.P = {n: np.asarray(p, float) for n, p in P.items()}
        self.rms_px = rms_px or {}
        self.size = size or {}

    @property
    def names(self) -> list[str]:
        return list(self.P)

    @classmethod
    def fit(cls, samples: list[tuple[Vec3, dict[str, UV]]]) -> Calibration:
        """DLT per camera over the samples that camera saw. Cameras with fewer than 6 are skipped."""
        names = sorted({n for _, uvs in samples for n in uvs})
        P, rms = {}, {}
        for n in names:
            pairs = [(xyz, uvs[n]) for xyz, uvs in samples if uvs.get(n) is not None]
            if len(pairs) < MIN_SAMPLES:
                print(f"calib: {n} has {len(pairs)} samples, need {MIN_SAMPLES}; skipped")
                continue
            xyz = np.array([p[0] for p in pairs], float)
            uv = np.array([p[1] for p in pairs], float)
            P[n] = dlt(xyz, uv)
            rms[n] = float(np.sqrt((((_proj(P[n], xyz) - uv) ** 2).sum(axis=1)).mean()))
        if not P:
            raise ValueError("no camera has enough samples to fit")
        return cls(P, rms)

    def project(self, name: str, xyz: Vec3) -> UV:
        u, v = _proj(self.P[name], np.asarray(xyz, float)[None])[0]
        return float(u), float(v)

    def triangulate(self, uvs: dict[str, UV]) -> Vec3:
        """Linear least squares over every calibrated camera with a UV, reweighted by depth once."""
        rows = [(self.P[n], uv) for n, uv in uvs.items() if uv is not None and n in self.P]
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
        P, (u, v) = self.P[name], uv
        r1, r2 = u * P[2] - P[0], v * P[2] - P[1]
        A = np.array([r1[:2], r2[:2]])
        b = -np.array([r1[2] * z + r1[3], r2[2] * z + r2[3]])
        x, y = np.linalg.solve(A, b)
        return float(x), float(y), float(z)

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
            "rms_px": self.rms_px,
            "size": self.size,
        }
        path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text())
        return cls(data["P"], data.get("rms_px"), data.get("size"))


def _proj(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    h = P @ np.c_[xyz, np.ones(len(xyz))].T
    return (h[:2] / h[2]).T
