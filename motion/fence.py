"""Software workspace limits for the tool tip (S1, head zone)."""

from __future__ import annotations

import math

from core.types import Vec3


def _inside(box: dict | None, p: Vec3) -> bool:
    """Inside a box {x, y, z} or, with `r`, a ring {r: [min, max], z} around the base axis."""
    if not box:
        return False
    if "r" in box:
        r = math.hypot(p[0], p[1])
        return box["r"][0] - 1e-6 <= r <= box["r"][1] + 1e-6 and (
            box["z"][0] - 1e-6 <= p[2] <= box["z"][1] + 1e-6
        )
    return all(box[k][0] - 1e-6 <= v <= box[k][1] + 1e-6 for k, v in zip("xyz", p))


def to_fence_frame(p: Vec3, yaw_deg: float) -> Vec3:
    """Base frame -> the fence's frame, which is the base frame turned by `yaw_deg` of shoulder_pan
    (positive pan swings toward -y). The fence box sits wherever the plate is, not only straight
    ahead of the base."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x, y, z = p
    return (x * c - y * s, x * s + y * c, z)


def from_fence_frame(p: Vec3, yaw_deg: float) -> Vec3:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x, y, z = p
    return (x * c + y * s, -x * s + y * c, z)


class Fence:
    """`acquire`: inside the fence box, outside the head zone. `deliver`: also the stage -> mouth
    corridor. `transit` (joint moves to taught poses): anywhere above the table outside the head
    zone, plus the corridor, which needs `set_segment(stage_xyz, mouth_xyz)` (Motion sets it)."""

    def __init__(self, cfg: dict):
        self.box = cfg["fence"]
        self.yaw = float(cfg.get("fence_yaw_deg", 0))
        self.head = cfg.get("head_zone")
        self.radius = float(cfg.get("deliver_radius_mm", 40))
        self.table_z = float(cfg.get("table_z", 0))
        self.segment: tuple[Vec3, Vec3] | None = None

    def set_segment(self, stage: Vec3, mouth: Vec3) -> None:
        self.segment = (tuple(stage), tuple(mouth))

    def around_mouth(self, stage: Vec3, mouth: Vec3, half_mm: float = 110.0) -> None:
        """Head zone and corridor from a tracked mouth: a box round the head (chin to crown) that
        only the stage -> mouth corridor enters, and never past the mouth point."""
        x, y, z = mouth
        self.head = {"x": [x - half_mm, x + half_mm], "y": [y - half_mm, y + half_mm],
                     "z": [z - 90.0, z + 200.0]}
        self.set_segment(stage, mouth)

    def in_corridor(self, p: Vec3) -> bool:
        if not self.segment:
            return False
        a, b = self.segment
        ab = [b[i] - a[i] for i in range(3)]
        n2 = sum(v * v for v in ab) or 1e-9
        s = sum((p[i] - a[i]) * ab[i] for i in range(3)) / n2
        if s > 1 + 1e-9:  # flat end: never past the mouth point (S1)
            return False
        s = max(s, 0.0)
        return math.dist(p, [a[i] + s * ab[i] for i in range(3)]) <= self.radius

    def check(self, xyz: Vec3, stage: str = "acquire") -> str | None:
        """None when the tip may be at `xyz` in this stage, else the reason."""
        x, y, z = xyz
        # 15 mm of slack: after a descent the measured tip sags a little under the floor, and a
        # move away from there must not be refused (seen live: every move after a descend failed).
        if z < self.table_z - 15.0:
            return f"z {z:.0f} below the table"
        if stage in ("deliver", "transit") and self.in_corridor(xyz):
            return None
        if _inside(self.head, xyz):
            return f"({x:.0f}, {y:.0f}, {z:.0f}) in the head zone"
        if stage == "transit":
            return None
        if not _inside(self.box, to_fence_frame(xyz, self.yaw)):
            what = "reach ring" if "r" in self.box else "fence box"
            return f"({x:.0f}, {y:.0f}, {z:.0f}) outside the {what}"
        return None
