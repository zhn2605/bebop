"""Analytic FK/IK for the SO-101: pan yaw plus a planar shoulder/elbow/wrist chain.

Joint angles in and out are LeRobot degrees (use_degrees=True). Internally they map to URDF degrees
(`so101_new_calib.urdf`, q = 0 is the "L" pose: upper arm straight up, forearm and gripper straight
out) with `urdf = sign * (lerobot - zero_offset_deg)` per joint. URDF signs: +pan swings the arm to
the right (-y), +shoulder_lift tips the upper arm forward, +elbow_flex and +wrist_flex bend down.
"""

from __future__ import annotations

import math

from core.types import JOINTS, Pose, Unreachable, Vec3


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def approach(pitch: float, yaw: float) -> Vec3:
    """Unit vector the tool points along: pitch in degrees (-90 = down), yaw in radians (the pan
    direction, atan2(y, x) of the tip)."""
    p = math.radians(pitch)
    return (math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), math.sin(p))


class Kinematics:
    def __init__(self, geometry: dict, tool_offsets: dict, joints: dict | None = None):
        g = geometry
        self.base_z = float(g.get("base_z", 0.0))
        self.sr, self.sz = (float(v) for v in g["shoulder"])
        ux, uz = (float(v) for v in g["upper_arm"])
        fx, fz = (float(v) for v in g["forearm"])
        self.l1, self.a1 = math.hypot(ux, uz), math.degrees(math.atan2(uz, ux))
        self.l2, self.a2 = math.hypot(fx, fz), math.degrees(math.atan2(fz, fx))
        self.grip = [float(v) for v in g["gripper_frame"]]
        self.tools = {k: [float(v) for v in t] for k, t in tool_offsets.items()}
        joints = joints if joints is not None else g.get("joints", {})
        self.zero = {j: float(joints.get(j, {}).get("zero_offset_deg", 0.0)) for j in JOINTS}
        self.sign = {j: float(joints.get(j, {}).get("sign", 1)) for j in JOINTS}
        self.limits = {
            j: (float(joints[j]["min"]), float(joints[j]["max"]))
            for j in JOINTS
            if "min" in joints.get(j, {})
        }

    def to_urdf(self, q: dict) -> dict:
        return {j: self.sign[j] * (q[j] - self.zero[j]) for j in JOINTS}

    def from_urdf(self, u: dict) -> dict:
        return {j: self.sign[j] * u[j] + self.zero[j] for j in JOINTS}

    def check_limits(self, q: dict) -> str | None:
        for j, (lo, hi) in self.limits.items():
            if not lo - 1e-6 <= q[j] <= hi + 1e-6:
                return f"{j} {q[j]:.1f} outside [{lo:g}, {hi:g}]"
        return None

    def reach(self, tool: str = "jaw") -> float:
        """Horizontal distance from the pan axis to the tip with the arm stretched out level, mm."""
        return self.sr + self.l1 + self.l2 + self._tool(tool)[0]

    def _tool(self, tool: str) -> list[float]:
        t = self.tools[tool]
        return [self.grip[i] + t[i] for i in range(3)]

    def _rolled(self, tool: str, roll: float) -> tuple[float, float, float]:
        tx, ty, tz = self._tool(tool)
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        return tx, ty * c + tz * s, -ty * s + tz * c

    def fk(self, q: dict, tool: str = "jaw") -> Pose:
        u = self.to_urdf(q)
        lift, elbow = u["shoulder_lift"], u["elbow_flex"]
        t1, t2 = math.radians(self.a1 - lift), math.radians(self.a2 - lift - elbow)
        pitch = -(lift + elbow + u["wrist_flex"])
        p = math.radians(pitch)
        tx, ty, tz = self._rolled(tool, u["wrist_roll"])
        r = (
            self.sr
            + self.l1 * math.cos(t1)
            + self.l2 * math.cos(t2)
            + tx * math.cos(p)
            - tz * math.sin(p)
        )
        z = (
            self.sz
            + self.l1 * math.sin(t1)
            + self.l2 * math.sin(t2)
            + tx * math.sin(p)
            + tz * math.cos(p)
        )
        yaw = math.radians(-u["shoulder_pan"])
        x = r * math.cos(yaw) - ty * math.sin(yaw)
        y = r * math.sin(yaw) + ty * math.cos(yaw)
        return Pose((x, y, z + self.base_z), _wrap(pitch), u["wrist_roll"])

    def chain(self, q: dict, tool: str = "jaw") -> list[Vec3]:
        """Base (table under the pan axis), shoulder, elbow, wrist and tool tip, base frame mm."""
        u = self.to_urdf(q)
        lift, elbow = u["shoulder_lift"], u["elbow_flex"]
        t1, t2 = math.radians(self.a1 - lift), math.radians(self.a2 - lift - elbow)
        yaw = math.radians(-u["shoulder_pan"])
        pts, r, z = [(0.0, 0.0, 0.0)], self.sr, self.sz
        for dr, dz in (
            (0.0, 0.0),
            (self.l1 * math.cos(t1), self.l1 * math.sin(t1)),
            (self.l2 * math.cos(t2), self.l2 * math.sin(t2)),
        ):
            r, z = r + dr, z + dz
            pts.append((r * math.cos(yaw), r * math.sin(yaw), z + self.base_z))
        return [*pts, self.fk(q, tool).xyz]

    def ik_any_pitch(
        self, pose: Pose, tool: str = "jaw", q_seed: dict | None = None,
        lo: float = -120.0, hi: float = 0.0,
    ) -> tuple[dict, Pose]:
        """IK at `pose`, else at the tool pitch nearest it (1 deg steps within [lo, hi]) that
        reaches the point: the pitch is a preference, never a reason not to move. Returns the
        joints and the pose actually used. Unreachable only when no pitch reaches the point."""
        try:
            return self.ik(pose, tool, q_seed), pose
        except Unreachable as e:
            first = e
        for k in range(1, int(hi - lo) + 1):
            for p in (pose.pitch - k, pose.pitch + k):
                if lo <= p <= hi:
                    cand = Pose(pose.xyz, float(p), pose.roll)
                    try:
                        return self.ik(cand, tool, q_seed), cand
                    except Unreachable:
                        continue
        raise Unreachable(f"{first}; no tool pitch from {lo:.0f} to {hi:.0f} reaches it")

    def ik(self, pose: Pose, tool: str = "jaw", q_seed: dict | None = None) -> dict:
        """Joint solution (LeRobot deg) inside the limits, closest to `q_seed`, else elbow-up."""
        x, y, z = pose.xyz
        z -= self.base_z
        tx, ty, tz = self._rolled(tool, pose.roll)
        rho = math.hypot(x, y)
        if rho <= abs(ty) + 1e-6:
            raise Unreachable(f"{pose.xyz} too close to the pan axis")
        r = math.sqrt(rho * rho - ty * ty)
        pan = -math.degrees(math.atan2(y, x) - math.atan2(ty, r))
        p = math.radians(pose.pitch)
        a = r - (tx * math.cos(p) - tz * math.sin(p)) - self.sr
        b = z - (tx * math.sin(p) + tz * math.cos(p)) - self.sz
        c = (a * a + b * b - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if abs(c) > 1:
            raise Unreachable(
                f"{pose.xyz} pitch {pose.pitch:.0f}: wrist {math.hypot(a, b):.0f} mm from the "
                f"shoulder, max {self.l1 + self.l2:.0f}"
            )
        sols, why = [], ""
        for d in (-math.acos(c), math.acos(c)):
            t1 = math.atan2(b, a) - math.atan2(
                self.l2 * math.sin(d), self.l1 + self.l2 * math.cos(d)
            )
            lift = _wrap(self.a1 - math.degrees(t1))
            elbow = _wrap(self.a2 - self.a1 - math.degrees(d))
            u = {
                "shoulder_pan": _wrap(pan),
                "shoulder_lift": lift,
                "elbow_flex": elbow,
                "wrist_flex": _wrap(-pose.pitch - lift - elbow),
                "wrist_roll": pose.roll,
            }
            q = self.from_urdf(u)
            bad = self.check_limits(q)
            if bad:
                why = why or bad
            else:
                sols.append(q)
        if not sols:
            raise Unreachable(f"{pose.xyz} pitch {pose.pitch:.0f}: {why}")
        if q_seed:
            return min(sols, key=lambda s: sum((s[j] - q_seed.get(j, s[j])) ** 2 for j in JOINTS))
        return sols[0]
