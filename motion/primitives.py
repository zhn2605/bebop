"""Fenced, speed-capped arm actions. Each returns an ActionResult (keys: CORE_V2.md, Metrics).

On a stop the action raises `Stopped` with the partial result attached as `e.result`.
`tip_uv`/`marker_uv`/`px`/`mm_from_marker` are added by the orchestrator, not here.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable

from core.types import GRIPPER, JOINTS, ActionResult, Pose, Stopped, TrackResult, Unreachable
from motion.kinematics import approach


def _lerp(a, b, s: float) -> tuple:
    return tuple(a[i] + (b[i] - a[i]) * s for i in range(len(a)))


def _r(v):
    return [round(float(x), 1) for x in v] if isinstance(v, (tuple, list)) else round(float(v), 3)


def _action(fn):
    """Run `fn(self, r, ...)` with a fresh ActionResult `r`; stamp times; attach it to Stopped."""

    @functools.wraps(fn)
    def run(self, *args, **kwargs) -> ActionResult:
        r = ActionResult(fn.__name__, True, t0=time.time())
        try:
            fn(self, r, *args, **kwargs)
        except Stopped as e:
            r.ok, r.stopped = False, True
            r.note = f"{r.note}; {e}" if r.note else str(e)
            e.result = r
            raise
        finally:
            r.t1 = time.time()
        return r

    return run


class Motion:
    def __init__(self, arm, kin, fence, cfg: dict, poses: dict, pause=None):
        """`cfg` is workspace.yaml; robot settings come from `arm.cfg`. `pause`: the pause Flag."""
        self.arm, self.kin, self.fence, self.cfg = arm, kin, fence, cfg
        self.poses = poses or {}
        self.pause = pause
        self.bowl_rim_z = float(cfg.get("bowl_rim_z", 40))
        self.grasp_clearance = float(cfg.get("grasp_clearance_mm", 8))
        self.contact_load = float(arm.cfg.get("contact_load", 0.25))
        self.track_err = float(arm.cfg.get("track_err_deg", 6))
        self._segment()

    @property
    def tip(self) -> Pose:
        return self.arm.state().tip

    # --- helpers ----------------------------------------------------------------------------------

    def _speed(self, speed: str) -> float:
        return self.arm.speed * (self.arm.slow_factor if speed == "slow" else 1.0)

    def _segment(self) -> None:
        """Deliver corridor from the taught stage and mouth poses with the active tool."""
        if "stage" in self.poses and "mouth" in self.poses:
            tool = self.arm.tool
            self.fence.set_segment(
                self.kin.fk(self.poses["stage"], tool).xyz,
                self.kin.fk(self.poses["mouth"], tool).xyz,
            )

    def _path_reason(self, q0: dict, q1: dict, stage: str, n: int = 10) -> str | None:
        """Fence-check the tip along the joint-space path q0 -> q1 (what move_joints will do)."""
        for i in range(1, n + 1):
            q = {j: q0[j] + (q1[j] - q0[j]) * i / n for j in JOINTS}
            why = self.fence.check(self.kin.fk(q, self.arm.tool).xyz, stage)
            if why:
                return why
        return None

    def _go(
        self, r: ActionResult, pose: Pose, speed: str, stage: str, gate=None
    ) -> TrackResult | None:
        """Fence-check the target and 5 points on the straight tip path, IK, joint interpolation
        (via
        the 5 points when the joint path itself would leave the fence). None = refused, see r.note.
        When the start is outside the stage's zone (after a transit move) the path points are only
        held to the transit rule."""
        st = self.arm.state()
        a = st.tip
        start_ok = self.fence.check(a.xyz, stage) is None
        for i in range(1, 7):
            p = _lerp(a.xyz, pose.xyz, i / 6)
            why = self.fence.check(p, stage if (start_ok or i == 6) else "transit")
            if why:
                r.ok, r.note = False, f"fence: {why}"
                return None
        try:
            q_end, used = self.kin.ik_any_pitch(pose, self.arm.tool, st.q)
            legs = [q_end]
            if self._path_reason(st.q, q_end, stage if start_ok else "transit"):
                legs, seed = [], st.q
                for i in range(1, 7):
                    s = i / 6
                    wp = Pose(
                        _lerp(a.xyz, used.xyz, s),
                        a.pitch + (used.pitch - a.pitch) * s,
                        a.roll + (used.roll - a.roll) * s,
                    )
                    seed, _ = self.kin.ik_any_pitch(wp, self.arm.tool, seed)
                    legs.append(seed)
        except Unreachable as e:
            r.ok, r.note = False, f"unreachable: {e}"
            return None
        if used.pitch != pose.pitch:
            r.predicted["pitch"] = _r(used.pitch)
            note = f"pitch {pose.pitch:.0f} unreachable here, used {used.pitch:.0f}"
            if note not in r.note:
                r.note = f"{r.note}; {note}" if r.note else note
        err, load, dur = 0.0, 0.0, 0.0
        for q in legs:
            tr = self.arm.move_joints(q, 0.0, self.pause, self._speed(speed), gate)
            err, load, dur = tr.max_err_deg, max(load, tr.max_load), dur + tr.duration_s
            if tr.stopped:
                return TrackResult(err, load, dur, stopped=True)
        return TrackResult(err, load, dur)

    def _up(self, a: Pose, dz: float) -> Pose:
        """`a` raised by dz, the pitch relaxed toward level until IK reaches it (high up, pointing
        straight down runs out of wrist_flex)."""
        xyz = (a.xyz[0], a.xyz[1], a.xyz[2] + dz)
        for pitch in (a.pitch, -75, -60, -45, -30, 0):
            if pitch >= a.pitch:
                try:
                    self.kin.ik(Pose(xyz, pitch, a.roll), self.arm.tool)
                    return Pose(xyz, pitch, a.roll)
                except Unreachable:
                    pass
        return Pose(xyz, a.pitch, a.roll)

    def _measure(self, r: ActionResult, tr: TrackResult, target, contact_rule: bool = True) -> None:
        tip = self.arm.state().tip
        r.measured.update(
            tip_xyz=_r(tip.xyz), track_err_deg=_r(tr.max_err_deg), max_load=_r(tr.max_load)
        )
        if target is not None:
            r.error["xyz_mm"] = _r(math.dist(tip.xyz, target))
        if contact_rule and tr.max_err_deg > self.track_err and tr.max_load > self.contact_load:
            r.ok = False
            r.note = f"{r.note}; contact" if r.note else "contact"

    # --- actions ----------------------------------------------------------------------------------

    @_action
    def move_to(
        self, r: ActionResult, pose: Pose, speed: str = "normal", stage: str = "acquire"
    ) -> None:
        r.predicted.update(tip_xyz=_r(pose.xyz), pitch=_r(pose.pitch))
        tr = self._go(r, pose, speed, stage)
        if tr is not None:
            self._measure(r, tr, pose.xyz)

    @_action
    def descend_until_contact(
        self,
        r: ActionResult,
        max_dz: float,
        load: float | None = None,
        expect_z: float | None = None,
    ) -> None:
        """Down 5 mm at a time at slow speed until a joint load rises `load` (default contact_load)
        above its value at the start, `max_dz` is done, or the tip is at the fence floor (the last
        step is clamped to it). `expect_z` = predicted contact height."""
        limit = float(load if load is not None else self.contact_load)
        st = self.arm.state()
        a, base = st.tip, st.load
        floor = max(self.fence.table_z + self.grasp_clearance, float(self.fence.box["z"][0]))
        r.predicted["contact_z"] = _r(expect_z if expect_z is not None
                                      else a.xyz[2] + max_dz * approach(a.pitch, 0.0)[2])
        # Along the way the tool points (straight down at pitch -90), so a tilted gripper slides in
        # toward the food instead of dropping sideways onto it. dz = distance along that axis.
        d = approach(a.pitch, math.atan2(a.xyz[1], a.xyz[0]))
        down = max(-d[2], 0.2)  # height lost per mm along the axis
        dz, peak, contact, at_floor = 0.0, 0.0, False, False
        while dz < max_dz - 1e-6:
            step = min(5.0, max_dz - dz, (a.xyz[2] - floor) / down - dz)
            if step <= 1e-6:
                at_floor = True
                break
            s = dz + step
            tr = self._go(
                r,
                Pose(tuple(a.xyz[i] + d[i] * s for i in range(3)), a.pitch, a.roll),
                "slow",
                "acquire",
            )
            if tr is None:
                break
            dz += step
            now = self.arm.state().load
            joint = max(JOINTS, key=lambda j: abs(now[j] - base[j]))
            rise = abs(now[joint] - base[joint])
            peak = max(peak, rise)
            if rise > limit:
                contact = True
                break
        z = self.arm.state().tip.xyz[2]
        r.measured.update(contact_z=_r(z), max_load=_r(peak), contact=contact, dz=_r(dz))
        if contact:
            r.measured["contact_joint"] = joint
        r.error["z_mm"] = _r(z - r.predicted["contact_z"])
        if not contact and r.ok:
            r.note = "reached floor, no contact" if at_floor else "no contact"

    @_action
    def open(self, r: ActionResult, width_mm: float) -> None:
        top = float(self.arm.grip.get("open_mm_at_100", 60))
        r.predicted["width_mm"] = _r(width_mm)
        if width_mm > top:
            r.note = f"clamped to {top:g} mm"
        got = self.arm.set_gripper_mm(min(width_mm, top))
        r.measured["width_mm"] = _r(got)
        r.error["mm"] = _r(got - width_mm)
        r.ok = abs(got - width_mm) <= 5 or width_mm > top

    @_action
    def close(self, r: ActionResult, expect_mm: float | None = None) -> None:
        """Close on the object. `expect_mm` = the object width the orchestrator predicted."""
        got = self.arm.close_gripper()
        r.predicted["width_mm"] = _r(expect_mm) if expect_mm is not None else None
        r.measured.update(width_mm=_r(got), load=_r(self.arm.state().load[GRIPPER]))
        r.ok = got >= 3
        if expect_mm is not None:
            r.error["mm"] = _r(got - expect_mm)
            # VLM boxes run loose, so a narrower grasp than predicted is fine; only something much
            # wider than the object (the bowl rim, the spoon) fails.
            if got > expect_mm * 1.3 + 12:
                r.ok, r.note = False, "closed on something wider than the object"
        if got < 3:
            r.note = "closed on nothing"

    @_action
    def lift(self, r: ActionResult, dz: float | None = None) -> None:
        pose = self._up(self.tip, float(dz if dz is not None else self.cfg.get("hover_dz", 40)))
        r.predicted.update(tip_xyz=_r(pose.xyz), pitch=_r(pose.pitch))
        tr = self._go(r, pose, "normal", "acquire")
        if tr is not None:
            self._measure(r, tr, pose.xyz)

    @_action
    def goto(self, r: ActionResult, name: str, speed: str = "normal") -> None:
        """Joint move to the taught pose `name` (tools.teach --goto)."""
        self._goto(r, name, speed)

    def _goto(self, r: ActionResult, name: str, speed: str = "normal") -> None:
        """Joint move to a taught pose; the tip path is held to the transit rule."""
        q = self.poses.get(name)
        if not q:
            r.ok, r.note = False, f"pose '{name}' not taught (python -m tools.teach)"
            return
        q = {j: float(q[j]) for j in JOINTS}
        target = self.kin.fk(q, self.arm.tool).xyz
        r.predicted.update(tip_xyz=_r(target))
        why = self._path_reason(self.arm.state().q, q, "transit")
        if why:
            r.ok, r.note = False, f"fence: {why}"
            return
        tr = self.arm.move_joints(q, 0.0, self.pause, self._speed(speed))
        self._measure(r, tr, target)

    @_action
    def home(self, r: ActionResult, speed: str = "normal") -> None:
        self._goto(r, "home", speed)

    @_action
    def stage(self, r: ActionResult, speed: str = "normal") -> None:
        self._segment()
        self._goto(r, "stage", speed)

    @_action
    def retract(self, r: ActionResult) -> None:
        """Back along the deliver corridor to `stage` when in it, else up by retract_dz; home."""
        self._segment()
        a = self.tip
        if (
            "stage" in self.poses
            and self.fence.in_corridor(a.xyz)
            and self.fence.check(a.xyz, "acquire")
        ):
            s = self.kin.fk(self.poses["stage"], self.arm.tool)
            tr = self._go(r, s, "slow", "deliver")
        else:
            tr = self._go(
                r, self._up(a, float(self.cfg.get("retract_dz", 60))), "normal", "transit"
            )
            if tr is None and "unreachable" in r.note:
                # Lifting straight up can run the wrist out of range (seen live from home);
                # the joint move home is still safe, so take it instead of failing.
                r.ok, r.note = True, "up pose unreachable, went home directly"
                self._goto(r, "home")
                return
        if tr is None:
            return
        self._goto(r, "home")

    @_action
    def present(
        self,
        r: ActionResult,
        gate: Callable[[], str | None],
        dist: Callable[[], float | None] | None = None,
    ) -> None:
        """From `stage` toward `mouth` at slow speed, stopping stop_short_mm before the mouth or
        when
        gate() returns a reason. `dist()` gives the ultrasonic reading for the result."""
        pc = self.cfg.get("present", {})
        short = float(pc.get("stop_short_mm", 50))
        r.predicted["stop_short_mm"] = short
        if "stage" not in self.poses or "mouth" not in self.poses:
            r.ok, r.note = False, "stage/mouth not taught (python -m tools.teach)"
            return
        self._segment()
        tool = self.arm.tool
        s, m = self.kin.fk(self.poses["stage"], tool), self.kin.fk(self.poses["mouth"], tool)
        if math.dist(self.tip.xyz, s.xyz) > 30:
            r.ok, r.note = False, "not at stage; run stage() first"
            return
        L = math.dist(s.xyz, m.xyz)
        target = _lerp(s.xyz, m.xyz, max(0.0, (L - short) / L)) if L > 0 else s.xyz
        tr = self._go(r, Pose(target, m.pitch, m.roll), pc.get("speed", "slow"), "deliver", gate)
        if tr is None:
            return
        self._measure(r, tr, target)
        r.measured.update(
            dist_cm=dist() if dist else None,
            stopped_by=self.arm.halt_reason or "stop_short",
            mm_to_mouth=_r(math.dist(self.tip.xyz, m.xyz)),
        )

    @_action
    def scoop(self, r: ActionResult, depth_mm: float, drag_mm: float, tilt_deg: float) -> None:
        """From a hover over the target: pitch down, dip depth_mm below bowl_rim_z, drag drag_mm
        toward
        the base, tilt up by tilt_deg, lift back to the hover height."""
        a = self.tip
        x0, y0, z0 = a.xyz
        pitch = float(self.cfg.get("scoop_pitch", -60))
        zd = self.bowl_rim_z - depth_mm
        rho = math.hypot(x0, y0) or 1.0
        x1, y1 = x0 - x0 / rho * drag_mm, y0 - y0 / rho * drag_mm
        r.predicted.update(depth_mm=_r(depth_mm), dip_z=_r(zd))
        base = self.arm.state().load["wrist_flex"]
        legs = [
            (Pose((x0, y0, z0), pitch, a.roll), "normal", False),
            (Pose((x0, y0, zd), pitch, a.roll), "slow", True),
            (Pose((x1, y1, zd), pitch, a.roll), "slow", True),
            (Pose((x1, y1, zd), pitch + tilt_deg, a.roll), "slow", True),
            (Pose((x1, y1, z0), pitch + tilt_deg, a.roll), "normal", False),
        ]
        peak = 0.0
        for pose, speed, dip in legs:
            tr = self._go(r, pose, speed, "acquire")
            if tr is None:
                break
            if dip:
                peak = max(peak, self.arm.peak_load.get("wrist_flex", 0.0))
            else:
                self._measure(r, tr, None)
        r.measured.update(
            peak_wrist_load=_r(peak), contact=abs(peak - abs(base)) > self.contact_load
        )
