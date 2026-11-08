"""SO-101 arm through LeRobot (imported lazily) or a fake: min-jerk joint moves gated by Flags.

Angles are LeRobot degrees (use_degrees=True); the gripper is LeRobot's 0 (closed) .. 100 (open)
inside `move_joints` and millimetres between the jaw tips everywhere else.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from core.types import GRIPPER, JOINTS, ArmState, Flag, Stopped, TrackResult
from motion.kinematics import Kinematics

ROOT = Path(__file__).resolve().parents[1]
MOTORS = (*JOINTS, GRIPPER)


def _exit_signal(signum: int, frame) -> None:
    if signum == signal.SIGHUP:  # the terminal is gone: a print on the fold-to-rest path would raise
        sys.stdout = sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    raise KeyboardInterrupt


def interrupt_on_exit_signals() -> None:
    """SIGTERM (kill) and SIGHUP (a closed terminal) raise KeyboardInterrupt like Ctrl+C, so every
    entry point's Ctrl+C path folds the arm to rest instead of dropping it. Main thread only."""
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, _exit_signal)


def find_port(port: str) -> str:
    """`port`, or the first match when it is a glob: the adapter's by-id name carries its serial
    number, which changes when the board is swapped (seen live)."""
    if "*" not in port:
        return port
    import glob

    found = sorted(glob.glob(port))
    if not found:
        raise ConnectionError(f"no arm adapter matches {port}: is the arm's USB plugged in?")
    return found[0]


def decode_load(raw: int) -> float:
    """Present_Load -> signed fraction of max torque. LeRobot 0.4.4 already returns it signed
    (verified on the arm: the elbow read -96 while holding), older versions return the raw
    sign-magnitude register with bit 10 as the direction, so a value >= 1024 is decoded that way."""
    raw = int(raw)
    if raw >= 0x400:
        raw = -(raw & 0x3FF)
    return max(-1.0, min(raw / 1000.0, 1.0))


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class Arm:
    def __init__(
        self, cfg: dict, stop: Flag, fake: bool | dict = False, kin: Kinematics | None = None
    ):
        """`fake=True` loads config/fake.yaml; a dict is used as that file's contents."""
        self.cfg, self.stop = cfg, stop
        if fake is True:
            fake = yaml.safe_load((ROOT / "config" / "fake.yaml").read_text()) or {}
        self.fake: dict | None = fake or None
        self.kin = kin or Kinematics(cfg["geometry"], cfg["tool_offsets"], cfg.get("joints"))
        self.tool = "jaw"
        self.hz = float(cfg.get("hz", 50))
        self.speed = float(cfg.get("speed_deg_s", 60))
        self.slow_factor = float(cfg.get("slow_factor", 0.4))
        self.grip = cfg.get("gripper", {})
        self.frozen = False
        self.torque_on = False
        self.halt_reason = ""  # why the last move_joints gate halted it
        self.peak_load: dict[str, float] = {}  # worst |load| per motor during the last move_joints
        self.robot = None
        self._lock = threading.RLock()
        self._goal: dict[str, float] = {}
        self._pos: dict[str, float] = {}  # fake present position
        self._holding = False  # fake: gripper closed on the food

    # --- connection -------------------------------------------------------------------------------

    def connect(self) -> None:
        interrupt_on_exit_signals()
        if self.fake:
            fa = self.fake.get("arm", {})
            start = self.fake.get("poses", {}).get(fa.get("start", "home"), {})
            self._pos = {m: float(start.get(m, 0.0)) for m in MOTORS}
            self._pos[GRIPPER] = self.pct(float(fa.get("gripper_mm", 0.0)))
            self._goal, self.torque_on = dict(self._pos), True
            return
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # lerobot >= 0.4.3
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cal_dir = ROOT / self.cfg.get("calibration_dir", "calibration")
        rid = self.cfg.get("id", "feeder_arm")
        if not (cal_dir / f"{rid}.json").is_file():
            raise FileNotFoundError(
                f"no calibration {cal_dir / (rid + '.json')}; run lerobot-calibrate first"
            )
        rc = SO101FollowerConfig(
            port=find_port(self.cfg["port"]),
            id=rid,
            calibration_dir=cal_dir,
            use_degrees=True,
            max_relative_target=float(self.cfg.get("max_relative_target", 5.0)),
        )
        for attempt in range(3):
            self.robot = SO101Follower(rc)
            try:
                self.robot.connect(calibrate=False)
                break
            except (RuntimeError, ConnectionError) as e:
                # At 5 V, torque coming on in all six servos at once sags the supply under their
                # 4.5 V limit ("Input voltage error", seen live); a second try usually passes.
                if "voltage" not in str(e).lower() or attempt == 2:
                    raise
                print(f"servo supply sagged while connecting ({str(e)[-40:]}); retrying")
                with contextlib.suppress(Exception):
                    self.robot.bus.port_handler.closePort()
                time.sleep(1.0)
        bus = self.robot.bus
        if not self.robot.is_calibrated:  # servos disagree with the file: write the file, no prompt
            with bus.torque_disabled():
                bus.write_calibration(self.robot.calibration)
        # LeRobot sets P=16 against shakiness; at 5 V that leaves the lift sagging several degrees
        # under gravity (seen live: goal -79, held -84). Raise the gains per robot.yaml.
        for name, value in self.cfg.get("pid", {}).items():
            for m in (*JOINTS, GRIPPER):
                bus.write(name, m, int(value))
        # LeRobot pretty-prints every goal it clamps to max_relative_target; our interpolation
        # already limits speed, so that log is noise.
        logging.getLogger("lerobot").setLevel(logging.ERROR)
        self.torque(True)

    def disconnect(self, rest: bool = True) -> None:
        """Fold slowly into robot.yaml's `rest_pose` (the pose the arm settles into without
        torque, so nothing drops when torque goes off), then torque off and close the port. Also
        after a stop: torque off without the fold drops the arm."""
        if rest and self.robot is not None and self.torque_on:
            self.stop.clear()
            pose = self.cfg.get("rest_pose")
            if pose:
                try:
                    self.move_joints(dict(pose), 0.0, speed_deg_s=self.speed * self.slow_factor)
                except Exception as e:  # noqa: BLE001  (torque off is still the right next step)
                    print(f"rest move failed: {e}")
        self.torque_on = False
        if self.robot is not None:
            with self._lock:
                self.robot.disconnect()
            self.robot = None

    def torque(self, on: bool, motors: list[str] | None = None) -> None:
        """On: goal = present first so nothing jumps, then torque and caps. Off: the arm sags.
        `motors` limits it to those servos (teaching by hand keeps the gripper holding)."""
        if self.robot is None:
            self.torque_on = on if motors is None else self.torque_on
            return
        bus = self.robot.bus
        with self._lock:
            if on:
                pos = bus.sync_read("Present_Position")
                bus.sync_write("Goal_Position", pos)
                self._goal = dict(pos)
                bus.enable_torque(motors)
                for m, v in self.cfg.get("torque_caps", {}).items():
                    bus.write("Torque_Limit", m, int(v))
            else:
                bus.disable_torque(motors)
        if motors is None:
            self.torque_on = on

    # --- low level --------------------------------------------------------------------------------

    def _read(self) -> tuple[dict[str, float], dict[str, float]]:
        """Present position (deg, gripper 0..100) and signed load fraction for every motor."""
        if self.fake:
            load = {m: 0.0 for m in MOTORS}
            if self._fake_contact():
                load["wrist_flex"], load["shoulder_lift"] = 0.5, 0.3
            if self._holding:
                load[GRIPPER] = float(self.grip.get("close_load", 0.3)) + 0.1
            return dict(self._pos), load
        bus = self.robot.bus
        with self._lock:
            # retries: at 5 V the servos brown out for a moment when a move starts (seen live)
            pos = bus.sync_read("Present_Position", num_retry=3)
            raw = bus.sync_read("Present_Load", normalize=False, num_retry=3)
        return {m: float(pos[m]) for m in MOTORS}, {m: decode_load(raw[m]) for m in MOTORS}

    def _send(self, goal: dict[str, float]) -> None:
        if self.fake:
            self._pos.update(goal)
            self._goal.update(goal)
            return
        with self._lock:
            sent = self.robot.send_action({f"{k}.pos": v for k, v in goal.items()})
        self._goal.update({k.removesuffix(".pos"): float(v) for k, v in sent.items()})

    def _clamp(self, k: str, v: float) -> float:
        lo, hi = (0.0, 100.0) if k == GRIPPER else self.kin.limits.get(k, (-180.0, 180.0))
        return min(max(v, lo), hi)

    def _check_stop(self) -> None:
        if self.stop.is_set():
            self.freeze()
            raise Stopped(f"stop: {self.stop.reason}")

    def _hold(self, pause: Flag) -> float:
        """Hold the last goal while `pause` is set. Returns the time held."""
        t0, limit = time.monotonic(), float(self.cfg.get("pause_timeout_s", 30))
        while pause.is_set():
            self._check_stop()
            if time.monotonic() - t0 > limit:
                self.freeze()
                raise Stopped(f"paused longer than {limit:g} s: {pause.reason}")
            time.sleep(1.0 / self.hz)
        return time.monotonic() - t0

    def _wall(self, t: float) -> float:
        return t * float(self.fake.get("arm", {}).get("time_scale", 1.0)) if self.fake else t

    # --- public -----------------------------------------------------------------------------------

    def pct(self, mm: float) -> float:
        """Gripper opening mm -> LeRobot 0..100 (two-point calibration in robot.yaml)."""
        c0, c1 = (
            float(self.grip.get("closed_mm_at_0", 0.0)),
            float(self.grip.get("open_mm_at_100", 60.0)),
        )
        return (mm - c0) / (c1 - c0) * 100.0

    def mm(self, pct: float) -> float:
        c0, c1 = (
            float(self.grip.get("closed_mm_at_0", 0.0)),
            float(self.grip.get("open_mm_at_100", 60.0)),
        )
        return c0 + (c1 - c0) * pct / 100.0

    def state(self) -> ArmState:
        pos, load = self._read()
        q = {j: pos[j] for j in JOINTS}
        goal = {j: self._goal.get(j, q[j]) for j in JOINTS}
        return ArmState(
            time.time(), q, goal, load, self.mm(pos[GRIPPER]), self.kin.fk(q, self.tool)
        )

    def freeze(self) -> None:
        """Goal = present position on every servo: the arm holds where it is."""
        if self.fake:
            self._goal = dict(self._pos)
        else:
            bus = self.robot.bus
            with self._lock:
                pos = bus.sync_read("Present_Position")
                bus.sync_write("Goal_Position", pos)
            self._goal = {m: float(v) for m, v in pos.items()}
        self.frozen = True

    def move_joints(
        self,
        q: dict[str, float],
        duration_s: float = 0.0,
        pause: Flag | None = None,
        speed_deg_s: float | None = None,
        gate: Callable[[], str | None] | None = None,
    ) -> TrackResult:
        """Min-jerk from the present position to `q` (any subset of JOINTS and "gripper").

        `duration_s` is raised so no joint's peak speed exceeds `speed_deg_s` (default: the cap).
        Every tick: `stop` -> freeze and raise Stopped; `pause` -> hold, resume when cleared;
        `gate()` returning a reason -> freeze, `halt_reason` = it, return with stopped=True.
        """
        self._check_stop()
        self.frozen, self.halt_reason = False, ""
        start, self.last_load = self._read()
        target = {k: self._clamp(k, float(v)) for k, v in q.items()}
        speed = min(float(speed_deg_s or self.speed), self.speed)
        gspeed = float(self.grip.get("speed_pct_s", 60)) * speed / self.speed
        need = [
            1.875 * abs(target[k] - start[k]) / (gspeed if k == GRIPPER else speed) for k in target
        ]
        dur = max([float(duration_s), 1.0 / self.hz, *need])
        wall, dt = self._wall(dur), 1.0 / self.hz
        t0, held, max_load, jam = time.monotonic(), 0.0, 0.0, 0
        jam_deg = float(self.cfg.get("collision_lag_deg", 8.0))
        jam_load = float(self.cfg.get("collision_load", 0.35))
        pos, goal, self.peak_load = start, dict(start), {m: 0.0 for m in MOTORS}
        while True:
            tick = time.monotonic()
            self._check_stop()
            if pause is not None and pause.is_set():
                held += self._hold(pause)
                continue
            if gate is not None:
                why = gate()
                if why:
                    err = max([abs(goal[k] - pos[k]) for k in target if k in JOINTS] + [0.0])
                    self.freeze()
                    self.halt_reason = why
                    return TrackResult(err, max_load, time.monotonic() - t0, stopped=True)
            s = 1.0 if wall <= 0 else (tick - t0 - held) / wall
            m = min_jerk(s)
            goal = {k: start[k] + (target[k] - start[k]) * m for k in target}
            self._send(goal)
            pos, load = self._read()
            self.last_load = load
            self._peak(load)
            max_load = max([max_load] + [abs(load[j]) for j in JOINTS])
            # A joint far behind its goal while pushing hard has hit something: freeze instead of
            # grinding on (seen live: a calibration move into the table broke the wrist).
            lag = max([abs(goal[k] - pos[k]) for k in target if k in JOINTS] + [0.0])
            jam = jam + 1 if lag > jam_deg and max(abs(load[j]) for j in JOINTS) > jam_load else 0
            if jam >= 3:
                self.freeze()
                self.halt_reason = "collision"
                return TrackResult(lag, max_load, time.monotonic() - t0, stopped=True)
            if s >= 1.0:
                break
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        settle = 0.0 if self.fake else 0.3
        t_end = time.monotonic() + settle
        while True:
            err = max([abs(target[k] - pos[k]) for k in target if k in JOINTS] + [0.0])
            if err < 1.5 or time.monotonic() >= t_end:
                break
            time.sleep(dt)
            self._check_stop()
            pos, load = self._read()
            self._peak(load)
            max_load = max([max_load] + [abs(load[j]) for j in JOINTS])
        return TrackResult(err, max_load, time.monotonic() - t0)

    def _peak(self, load: dict[str, float]) -> None:
        for m, v in load.items():
            self.peak_load[m] = max(self.peak_load.get(m, 0.0), abs(v))

    def _fake_contact(self) -> bool:
        """Fake contact: the tip at the food's top (food_width_mm high) while within food_radius_mm
        of food_xyz horizontally, else at floor_z + 2 mm."""
        f = self.fake
        x, y, z = self.kin.fk(self._pos, self.tool).xyz
        surface = float(f.get("arm", {}).get("floor_z", 0.0)) + 2.0
        if "food_xyz" in f:
            fx, fy, fz = (float(v) for v in f["food_xyz"])
            if math.hypot(x - fx, y - fy) <= float(f.get("food_radius_mm", 15)):
                surface = fz + float(f.get("food_width_mm", 25)) / 2
        return z <= surface

    def _fake_food(self) -> bool:
        """Fake grasp: tip within food_radius_mm of food_xyz horizontally and not above its top."""
        f = self.fake
        if not f or "food_xyz" not in f:
            return False
        x, y, z = self.kin.fk(self._pos, self.tool).xyz
        fx, fy, fz = (float(v) for v in f["food_xyz"])
        r = float(f.get("food_radius_mm", 15))
        return math.hypot(x - fx, y - fy) <= r and z <= fz + r

    def set_gripper_mm(self, mm: float, duration_s: float = 0.0) -> float:
        """Open/close to `mm` between the jaw tips; returns the measured opening."""
        top = float(self.grip.get("open_mm_at_100", 60.0))
        mm = min(max(mm, float(self.grip.get("closed_mm_at_0", 0.0))), top)
        if self.fake and self._holding and mm > float(self.fake.get("food_width_mm", 0)):
            self._holding = False
        self.move_joints({GRIPPER: self.pct(mm)}, duration_s)
        return self.mm(self._read()[0][GRIPPER])

    def close_gripper(self, load_limit: float | None = None) -> float:
        """Close slowly until the gripper load passes `load_limit` or it is shut; returns mm.
        On contact it keeps squeezing 3 units past the stall point to hold the object."""
        limit = float(load_limit if load_limit is not None else self.grip.get("close_load", 0.3))
        speed = self.speed * self.slow_factor
        if self.fake:
            if self._fake_food():
                width = float(self.fake.get("food_width_mm", 25))
                self.move_joints({GRIPPER: self.pct(width)}, speed_deg_s=speed)
                self._holding = True
                return width
            self.move_joints({GRIPPER: 0.0}, speed_deg_s=speed)
            return 0.0
        self._check_stop()
        self.frozen = False
        rate = float(self.grip.get("speed_pct_s", 60)) * self.slow_factor
        pos, _ = self._read()
        g0, t0, dt, stall = pos[GRIPPER], time.monotonic(), 1.0 / self.hz, 0
        while True:
            tick = time.monotonic()
            self._check_stop()
            t = tick - t0
            goal = max(g0 - rate * t, 0.0)
            self._send({GRIPPER: goal})
            pos, load = self._read()
            # Closed on something = the jaw stops following its goal for 3 ticks. The load alone
            # fired on the motor's own effort at 5 V (seen live: every close stopped 1-3 mm in).
            stall = stall + 1 if t > 0.3 and pos[GRIPPER] - goal > 8.0 else 0
            if stall >= 3 or abs(load[GRIPPER]) > max(limit, 0.8):
                self._send({GRIPPER: max(pos[GRIPPER] - 3.0, 0.0)})
                break
            if goal <= 0.0 and (pos[GRIPPER] < 1.5 or t > g0 / rate + 1.0):
                break
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        return self.mm(pos[GRIPPER])
