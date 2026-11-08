"""Arm check: status (LeRobot and URDF degrees, FK tip, loads) and a REPL to move the arm.

    python -m tools.arm_check [--fake] [--tool jaw|spoon]

Commands: xyz X Y Z [PITCH] | q NAME DEG | open MM | close | home | descend DZ | tool jaw|spoon |
torque on|off | watch [N] | (empty) status | quit. Space stops a running move (the arm freezes;
the next command clears the stop). Ctrl+C freezes, then turns torque off.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from pathlib import Path

import yaml

from core.types import GRIPPER, JOINTS, ActionResult, Flag, Pose, Stopped
from hw.arm import Arm
from motion.fence import Fence
from motion.kinematics import Kinematics
from motion.primitives import Motion

ROOT = Path(__file__).resolve().parents[1]
SHORT = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wflex",
    "wrist_roll": "roll",
}
ALIAS = {v: k for k, v in SHORT.items()} | {"grip": GRIPPER}
SAG = "WARNING: torque goes OFF and the arm will sag. Support it now."


def load_cfg(name: str) -> dict:
    p = ROOT / "config" / name
    return (yaml.safe_load(p.read_text()) or {}) if p.is_file() else {}


def setup(fake: bool) -> tuple[Arm, Motion, Flag]:
    """Connected arm and Motion from config/, the fake ones with fake.yaml's poses and food."""
    rc, ws = load_cfg("robot.yaml"), load_cfg("workspace.yaml")
    fk = load_cfg("fake.yaml") if fake else {}
    stop = Flag("stop")
    kin = Kinematics(rc["geometry"], rc["tool_offsets"], rc.get("joints"))
    arm = Arm(rc, stop, fake=fk or False, kin=kin)
    arm.connect()
    motion = Motion(
        arm, kin, Fence(ws), ws, (fk.get("poses") if fake else load_cfg("poses.yaml")) or {}
    )
    if fake:
        motion.bowl_rim_z = float(fk.get("bowl_rim_z", motion.bowl_rim_z))
    return arm, motion, stop


def status(arm: Arm) -> str:
    st = arm.state()
    u = arm.kin.to_urdf(st.q)
    x, y, z = st.tip.xyz
    return "\n".join(
        [
            "lerobot "
            + "  ".join(f"{SHORT[j]} {st.q[j]:+7.1f}" for j in JOINTS)
            + f"  | gripper {st.gripper_mm:.1f} mm",
            "urdf    " + "  ".join(f"{SHORT[j]} {u[j]:+7.1f}" for j in JOINTS),
            (
                f"tip/{arm.tool:<5} x {x:7.1f}  y {y:7.1f}  z {z:7.1f}"
                f"  pitch {st.tip.pitch:+6.1f}  roll {st.tip.roll:+6.1f}"
            ),
            "load    "
            + "  ".join(f"{SHORT.get(m, 'grip')} {st.load[m]:+.2f}" for m in (*JOINTS, GRIPPER)),
        ]
    )


def fmt(r: ActionResult) -> str:
    return (
        f"{r.name}: ok={r.ok} pred={r.predicted} meas={r.measured} err={r.error}"
        f"{' stopped' if r.stopped else ''}{' note=' + r.note if r.note else ''}"
        f" ({r.t1 - r.t0:.1f} s)"
    )


class Watch:
    """While a command runs: print the tip at 2 Hz; on a tty, space trips `stop`."""

    def __init__(self, arm: Arm, stop: Flag, tty: bool):
        self.arm, self.stop, self.tty, self.alive = arm, stop, tty, False

    def __enter__(self):
        self.alive, self.old = True, None
        if self.tty:
            import termios
            import tty

            self.old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self) -> None:
        t0 = next_print = time.monotonic()
        while self.alive:
            if (
                self.tty
                and select.select([sys.stdin], [], [], 0.05)[0]
                and os.read(sys.stdin.fileno(), 1) == b" "
            ):
                self.stop.trip("space")
            elif not self.tty:
                time.sleep(0.05)
            if time.monotonic() >= next_print + 0.5:
                next_print = time.monotonic()
                st = self.arm.state()
                print(
                    f"  {next_print - t0:4.1f} s  tip ({', '.join(f'{v:.0f}' for v in st.tip.xyz)})"
                    f" pitch {st.tip.pitch:+.0f}"
                    f"  max load {max(abs(v) for v in st.load.values()):.2f}",
                    flush=True,
                )

    def __exit__(self, *exc):
        self.alive = False
        self.t.join()
        if self.old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, self.old)


def run(cmd: list[str], arm: Arm, motion: Motion, tty: bool) -> None:
    c, a = cmd[0], cmd[1:]
    if c == "xyz":
        tip = motion.tip
        pitch = float(a[3]) if len(a) > 3 else tip.pitch
        print(fmt(motion.move_to(Pose((float(a[0]), float(a[1]), float(a[2])), pitch, tip.roll))))
    elif c == "q":
        name = ALIAS.get(a[0], a[0])
        if name not in (*JOINTS, GRIPPER):
            print(f"unknown joint {a[0]}")
            return
        if name != GRIPPER:
            q0 = arm.state().q
            why = motion._path_reason(q0, q0 | {name: float(a[1])}, "transit")
            if why:
                print(f"refused, fence: {why}")
                return
        tr = arm.move_joints({name: float(a[1])}, 0.0, speed_deg_s=arm.speed * arm.slow_factor)
        print(
            f"q {name} -> {float(a[1]):.1f}: err {tr.max_err_deg:.1f} deg,"
            f" max load {tr.max_load:.2f}, {tr.duration_s:.1f} s"
        )
    elif c == "open":
        print(fmt(motion.open(float(a[0]))))
    elif c == "close":
        print(fmt(motion.close()))
    elif c == "home":
        print(fmt(motion.home()))
    elif c == "descend":
        print(fmt(motion.descend_until_contact(float(a[0]))))
    elif c == "tool":
        arm.tool = a[0]
        motion._segment()
    elif c == "torque":
        if a[0] == "off":
            print(SAG)
            time.sleep(0 if arm.fake else 2)
        arm.torque(a[0] == "on")
    elif c == "watch":
        n = int(a[0]) if a else 4
        while True:
            print(status(arm), flush=True)
            n -= 1
            if (tty and select.select([sys.stdin], [], [], 0.5)[0]) or (not tty and n <= 0):
                if tty:
                    sys.stdin.readline()
                return
            if not tty:
                time.sleep(0.5)
    else:
        print(__doc__)
        return
    print(status(arm))


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 status and REPL")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--tool", default="jaw", choices=("jaw", "spoon"))
    args = ap.parse_args()
    arm, motion, stop = setup(args.fake)
    arm.tool = args.tool
    motion._segment()
    from dashboard.client import for_tool

    pub = for_tool("arm-check", arm, stop, fake=args.fake)
    tty = sys.stdin.isatty()
    print(status(arm))
    try:
        while True:
            line = input("arm> ") if tty else sys.stdin.readline()
            if not tty:
                if not line:
                    break
                print(f"arm> {line.strip()}")
            cmd = line.split()
            if not cmd:
                print(status(arm))
                continue
            if cmd[0] in ("quit", "exit"):
                break
            if stop.is_set():
                print(f"(clearing stop: {stop.reason})")
                stop.clear()
            pub.state.set(action=line.strip())
            try:
                with Watch(arm, stop, tty):
                    run(cmd, arm, motion, tty)
            except Stopped as e:
                print(f"STOPPED: {e}; arm frozen={arm.frozen}")
                if getattr(e, "result", None):
                    print(fmt(e.result))
            except (ValueError, IndexError) as e:
                print(f"bad command: {e}")
    except (KeyboardInterrupt, EOFError):
        stop.trip("ctrl+c")
        arm.freeze()
        stop.clear()  # so disconnect() still folds to rest instead of dropping the arm
        print("\nfrozen; folding to rest")
    finally:
        arm.disconnect()  # folds to rest_pose slowly, then torque off
        pub.stop()
        print("disconnected, torque off")


if __name__ == "__main__":
    main()
