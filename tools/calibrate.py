"""Fit the camera calibration from the tape marker on the fixed jaw tip.

    python -m tools.calibrate [--fake] [--out PATH]
        visits a grid of tip poses inside the fence, finds the marker in `top` and `side`, fits a
        DLT per camera, prints RMS px, saves config/calib.json (--fake: data/calib_fake.json)
    python -m tools.calibrate --check [--here] [--fake]
        moves to the fence center (or stays, with --here), writes data/calib_check/<cam>.jpg with
        the predicted tip (hollow) and the detected marker (filled), prints the px error

Keep everyone out of reach of the arm while it runs. Ctrl+C freezes the arm.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from core.state import State
from core.types import UV, Flag, Pose, Stopped, Unreachable
from motion.fence import from_fence_frame
from perception.calib import Calibration
from perception.cameras import Cameras, FakeWorld
from perception.marker import find_marker
from perception.overlay import draw

ROOT = Path(__file__).resolve().parent.parent
CAMS = ("top", "side")
FENCE = {"x": [150.0, 330.0], "y": [-120.0, 120.0], "z": [10.0, 180.0]}


def load(name: str) -> dict:
    path = ROOT / "config" / name
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def fence_box(ws: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of workspace.yaml's calib_box (or a box-shaped fence); {x: [lo, hi], y, z} or
    {min: xyz, max: xyz}. A ring-shaped fence has no corners to sweep, so the box is separate."""
    f = ws.get("calib_box") or ws.get("fence") or {}
    if "r" in f:
        f = {}
    if "min" in f and "max" in f:
        return np.array(f["min"], float), np.array(f["max"], float)
    axes = [f.get(a) or FENCE[a] for a in "xyz"]
    return np.array([a[0] for a in axes], float), np.array([a[1] for a in axes], float)


def grid(lo: np.ndarray, hi: np.ndarray, yaw_deg: float = 0.0) -> list[Pose]:
    """3x3 xy at two heights with pitch -60 (snake order), then 3 at a middle height, pitch -30.
    Built in the fence frame and turned by `yaw_deg` into the base frame (workspace.yaml's
    `fence_yaw_deg`), so the grid lands over the plate wherever it is around the base."""
    poses = _grid(lo, hi)
    return [Pose(from_fence_frame(p.xyz, yaw_deg), p.pitch) for p in poses]


def _grid(lo: np.ndarray, hi: np.ndarray) -> list[Pose]:
    xs, ys = (
        np.linspace(lo[i] + 0.15 * (hi[i] - lo[i]), hi[i] - 0.15 * (hi[i] - lo[i]), 3)
        for i in (0, 1)
    )
    zs = [lo[2] + 0.2 * (hi[2] - lo[2]), lo[2] + 0.7 * (hi[2] - lo[2])]
    poses = []
    for z in zs:
        for i, x in enumerate(xs):
            for y in ys if i % 2 == 0 else ys[::-1]:
                poses.append(Pose((float(x), float(y), float(z)), -60.0))
    zm = float(np.mean(zs))
    for x, y in ((xs[1], ys[0]), (xs[2], ys[1]), (xs[1], ys[2])):
        poses.append(Pose((float(x), float(y), zm), -30.0))
    return poses


def build(fake: bool, stop: Flag):
    """Connected Arm (jaw tool), Motion and workspace.yaml; fake mode uses config/fake.yaml."""
    from hw.arm import Arm
    from motion.fence import Fence
    from motion.primitives import Motion

    robot, ws = load("robot.yaml"), load("workspace.yaml")
    fake_cfg = load("fake.yaml") if fake else None
    arm = Arm(robot, stop, fake=fake_cfg or False)
    arm.connect()
    # The tape sits behind the jaw tips; robot.yaml's `marker` tool offset puts the FK there.
    arm.tool = "marker" if "marker" in arm.kin.tools else "jaw"
    poses = (fake_cfg.get("poses") if fake else load("poses.yaml")) or {}
    return arm, Motion(arm, arm.kin, Fence(ws), ws, poses), ws


def frames_after(cams: Cameras, t0: float, timeout: float = 2.0) -> dict:
    """The first frame of each of CAMS taken after t0."""
    out, t_end = {}, time.time() + timeout
    while len(out) < len(CAMS) and time.time() < t_end:
        for n in CAMS:
            got = cams.latest(n)
            if n not in out and got is not None and got[1] > t0:
                out[n] = got[0]
        time.sleep(0.01)
    return out


def goto(motion, pose: Pose) -> str | None:
    """Move slowly; the reason on failure (unreachable, fence), else None."""
    try:
        r = motion.move_to(pose, speed="slow")
    except Unreachable as e:
        return f"unreachable ({e})"
    return None if r.ok else (r.note or "move failed")


def pick_moving(records: list[dict], cam: str, radius: float = 12.0) -> list[UV | None]:
    """Per record, the largest blob of `cam` that is not static. A blob position seen in at
    least a third of the records (within `radius` px) is food or clutter: the tape moves with
    the arm, so it never repeats. Seen live: donut holes share the tape's hue."""
    import numpy as np

    all_pts = [uv for r in records for uv, _ in r["blobs"].get(cam, [])]
    threshold = max(2, len(records) // 3)

    def static(uv: UV) -> bool:
        hits = sum(1 for p in all_pts if np.hypot(p[0] - uv[0], p[1] - uv[1]) <= radius)
        return hits >= threshold

    out = []
    for r in records:
        moving = [uv for uv, _ in r["blobs"].get(cam, []) if not static(uv)]
        out.append(moving[0] if moving else None)
    return out


def calibrate(
    arm, motion, cams, cam_cfg, poses: list[Pose], out: Path, state: State
) -> Calibration | None:
    import cv2

    from perception.marker import find_blobs

    frames_dir = ROOT / "data" / "calib_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records, size = [], {}
    for i, pose in enumerate(poses, 1):
        x, y, z = pose.xyz
        head = f"{i:2d}/{len(poses)} ({x:4.0f},{y:4.0f},{z:4.0f}) p{pose.pitch:.0f}"
        state.set(action=f"pose {i}/{len(poses)}", target_xyz=pose.xyz)
        why = goto(motion, pose)
        if why:
            print(f"{head} skipped: {why}")
            continue
        time.sleep(0.5)
        frames = frames_after(cams, time.time())
        for n, f in frames.items():
            cv2.imwrite(str(frames_dir / f"{i:02d}-{n}.jpg"), f)
        blobs = {n: find_blobs(f, cam_cfg)[:6] for n, f in frames.items()}
        size.update({n: [f.shape[1], f.shape[0]] for n, f in frames.items()})
        tip = tuple(float(c) for c in arm.state().tip.xyz)
        seen = "  ".join(f"{n} {len(blobs.get(n, []))} blobs" for n in CAMS)
        print(f"{head} tip ({tip[0]:.0f},{tip[1]:.0f},{tip[2]:.0f})  {seen}")
        records.append({"tip": tip, "blobs": blobs})
        for n in CAMS:
            state.annotate(n, [
                {"type": "point", "uv": uv, "color": "green" if k == 0 else "orange",
                 "label": f"pose {i}" if k == 0 else ""}
                for k, (uv, _) in enumerate(blobs.get(n, []))
            ])
    state.set(action="home, then fit")
    home(motion)
    picked = {n: pick_moving(records, n) for n in CAMS}
    samples = []
    for k, r in enumerate(records):
        uvs = {n: picked[n][k] for n in CAMS if picked[n][k] is not None}
        print(f"   {k + 1:2d}: " + "  ".join(f"{n} {_fmt(uvs.get(n))}" for n in CAMS))
        if all(n in uvs for n in CAMS):
            samples.append((r["tip"], uvs))
    print("frames saved under data/calib_frames/")
    print(f"{len(samples)} samples seen by {' and '.join(CAMS)}")
    if len(samples) < 8:
        print(
            "need at least 8: check the marker HSV range, lighting and that the cameras see the tip"
        )
        state.set(stage="FAILED", help=f"only {len(samples)} samples; need 8")
        return None
    cal = robust_fit(samples)
    cal.size = size
    for n, rms in cal.rms_px.items():
        print(f"{n}: RMS {rms:.2f} px")
    state.set(err_px={n: round(float(r), 1) for n, r in cal.rms_px.items()})
    if len(CAMS) >= 2:
        err = [np.linalg.norm(np.subtract(cal.triangulate(uvs), xyz)) for xyz, uvs in samples]
        print(f"triangulated marker vs FK tip: mean {np.mean(err):.1f} mm, max {np.max(err):.1f} mm")
        state.set(err_mm=float(np.mean(err)))
    cal.save(out)
    print(f"saved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    state.set(stage="CALIBRATED", action=f"saved {out.name}")
    for n in CAMS:
        state.annotate(n, [])
    return cal


def robust_fit(samples: list) -> Calibration:
    """Fit, drop samples whose reprojection error is over max(15 px, 3x the median), refit.
    Seen live: when the tape is hidden, the largest yellow blob is clutter (a coat rack), and one
    such sample ruins a plain least-squares fit."""
    keep = list(range(len(samples)))
    while True:
        cal = Calibration.fit([samples[i] for i in keep])
        worst = {}
        for i in keep:
            xyz, uvs = samples[i]
            errs = [
                np.hypot(*np.subtract(cal.project(n, xyz), uv)) for n, uv in uvs.items() if n in cal.P
            ]
            worst[i] = max(errs) if errs else 0.0
        thr = max(15.0, 3 * float(np.median(list(worst.values()))))
        good = [i for i in keep if worst[i] <= thr]
        if len(good) == len(keep) or len(good) < 8:
            dropped = [i + 1 for i in range(len(samples)) if i not in keep]
            if dropped:
                print(f"dropped outlier samples {dropped}")
            return cal
        keep = good


def check(
    arm, motion, cams, cam_cfg, cal: Calibration, here: bool, ws: dict, state: State
) -> None:
    import cv2

    if not here:
        lo, hi = fence_box(ws)
        center = (lo + hi) / 2
        c = (float(center[0]), float(center[1]), float(lo[2] + 0.2 * (hi[2] - lo[2])))
        state.set(action="move to the sweep center")
        why = goto(motion, Pose(from_fence_frame(c, float(ws.get("fence_yaw_deg", 0))), -60.0))
        if why:
            print(f"cannot reach the fence center: {why}")
            return
        time.sleep(0.5)
    tip = tuple(float(c) for c in arm.state().tip.xyz)
    frames = frames_after(cams, time.time())
    out = ROOT / "data" / "calib_check"
    out.mkdir(parents=True, exist_ok=True)
    print(f"tip (FK) ({tip[0]:.0f},{tip[1]:.0f},{tip[2]:.0f})")
    uvs = {}
    for n, frame in frames.items():
        if n not in cal.P:
            continue
        pred, uv = cal.project(n, tip), find_marker(frame, cam_cfg)
        uvs[n] = uv
        ann = [
            {"type": "point", "uv": pred, "hollow": True, "color": "magenta", "label": "gripper from joints"}
        ]
        if uv:
            ann.append({"type": "point", "uv": uv, "color": "green", "label": "gripper seen"})
        cv2.imwrite(str(out / f"{n}.jpg"), draw(frame, ann))
        state.annotate(n, ann)
        d = f"{np.hypot(*np.subtract(uv, pred)):.1f} px" if uv else "marker not found"
        print(
            f"{n}: predicted {_fmt(pred)} marker {_fmt(uv)} error {d} -> data/calib_check/{n}.jpg"
        )
    state.set(err_px={
        n: round(float(np.hypot(*np.subtract(uv, cal.project(n, tip)))), 1)
        for n, uv in uvs.items() if uv
    })
    if len(CAMS) >= 2 and all(uvs.get(n) for n in CAMS):
        xyz = cal.triangulate(uvs)
        print(f"triangulated marker vs FK tip: {np.linalg.norm(np.subtract(xyz, tip)):.1f} mm")
        state.set(err_mm=float(np.linalg.norm(np.subtract(xyz, tip))))
    if not here:
        home(motion)


def home(motion) -> None:
    r = motion.home(speed="slow")
    if not r.ok:
        print(f"home: {r.note}")


def _fmt(uv) -> str:
    return f"({uv[0]:6.1f},{uv[1]:6.1f})" if uv else "     ---     "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="fake arm and fake cameras")
    ap.add_argument("--check", action="store_true", help="draw the predicted tip, no fit")
    ap.add_argument("--here", action="store_true", help="--check at the current pose, no move")
    ap.add_argument("--out", type=Path, help="calibration file (default config/calib.json)")
    args = ap.parse_args()
    out = args.out or ROOT / ("data/calib_fake.json" if args.fake else "config/calib.json")
    cam_cfg = load("cameras.yaml")
    stop = Flag("stop")
    if not args.fake:
        from hw.takeover import take_over

        take_over()
    arm, motion, ws = build(args.fake, stop)
    world = None
    if args.fake:
        fk = arm.fake
        world = FakeWorld(
            lambda: arm.state().tip,
            None,
            fk.get("food_xyz", (250, 0, 15)),
            fk.get("food_radius_mm", 15),
        )
    cams = Cameras(cam_cfg, fake=world).start()
    from dashboard.client import Publisher

    state = State({"stop": stop})
    state.set(stage="CHECK" if args.check else "CALIBRATING", task="calibrate", fake=args.fake)
    pub = Publisher(
        "calibrate", state, cams, arm=arm, calib=Calibration.load(out) if out.is_file() else None,
        controls={"stop": stop.trip},
    ).start()
    try:
        global CAMS
        missing = [n for n in CAMS if n not in cams.names]
        CAMS = tuple(n for n in CAMS if n in cams.names)
        if missing:
            print(f"camera(s) {missing} did not open; check config/cameras.yaml")
        if missing and CAMS:
            print(f"fitting {CAMS[0]} alone; triangulation needs both cameras")
        if not CAMS:
            pass
        elif args.check:
            check(arm, motion, cams, cam_cfg, Calibration.load(out), args.here, ws, state)
        else:
            poses = grid(*fence_box(ws), float(ws.get("fence_yaw_deg", 0)))
            print(f"{len(poses)} poses; keep clear of the arm. Ctrl+C freezes it.")
            cal = calibrate(arm, motion, cams, cam_cfg, poses, out, state)
            if cal is not None:
                pub.calib = cal  # the streams now show the new fit's tip
    except (KeyboardInterrupt, Stopped):
        why = stop.reason or "ctrl-c"
        stop.trip(why)
        arm.freeze()
        stop.clear()  # so disconnect() still folds to rest instead of dropping the arm
        print(f"\nstopped ({why}); folding to rest")
        state.set(stage="STOPPED", action=why)
    finally:
        try:
            arm.disconnect()  # folds to rest_pose slowly, then torque off
        finally:
            pub.stop()
            cams.stop()


if __name__ == "__main__":
    main()
