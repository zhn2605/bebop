"""Orchestrator helpers: locate the target, run one plan action, fill its camera metrics, build the
stream overlays, and the fixed deliver sequence. `run` is the core.run.Runner."""

from __future__ import annotations

import atexit
import math
import os
import select
import socket
import sys
import threading
import time
from dataclasses import asdict

from core.types import Action, ActionResult, Pose

PLAN_CAMS = ("top", "side")
BITE_TIMEOUT_S = 8.0
READY_TIMEOUT_S = 20.0
MOUTH_OPEN_HOLD_S = 1.0
WIDE_FACTOR = 1.4  # VLM boxes run loose: escalate only past gripper_max * this, else clamp open


class NeedHelp(Exception):
    """End the goal here and wait for the caregiver (HELP)."""


class Abort(Exception):
    """End the task here (the VLM budget is spent)."""


def lan_url(port: int) -> str:
    """The dashboard URL other devices on the network can open (for the ntfy tap)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this picks the outgoing address
            host = s.getsockname()[0]
    except OSError:
        host = "localhost"
    return f"http://{host}:{port}/"


def keyboard(on_space) -> None:
    """Space in the terminal calls on_space: a cbreak-mode reader thread (Ctrl+C still works).
    Linux only; skipped when stdin is not a terminal."""
    if os.name != "posix" or not sys.stdin.isatty():
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)

    def loop() -> None:
        while True:
            if select.select([fd], [], [], 0.2)[0] and os.read(fd, 1) == b" ":
                on_space("space bar")

    threading.Thread(target=loop, name="keyboard", daemon=True).start()


def center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def locate(plan, calib, rim_z: float):
    """(target xyz, width mm) from the plan's pixel boxes: triangulated from the box centres when
    top and side both see the object, else the top ray at the bowl rim height; (None, None) when
    top does not see it. The width is the top box's width at the target height."""
    boxes = {c: b for c, b in (plan.target or {}).items() if c in calib.names}
    if "top" not in boxes:
        return None, None
    if "side" in boxes:
        xyz = calib.triangulate({c: center(boxes[c]) for c in PLAN_CAMS})
    else:
        xyz = calib.at_height("top", center(boxes["top"]), rim_z)
    xyz = tuple(round(float(v), 1) for v in xyz)
    return xyz, round(float(calib.width_mm("top", boxes["top"], xyz[2])), 1)


def target_pose(args: dict, target, hover_dz: float, pitch: float) -> Pose:
    """move_to's pose: at "target" means the target xyz raised by dz, with the plan's pitch."""
    if args.get("at", "target") != "target":
        raise ValueError(f"move_to at {args.get('at')!r}: only \"target\" is known")
    x, y, z = target
    dz = float(args.get("dz", hover_dz))
    return Pose((x, y, z + dz), float(args.get("pitch", pitch)), float(args.get("roll", 0.0)))


def dispatch(run, a: Action, target, width: float | None) -> ActionResult:
    """Runs one plan action on run.motion (or the planner, for verify) and returns its result."""
    m, args = run.motion, a.args
    if a.name == "move_to":
        pitch = -60.0 if run.mode == "spoon" else -90.0
        pose = target_pose(args, target, run.hover_dz, pitch)
        r = m.move_to(pose, speed=args.get("speed", "normal"))
        enrich_move(r, pose, run.cameras, run.calib, run.marker_cfg)
        return r
    if a.name == "descend_until_contact":
        return m.descend_until_contact(float(args.get("max_dz", 60)), load=args.get("load"),
                                       expect_z=target[2])
    if a.name == "open":
        want = float(args.get("width_mm", (width or 30) + 15))
        r = m.open(min(want, run.gripper_max))
        if want > run.gripper_max:
            clamp = f"clamped {want:.0f} mm to the gripper max {run.gripper_max:.0f} mm"
            r.note = f"{r.note}; {clamp}" if r.note else clamp
        return r
    if a.name == "close":
        return m.close(expect_mm=width)
    if a.name == "scoop":
        return m.scoop(float(args.get("depth_mm", 15)), float(args.get("drag_mm", 30)),
                       float(args.get("tilt_deg", 30)))
    if a.name == "lift":
        return m.lift(float(args.get("dz", 80)))
    if a.name in ("retract", "home", "stage"):
        return getattr(m, a.name)()
    if a.name == "present":
        return m.present(run.gate, run.dist_cm)
    if a.name == "verify":
        v = run.verify(str(args.get("question") or f"Is {run.object} held in the {run.tool_word}?"))
        return ActionResult("verify", v.ok, measured={"confidence": v.confidence, "note": v.note})
    if a.name == "ask_operator":
        raise NeedHelp(str(args.get("question") or "the planner asked for the operator"))
    raise ValueError(f"unknown action {a.name!r}")


def is_contact(r: ActionResult) -> bool:
    """Unexpected contact (tracking error with load), which freezes and retracts. Contact is the
    point of descend_until_contact, close and scoop, so it doesn't count there."""
    return (not r.ok and "contact" in r.note.split("; ")
            and r.name not in ("descend_until_contact", "close", "scoop"))


def fresh_frame(cameras, cam: str, after: float, timeout_s: float = 0.6):
    """The first frame of cam grabbed after `after`, or the latest one at the timeout."""
    deadline = time.time() + timeout_s
    while True:
        got = cameras.latest(cam)
        if got is not None and (got[1] >= after or time.time() > deadline):
            return got[0]
        if time.time() > deadline:
            return None
        time.sleep(0.02)


def enrich_move(r: ActionResult, pose: Pose, cameras, calib, marker_cfg: dict) -> None:
    """Adds the camera metrics of a move_to: predicted tip_uv per camera, the measured marker_uv,
    the error px between them, and mm_from_marker (the triangulated marker against the target)."""
    from perception.marker import find_marker

    after = r.t1 or time.time()
    tip_uv, marker_uv, px = {}, {}, {}
    for cam in PLAN_CAMS:
        if cam not in calib.names or cam not in cameras.names:
            continue
        tip_uv[cam] = [round(float(v), 1) for v in calib.project(cam, pose.xyz)]
        # The predicted pixel steers the detector past yellow clutter elsewhere in the frame.
        uv = find_marker(fresh_frame(cameras, cam, after), marker_cfg, near=tuple(tip_uv[cam]))
        if uv is not None:
            marker_uv[cam] = [round(float(v), 1) for v in uv]
            px[cam] = round(math.dist(tip_uv[cam], marker_uv[cam]), 1)
    r.predicted.setdefault("tip_xyz", [round(v, 1) for v in pose.xyz])
    r.predicted["tip_uv"] = tip_uv
    r.measured["marker_uv"] = marker_uv
    r.error["px"] = px
    if len(marker_uv) >= 2:
        seen = [round(float(v), 1) for v in calib.triangulate(marker_uv)]
        r.measured["marker_xyz"] = seen
        r.error["mm_from_marker"] = round(math.dist(seen, pose.xyz), 1)


def position_error_mm(results: list[ActionResult]) -> float | None:
    """The last move_to's camera error (mm_from_marker), else its FK error (xyz_mm)."""
    for r in reversed(results):
        if r.name == "move_to":
            e = r.error.get("mm_from_marker", r.error.get("xyz_mm"))
            return None if e is None else float(e)
    return None


def plan_annotations(plan, target, calib, label: str) -> dict[str, list]:
    """VLM boxes (rectangles with the object's name) and the target (a cross) per camera."""
    anns: dict[str, list] = {}
    for cam, box in (plan.target or {}).items():
        anns.setdefault(cam, []).append(
            {"type": "box", "box": [float(v) for v in box], "color": "yellow", "label": label})
    if target is not None:
        for cam in calib.names:
            uv = [float(v) for v in calib.project(cam, target)]
            anns.setdefault(cam, []).append(
                {"type": "cross", "uv": uv, "color": "red", "label": "food"})
    return anns


def move_annotations(r: ActionResult) -> dict[str, list]:
    """Where the gripper should be (hollow circle) and where the camera sees its tape (filled) per camera."""
    anns: dict[str, list] = {}
    for cam, uv in (r.predicted.get("tip_uv") or {}).items():
        anns.setdefault(cam, []).append(
            {"type": "point", "uv": uv, "color": "cyan", "label": "gripper goal", "hollow": True})
    for cam, uv in (r.measured.get("marker_uv") or {}).items():
        anns.setdefault(cam, []).append(
            {"type": "point", "uv": uv, "color": "green", "label": "gripper seen"})
    return anns


def _wait(run, done, timeout_s: float) -> bool:
    """Polls done() at 20 Hz until it is true (True) or the timeout (False); raises on stop."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run.check()
        if done():
            return True
        time.sleep(0.05)
    return False


def _mouth_open_for(run, hold_s: float):
    since = [None]

    def done() -> bool:
        if not run.face_state().mouth_open:
            since[0] = None
            return False
        since[0] = since[0] or time.time()
        return time.time() - since[0] >= hold_s

    return done


def deliver(run, obj: str, mode: str) -> bool:
    """The fixed deliver sequence; the VLM only verifies. stage -> wait for an open mouth or a
    "yes" -> present behind the gate -> solid: open once the mouth stayed open 1 s; spoon: hold
    until the mouth closes (bite) or 8 s -> retract -> verify the tool is empty."""
    run.begin_record("deliver", obj, mode)
    run.enter("MOVING", goal="deliver", attempt=1, action="stage")
    r = run.do("stage", run.motion.stage)
    if not r.ok:  # no point waiting for a mouth when the arm is not at the stage pose
        raise NeedHelp(f"stage: {r.note}")
    run.yes_evt.clear()
    run.enter("LISTENING", action="waiting for an open mouth or 'yes'")
    if not _wait(run, lambda: run.face_state().mouth_open or run.yes_evt.is_set(), READY_TIMEOUT_S):
        run.do("retract", run.motion.retract)
        raise NeedHelp(f"no open mouth or 'yes' within {READY_TIMEOUT_S:.0f} s at the stage pose")
    run.enter("PRESENTING", action="present")
    r = run.do("present", run.motion.present, run.gate, run.dist_cm)
    run.check()
    if mode == "solid":
        run.state.set(action="waiting for the mouth to stay open")
        if _wait(run, _mouth_open_for(run, MOUTH_OPEN_HOLD_S), BITE_TIMEOUT_S):
            run.do("open", run.motion.open, min(run.gripper_max, (run.width or 30) + 15))
            time.sleep(1.0)  # let them take it
    else:
        run.state.set(action="holding for the bite")
        _wait(run, lambda: not run.face_state().mouth_open, BITE_TIMEOUT_S)
    run.enter("MOVING", action="retract")
    run.do("retract", run.motion.retract)
    v = run.verify(f"Is the {run.tool_word} empty?")
    run.end_record(v, v.ok, note=f"present stopped by {r.measured.get('stopped_by', '-')}")
    return v.ok


def acquire(run) -> bool:
    """Up to max_attempts attempts, each told the previous ones' metrics; True once held."""
    history: list[dict] = []
    return any(attempt(run, n, history) for n in range(1, run.max_attempts + 1))


def attempt(run, n: int, history: list[dict]) -> bool:
    """One acquire attempt: frames -> plan -> target -> actions -> verify. True when held: the
    verify answer, and for solid also the close metric."""
    run.budget_check()
    run.enter("THINKING", goal="acquire", attempt=n, action="plan", action_args={})
    images = run.grab()
    tag = f"a{len(run.state.attempts) + 1}-plan"
    frames = {f"plan {c}": run.log.save_frame(c, j, tag) for c, j in images.items()}
    ctx = {"task": run.object, "mode": run.mode, "goal": "acquire",
           "capabilities": run.capabilities(), "attempts": history,
           "lessons": list(run.lessons), "images": images}
    plan = run.planner.plan(ctx)
    run.spent(plan.cost_usd, plan.latency_s)
    target, width = locate(plan, run.calib, run.rim_z)
    run.begin_record("acquire", run.object, run.mode, n=n, plan=plan, target_xyz=target,
                     width_mm=width, frames=frames)
    run.log.emit("plan", attempt=n, plan=plan, target_xyz=target, width_mm=width)
    run.plan_anns = plan_annotations(plan, target, run.calib, run.object)
    run.show({})
    run.state.set(target_xyz=target, err_px=None, err_mm=None)
    if plan.lesson:
        run.lessons.append(plan.lesson)
    run.check()
    if plan.escalate:
        run.end_record(None, False, note=f"escalated: {plan.escalate_reason}")
        raise NeedHelp(plan.escalate_reason or "the planner escalated")
    if width is not None and width > run.gripper_max * WIDE_FACTOR:
        run.end_record(None, False, note=f"{width:.0f} mm wide")
        raise NeedHelp(f"{run.object} looks {width:.0f} mm wide; the gripper opens "
                       f"{run.gripper_max:.0f} mm")
    run.width = width
    note, moved = "", False
    if target is None:
        note = "no target: the top camera has no box"
    else:
        run.enter("MOVING", action="")
        for a in plan.actions:
            r = run.do(a.name, dispatch, run, a, target, width, args=a.args)
            moved = moved or a.name not in ("verify", "ask_operator")
            if is_contact(r):
                note = f"contact during {a.name}: froze and retracted"
                run.arm.freeze()
                break
            if r.note.startswith("error") or (a.name == "move_to" and not r.ok):
                note = f"{a.name} failed: {r.note}"
                break
    closes = [r for r in run.results if r.name == "close"]
    if not note and run.mode == "solid" and closes and closes[-1].ok:
        # The close said held; make sure it is still between the jaws after the lift.
        if run.arm.state().gripper_mm < 3:
            note = "slipped after close: the gripper is shut"
    if not note and run.mode == "solid" and closes and not closes[-1].ok:
        note = f"close: {closes[-1].note}"  # the gripper already says it missed: no VLM verify
    verdict = None if note else run.verify(f"Is {run.object} held in the {run.tool_word}?")
    ok = bool(verdict and verdict.ok) and (run.mode == "spoon" or bool(closes and closes[-1].ok))
    run.end_record(verdict, ok, lesson=plan.lesson, note=note,
                   err_mm=position_error_mm(run.results))
    history.append({"reasoning": plan.reasoning, "results": [asdict(r) for r in run.results],
                    "verdict": asdict(verdict) if verdict else None, "note": note})
    if not ok and moved:
        if closes and closes[-1].measured.get("width_mm", 0) >= 3 and target is not None:
            # Something is still in the jaws: put it back over the target before retracting.
            run.enter("MOVING", action="open")
            hover = Pose((target[0], target[1], target[2] + run.hover_dz), -90.0)
            run.do("move_to", run.motion.move_to, hover)
            run.do("open", run.motion.open, min(run.gripper_max, (width or 40.0) + 15.0))
        run.enter("MOVING", action="retract")
        run.do("retract", run.motion.retract)
    return ok
