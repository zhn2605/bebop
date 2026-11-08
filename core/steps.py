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

from core.types import Action, ActionResult, Pose, Unreachable
from motion.kinematics import approach

PLAN_CAMS = ("top", "side")
BITE_TIMEOUT_S = 8.0
READY_TIMEOUT_S = 20.0
MOUTH_OPEN_HOLD_S = 1.0
WIDE_FACTOR = 1.4  # VLM boxes run loose: escalate only past gripper_max * this, else clamp open
SIDE_SLACK_PX = 10  # the top box's ray may pass this far outside the side box: still the same piece
# Hover correction (move_to): re-aim by where top sees the marker vs where the arm put it.
SERVO_STEPS = 2       # corrections at most
SERVO_DONE_MM = 3.0   # the offset changed less than this since the last correction: done
SERVO_MAX_MM = 40.0   # an offset this big is the wrong blob (the food, clutter), not the arm


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


def inside(uv, box, slack: float = 0.0) -> bool:
    return box[0] - slack <= uv[0] <= box[2] + slack and box[1] - slack <= uv[1] <= box[3] + slack


def locate(plan, calib, rim_z: float):
    """(target xyz, width mm) from the plan's pixel boxes; (None, None) when top has no box.
    The top box centre fixes x and y (from above every piece stands apart). The side box only
    picks the height along that ray, and only when the ray passes through it; otherwise it boxed
    another piece (from the side they line up behind each other) and the height is the bowl
    rim's. So the target always projects to the top box's centre. The width is the top box's
    width at the target height."""
    boxes = {c: b for c, b in (plan.target or {}).items() if c in calib.names}
    if "top" not in boxes:
        return None, None
    uv = center(boxes["top"])
    xyz = calib.at_height("top", uv, rim_z)
    if "side" in boxes:
        p, off = calib.along_ray("top", uv, "side", center(boxes["side"]), (rim_z - 60, rim_z + 60))
        if inside(calib.project("side", p), boxes["side"], SIDE_SLACK_PX):
            xyz = p
        else:
            print(f"the side box is {off:.0f} px off the top box's ray (another piece?): "
                  "height from the bowl rim")
    xyz = tuple(round(float(v), 1) for v in xyz)
    return xyz, round(float(calib.width_mm("top", boxes["top"], xyz[2])), 1)


def target_pose(args: dict, target, hover_dz: float, pitch: float, kin=None, tool="jaw") -> Pose:
    """move_to's pose: at "target" means dz back from the target along the tool axis (straight up
    at pitch -90), at the plan's pitch or the nearest one IK reaches the target with."""
    if args.get("at", "target") != "target":
        raise ValueError(f"move_to at {args.get('at')!r}: only \"target\" is known")
    x, y, z = target
    dz = float(args.get("dz", hover_dz))
    p, roll = float(args.get("pitch", pitch)), float(args.get("roll", 0.0))
    if kin is not None:
        try:
            p = kin.ik_any_pitch(Pose((x, y, z), p, roll), tool)[1].pitch
        except Unreachable:
            pass
    # Back along the tool axis, but at most 30 deg off vertical: at full reach the gripper tilts
    # far over, and a hover that far back sat 5 cm from the food (seen live).
    d = approach(min(p, -60.0), math.atan2(y, x))
    return Pose((x - d[0] * dz, y - d[1] * dz, z - d[2] * dz), p, roll)


def dispatch(run, a: Action, target, width: float | None) -> ActionResult:
    """Runs one plan action on run.motion (or the planner, for verify) and returns its result."""
    m, args = run.motion, a.args
    if a.name == "move_to":
        pitch = -60.0 if run.mode == "spoon" else -90.0
        # Hover correction: the top camera sees the marker and the food alike, so aiming by the
        # marker's offset cancels most calibration and arm error. aim = target + measured offset.
        applied, shifts = (0.0, 0.0), []
        for step in range(SERVO_STEPS + 1):
            aim = (target[0] + applied[0], target[1] + applied[1], target[2])
            pose = target_pose(args, aim, run.hover_dz, pitch, run.kin, run.arm.tool)
            r = m.move_to(pose, speed=args.get("speed", "normal"))
            enrich_move(r, pose, run)
            d = hover_offset(r, run.calib)
            done = d is None or math.dist(d, applied) < SERVO_DONE_MM
            if not r.ok or done or step == SERVO_STEPS:
                break
            if math.hypot(*d) > SERVO_MAX_MM:
                why = f"top camera sees the marker {math.hypot(*d):.0f} mm off: not trusted"
                r.note = f"{r.note}; {why}" if r.note else why
                break
            applied = d
            shifts.append([round(v, 1) for v in d])
        run.aim = aim  # the descent heads here, not back to the uncorrected target
        if shifts:
            r.measured["servo_mm"] = shifts
            why = f"re-aimed {math.hypot(*shifts[-1]):.0f} mm by the top camera"
            r.note = f"{r.note}; {why}" if r.note else why
        return r
    if a.name == "descend_until_contact":
        return m.descend_until_contact(float(args.get("max_dz", 60)), load=args.get("load"),
                                       expect_z=target[2], toward=run.aim)
    if a.name == "open":
        want = float(args.get("width_mm", (width or 30) + 15))
        r = m.open(min(want, run.gripper_max))
        if want > run.gripper_max:
            clamp = f"clamped {want:.0f} mm to the gripper max {run.gripper_max:.0f} mm"
            r.note = f"{r.note}; {clamp}" if r.note else clamp
        return r
    if a.name == "close":
        r = m.close(expect_mm=width)
        # Where the gripper really is once the jaws have shut, and how far that is from the food:
        # the cameras' tape position, moved by the arm's own tape-to-jaw offset, against the target
        # (both camera-found, so calibration error mostly cancels). One set of joints for the
        # offset: it is rigid, and commanded vs measured would add the sag to it.
        st = run.arm.state()
        enrich_move(r, st.tip, run)
        seen = r.measured.get("marker_xyz")
        if seen and target is not None:
            tape = run.kin.fk(st.q, "marker").xyz
            grab = [seen[i] + st.tip.xyz[i] - tape[i] for i in range(3)]
            r.error["to_food_mm"] = round(math.dist(grab[:2], target[:2]), 1)
        return r
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
        raise NeedHelp(f"Planner requested help: {args.get('question') or 'no reason given'}")
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


def tape_goal(run):
    """Where the tape should be: FK of the commanded joints at robot.yaml tool_offsets.marker, so
    sag and FK error count as error too."""
    tool = "marker" if "marker" in run.kin.tools else run.arm.tool
    return run.kin.fk(run.arm.state().goal, tool).xyz


def hover_offset(r: ActionResult, calib):
    """(dx, dy) mm: where the arm put the tape minus where the top camera sees it, at the tape's
    height; None when top didn't see it. Adding it to a target the top camera found
    lands the arm where the camera saw that target."""
    seen, goal = (r.measured.get("marker_uv") or {}).get("top"), r.predicted.get("tape_xyz")
    if seen is None or goal is None or "top" not in calib.names:
        return None
    x, y, _ = calib.at_height("top", seen, goal[2])
    return goal[0] - x, goal[1] - y


def enrich_move(r: ActionResult, pose: Pose, run) -> None:
    """Adds the camera metrics of a move: tip_uv (the jaw point, drawn as the gripper goal),
    tape_uv (where the commanded joints put the tape, tape_goal) against the seen marker_uv, the
    px error between those two, and mm_from_marker (the triangulated tape against tape_goal)."""
    from perception.marker import find_marker

    cameras, calib = run.cameras, run.calib
    after = r.t1 or time.time()
    tape = tape_goal(run)
    tip_uv, tape_uv, marker_uv, px = {}, {}, {}, {}
    for cam in PLAN_CAMS:
        if cam not in calib.names or cam not in cameras.names:
            continue
        tip_uv[cam] = [round(float(v), 1) for v in calib.project(cam, pose.xyz)]
        tape_uv[cam] = [round(float(v), 1) for v in calib.project(cam, tape)]
        # The predicted pixel steers the detector past yellow clutter elsewhere in the frame.
        uv = find_marker(fresh_frame(cameras, cam, after), run.marker_cfg, near=tuple(tape_uv[cam]))
        if uv is not None:
            marker_uv[cam] = [round(float(v), 1) for v in uv]
            px[cam] = round(math.dist(tape_uv[cam], marker_uv[cam]), 1)
    r.predicted.setdefault("tip_xyz", [round(v, 1) for v in pose.xyz])
    r.predicted["tape_xyz"] = [round(float(v), 1) for v in tape]
    r.predicted["tip_uv"] = tip_uv
    r.predicted["tape_uv"] = tape_uv
    r.measured["marker_uv"] = marker_uv
    r.error["px"] = px
    if len(marker_uv) >= 2:
        seen = [round(float(v), 1) for v in calib.triangulate(marker_uv)]
        r.measured["marker_xyz"] = seen
        r.error["mm_from_marker"] = round(math.dist(seen, tape), 1)


def position_error_mm(results: list[ActionResult]) -> float | None:
    """How far the gripper was from the food when it closed (cameras), else the last move_to's
    camera error (mm_from_marker) or FK error (xyz_mm)."""
    for r in reversed(results):
        if r.name == "close" and r.error.get("to_food_mm") is not None:
            return float(r.error["to_food_mm"])
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
    for cam, uv in (r.predicted.get("tape_uv") or {}).items():
        anns.setdefault(cam, []).append(
            {"type": "point", "uv": uv, "color": "green", "label": "tape expected", "hollow": True})
    for cam, uv in (r.measured.get("marker_uv") or {}).items():
        anns.setdefault(cam, []).append(
            {"type": "point", "uv": uv, "color": "green", "label": "tape seen"})
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


def _toward_base(mouth) -> tuple[float, float, float]:
    """Unit vector from the mouth straight back toward the arm's base axis, level."""
    n = math.hypot(mouth[0], mouth[1]) or 1.0
    return (-mouth[0] / n, -mouth[1] / n, 0.0)


def _out(mouth, d, mm: float):
    return tuple(mouth[i] + d[i] * mm for i in range(3))


def _approach_gate(run, mouth0, creeping: bool):
    """present's gate while the arm moves toward the face. Stops the move (a reason) on: stop;
    the face lost (also trips stop); the ultrasonic seeing the food within estop_cm of something
    on two readings in a row (trips stop: the arm then goes home); within stop_cm ("close
    enough", a normal end); within slow_cm while not creeping yet ("slow zone"); the head moving
    reaim_mm from where the approach aimed ("head moved")."""
    pc = run.present

    def gate() -> str | None:
        if run.stop.is_set():
            return f"stop: {run.stop.reason}"
        fs = run.face_state()
        if not fs.present or fs.looking_away:
            run.trip_stop("face lost while coming to the mouth")
            return "face lost"
        sure = run.food_cm(2)  # two readings in a row: one false echo does nothing
        if sure is not None and sure <= float(pc.get("estop_cm", 3)):
            run.trip_stop(f"ultrasonic: food {sure:.1f} cm from the face while moving")
            return "ultrasonic e-stop"
        if sure is not None and sure <= float(pc.get("stop_short_mm", 60)) / 10:
            return "close enough (ultrasonic)"
        now = run.food_cm(1)
        if not creeping and now is not None and now <= float(pc.get("slow_cm", 15)):
            return "slow zone"
        m = run.mouth_xyz()
        if m is not None and math.dist(m, mouth0) > float(pc.get("reaim_mm", 30)):
            return "head moved"
        return None

    return gate


def _bitten(run, near):
    """done() for _wait: the mouth opened, then closed while at the food (`near()`)."""
    seen_open = [False]

    def done() -> bool:
        fs = run.face_state()
        seen_open[0] = seen_open[0] or fs.mouth_open
        return seen_open[0] and not fs.mouth_open and near()

    return done


def deliver_tracked(run, obj: str, mode: str) -> bool:
    """Delivery to the tracked mouth; the VLM only verifies. Find a still face -> the stage point
    stage_mm out from the mouth toward the arm -> wait for an open mouth or "yes" -> slowly toward
    the mouth (creep once the ultrasonic is within slow_cm), stopping stop_short_mm short: the arm
    never feeds into the mouth, the person leans in the rest -> hold still (the ultrasonic no
    longer stops it: the person is meant to come close) until a bite (mouth open, then closed at
    the food) -> solid: let go; spoon: hold a second -> back out to the stage point -> home ->
    eaten = the tool is empty (VLM), with chewing seen by the face camera as the second sign."""
    pc = run.present
    short = float(pc.get("stop_short_mm", 60))
    run.arm.tool = "spoon" if mode == "spoon" else ("grasp" if "grasp" in run.kin.tools else "jaw")
    run.begin_record("deliver", obj, mode)
    run.enter("LISTENING", goal="deliver", attempt=1, action="looking for the face")
    if not _wait(run, lambda: run.mouth_xyz() is not None, READY_TIMEOUT_S):
        raise NeedHelp(f"No still face seen by the face camera within {READY_TIMEOUT_S:.0f} "
                       "seconds.")
    mouth = run.mouth_xyz()
    back = _toward_base(mouth)
    stage = _out(mouth, back, float(pc.get("stage_mm", 150)))
    run.motion.fence.around_mouth(stage, mouth)
    run.state.set(mouth_xyz=[round(v) for v in mouth])
    run.log.emit("mouth", xyz=mouth, how=run.mouth_track.how)
    run.enter("MOVING", action="to the front of the face")
    r = run.do("stage", run.motion.move_to, Pose(stage, 0.0), "slow", "deliver")
    if not r.ok and r.note.startswith("fence"):  # the straight line from the plate clips the
        run.do("home", run.motion.home, "slow")    # head zone: come from the base side instead
        r = run.do("stage", run.motion.move_to, Pose(stage, 0.0), "slow", "deliver")
    if not r.ok:
        raise NeedHelp(f"Move to the front of the face failed ({r.note}).")
    run.yes_evt.clear()
    run.enter("LISTENING", action="waiting for an open mouth or 'yes'")
    if not _wait(run, lambda: run.face_state().mouth_open or run.yes_evt.is_set(), READY_TIMEOUT_S):
        run.do("home", run.motion.home, "slow")
        raise NeedHelp(f"No open mouth and no \"yes\" within {READY_TIMEOUT_S:.0f} seconds in "
                       "front of the face.")
    run.enter("PRESENTING", action="coming to the mouth slowly")
    speed, reaims = "slow", int(pc.get("reaims", 3))
    while True:
        target = _out(mouth, back, short)
        creeping = speed == "creep"
        r = run.do("present", run.motion.approach, target, _approach_gate(run, mouth, creeping),
                   speed, run.food_cm)
        run.check()
        why = r.measured.get("stopped_by", "")
        if why == "slow zone":
            speed = "creep"
            run.state.set(action="close to the face: creeping")
            continue
        if why == "head moved" and reaims > 0 and (m := run.mouth_xyz()) is not None:
            reaims -= 1  # back out to the new stage point first, then come in again
            mouth, back = m, _toward_base(m)
            stage = _out(mouth, back, float(pc.get("stage_mm", 150)))
            run.motion.fence.around_mouth(stage, mouth)
            run.state.set(mouth_xyz=[round(v) for v in mouth], action="the head moved: re-aiming")
            run.do("stage", run.motion.move_to, Pose(stage, 0.0), "slow", "deliver")
            continue
        break
    run.state.set(action="holding still: lean in and take it")

    def near() -> bool:  # the mouth at the food: the ultrasonic, or the tracked mouth
        d, m = run.food_cm(1), run.mouth_xyz()
        food = run.arm.state().tip.xyz
        return ((d is not None and d <= float(pc.get("bite_cm", 3)))
                or (m is not None and math.dist(m, food) <= float(pc.get("bite_mm", 35))))

    bitten = _wait(run, _bitten(run, near), float(pc.get("bite_timeout_s", 15)))
    if bitten:
        if mode == "solid":
            run.do("open", run.motion.open, min(run.gripper_max, (run.width or 30) + 15))
        time.sleep(1.0)  # let them pull back with it (spoon: the lips scrape it off)
    run.enter("MOVING", action="back out")
    run.do("retract", run.motion.move_to, Pose(stage, run.arm.state().tip.pitch), "slow", "deliver")
    run.do("home", run.motion.home, "slow")
    chewing = _wait(run, lambda: run.face_state().chewing, 4.0)
    v = run.verify(f"Is the {run.tool_word} empty?")
    note = "; ".join(["bite seen" if bitten else "no bite",
                      "chewing" if chewing else "no chewing seen", f"present stopped by {why}"])
    run.end_record(v, v.ok, note=note)
    run.log.emit("eaten", ok=v.ok, bitten=bitten, chewing=chewing)
    return v.ok


def deliver(run, obj: str, mode: str) -> bool:
    """The fixed deliver sequence; the VLM only verifies. stage -> wait for an open mouth or a
    "yes" -> present behind the gate -> solid: open once the mouth stayed open 1 s; spoon: hold
    until the mouth closes (bite) or 8 s -> retract -> verify the tool is empty. With a tracked
    mouth (a calibrated face camera), deliver_tracked instead."""
    if run.tracking:
        return deliver_tracked(run, obj, mode)
    run.arm.tool = "spoon" if mode == "spoon" else "jaw"
    for name, what, where in (
        ("stage", "the spot in front of the mouth",
         "about 15 cm in front of the person's mouth, held level"),
        ("mouth", "the mouth", "right up to the person's lips"),
    ):
        if name not in run.motion.poses:
            run.teach_pose(name, what, where)
    run.begin_record("deliver", obj, mode)
    run.enter("MOVING", goal="deliver", attempt=1, action="stage")
    r = run.do("stage", run.motion.stage)
    if not r.ok:  # no point waiting for a mouth when the arm is not at the stage pose
        raise NeedHelp(f"Move to the spot in front of the mouth failed ({r.note}).")
    run.yes_evt.clear()
    run.enter("LISTENING", action="waiting for an open mouth or 'yes'")
    if not _wait(run, lambda: run.face_state().mouth_open or run.yes_evt.is_set(), READY_TIMEOUT_S):
        run.do("retract", run.motion.retract)
        raise NeedHelp(f"No open mouth and no \"yes\" within {READY_TIMEOUT_S:.0f} seconds at the "
                       "spot in front of the mouth.")
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
    if run.mode == "solid":
        # Aim the middle of the jaw gap at the food, not the fixed jaw tip (seen live: the fixed
        # jaw landed on the food and pushed it away). gap_dir: fixed -> moving jaw, gripper frame.
        half = min(run.gripper_max, (width or 30.0) + 15.0) / 2
        gap = (run.robot.get("gripper") or {}).get("gap_dir", [0.0, 0.0, 1.0])
        run.kin.tools["grasp"] = [half * float(v) for v in gap]
        run.arm.tool = "grasp"
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
        raise NeedHelp(f"Planner requested help: {plan.escalate_reason or 'no reason given'}")
    if width is not None and width > run.gripper_max * WIDE_FACTOR:
        run.end_record(None, False, note=f"{width:.0f} mm wide")
        raise NeedHelp(f"{run.object} measures {width:.0f} mm wide. The gripper opens "
                       f"{run.gripper_max:.0f} mm at most.")
    run.width = width
    note, moved = "", False
    if target is None:
        note = "no target: the top camera has no box"
    else:
        run.enter("MOVING", action="")
        run.aim = target  # move_to's hover correction shifts it
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
