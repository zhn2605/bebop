"""The orchestrator and CLI: READY -> acquire (plan, act, measure, verify, re-plan) -> deliver.

python -m core.run [--task TEXT | --voice] --mode solid|spoon [--fake] [--provider P] [--model M]
                   [--no-dashboard] [--port 8000] [--esp32 PORT] [--ntfy TOPIC] [--budget USD]

--task runs one task and exits. Without it the run listens: utterances -> intent() -> tasks.
Stop: a voice stop word, the ESP32 button, space in this terminal, or the dashboard. The arm
freezes, then goes home slowly and holds there with torque on; a stop on the way freezes it where it
is. Only the dashboard's Release turns torque off. Ctrl+C: freeze, fold to rest, torque off, exit 130.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import yaml

from core import steps
from core.notify import Notifier
from core.state import RUNS_DIR, EventLog, State
from core.steps import Abort, NeedHelp
from core.types import JOINTS, ActionResult, FaceState, Flag, Stopped, Verdict
from perception.mouth import EYES_MM, MouthTrack, locate_mouth

CONFIG = Path("config")
LED = {"READY": "green", "LISTENING": "green", "THINKING": "yellow", "MOVING": "yellow",
       "PRESENTING": "yellow", "STOPPED": "red", "HELP": "red"}
YES = {"yes", "yeah", "yep", "ready", "ok", "okay", "sure"}
CONTINUE = ("continue", "resume", "go on", "keep going")
SAG = "Torque off: the arm sags and drops. Keep hands and faces clear."
NO_FACE = FaceState(0.0, False, False, 0.0, 0.0, 0.0, False)


def load_config() -> dict[str, dict]:
    """config/<stem>.yaml for each stem; {} for a missing file."""
    cfg = {}
    for stem in ("robot", "cameras", "workspace", "brain", "poses", "fake"):
        path = CONFIG / f"{stem}.yaml"
        cfg[stem] = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    return cfg


class Runner:
    def __init__(self, args: argparse.Namespace, cfg: dict[str, dict]):
        self.args, self.cfg = args, cfg
        self.robot, self.ws, self.brain = cfg["robot"], cfg["workspace"], cfg["brain"]
        self.fake = cfg["fake"] if args.fake else None
        self.stop, self.pause = Flag("stop"), Flag("pause")
        self.state = State({"stop": self.stop, "pause": self.pause})
        self.log = EventLog(RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))
        self.notifier = Notifier(args.ntfy or os.environ.get("NTFY_TOPIC", ""))
        self.url = steps.lan_url(args.port)
        self.resume_evt, self.yes_evt = threading.Event(), threading.Event()
        self.heard: queue.Queue[str] = queue.Queue()
        self.max_attempts = int(args.attempts or self.brain.get("max_attempts", 3))
        self.budget = float(args.budget if args.budget is not None
                            else self.brain.get("run_budget_usd", 1.0))
        self.hover_dz = float(self.ws.get("hover_dz", 40))
        self.rim_z = float((self.fake or {}).get("bowl_rim_z", self.ws.get("bowl_rim_z", 30)))
        self.gripper_max = float((self.robot.get("gripper") or {}).get("max_mm", 60))
        self.marker_cfg = cfg["cameras"].get("marker") or {}
        # fake.yaml's `present:` overrides some of it (the fake person can't lean in)
        fake_present = (self.fake or {}).get("present") or {}
        self.present = {**(self.ws.get("present") or {}), **fake_present}
        self.mode, self.object, self.width = args.mode, "", None
        self.aim = None  # where the grab descends to: the target after the hover correction
        self.cost, self.released, self.alive = 0.0, False, True
        self.cur: int | None = None
        self.results: list[ActionResult] = []
        self.lessons: list[str] = []
        self.plan_anns: dict[str, list] = {}
        self.esp32 = self.listener = self.publisher = None

    @property
    def tool_word(self) -> str:
        return "spoon" if self.mode == "spoon" else "gripper"

    # -- setup ----------------------------------------------------------------------------------

    def build(self) -> None:
        """Arm, motion, calibration, cameras, face, planner and ESP32, each per its contract."""
        from brain.planner import Planner
        from hw.arm import Arm
        from motion.fence import Fence
        from motion.kinematics import Kinematics
        from motion.primitives import Motion
        from perception.calib import Calibration
        from perception.cameras import Cameras, FakeWorld
        from perception.face import FaceSignals

        self.kin = Kinematics(self.robot["geometry"], self.robot["tool_offsets"],
                              self.robot.get("joints"))
        if self.args.speed != 1.0:  # --speed scales every arm and gripper move
            self.robot["speed_deg_s"] = float(self.robot.get("speed_deg_s", 60)) * self.args.speed
            grip = self.robot.setdefault("gripper", {})
            grip["speed_pct_s"] = float(grip.get("speed_pct_s", 100)) * self.args.speed
        self.arm = Arm(self.robot, self.stop, fake=self.args.fake, kin=self.kin)
        taught = {k: v for k, v in self.cfg["poses"].items() if k != "fake"}
        poses = taught
        if self.fake is not None:  # the fake world's poses, from fake.yaml or poses.yaml's fake:
            poses = self.fake.get("poses") or self.cfg["poses"].get("fake") or taught
        self.motion = Motion(self.arm, self.kin, Fence(self.ws), self.ws, poses, pause=self.pause)
        if self.fake is not None:  # the fake world brings its own calibration (fake.yaml cameras)
            world = FakeWorld(lambda: self.arm.state().tip, None,
                              tuple(self.fake.get("food_xyz", (250.0, 0.0, 15.0))),
                              float(self.fake.get("food_radius_mm", 15.0)),
                              held_fn=lambda: self.arm._holding,
                              chain_fn=lambda: self.kin.chain(self.arm.state().q, self.arm.tool))
            self.calib = world.calib
            self.cameras = Cameras(self.cfg["cameras"], fake=world)
        else:
            path = CONFIG / "calib.json"
            if not path.exists():
                sys.exit(f"{path} is missing: run python -m tools.calibrate first")
            self.calib = Calibration.load(path)
            bad = self.calib.implausible(float(self.ws.get("table_z", 0)))
            if bad:
                sys.exit(f"{path} is no real camera ({'; '.join(bad)}): rerun tools.calibrate")
            self.cameras = Cameras(self.cfg["cameras"])
        self.face = FaceSignals(self.cameras, "face", self.cfg["cameras"], fake=self.args.fake)
        # The side camera's view of the mouth gives its distance (perception/mouth.py); MediaPipe
        # finds a face up to about a three-quarter view, so a strict profile falls back to the eyes.
        self.side_face = None
        if not self.args.fake and "face" in self.calib.names and "side" in self.calib.names:
            self.side_face = FaceSignals(self.cameras, "side", self.cfg["cameras"])
        self.mouth_track = MouthTrack()
        self._mouth_t = 0.0
        self.eyes_mm = float((self.cfg["cameras"].get("face") or {}).get("eyes_mm", EYES_MM))
        brain = dict(self.brain)
        brain.update({k: v for k, v in (("provider", self.args.provider),
                                        ("model", self.args.model)) if v})
        self.planner = Planner(brain, self.log)
        if self.args.esp32:
            from hw.esp32 import Esp32, find_port

            self.esp32 = Esp32(find_port(self.args.esp32), self.stop)

    def start(self) -> None:
        if not self.args.fake:
            from hw.takeover import take_over

            take_over()  # an older run still holding the arm and cameras gets folded and stopped
        self.cameras.start()
        self.face.start()
        if self.side_face is not None:
            self.side_face.start()
        self.arm.connect()
        self.state.set(fake=self.args.fake, mode=self.mode, max_attempts=self.max_attempts,
                       budget_usd=self.budget)
        if not self.args.no_dashboard:
            from dashboard.client import Publisher

            controls = {"stop": self.trip_stop, "resume": self.resume, "release": self.release,
                        "say": self.on_utterance}
            self.publisher = Publisher("bebop", self.state, self.cameras, controls=controls,
                                       port=self.args.port).start()
        if self.args.voice or not self.args.task:
            from voice.listener import Listener

            self.listener = Listener(self.brain.get("voice") or {}, self.on_utterance,
                                     lambda word: self.trip_stop(f"voice: {word}"))
            self.listener.start()
        steps.keyboard(self.trip_stop)
        threading.Thread(target=self.telemetry, name="telemetry", daemon=True).start()

    def close(self) -> None:
        self.alive = False
        for part in (self.listener, self.face, self.side_face, self.cameras):
            try:
                if part is not None:
                    part.stop()
            except Exception as e:
                print(f"close: {type(part).__name__}: {e}")
        if self.esp32 is not None:
            self.esp32.close()
        if getattr(self.arm, "torque_on", True):
            print(SAG)
        try:
            self.arm.disconnect()
        except Exception as e:
            print(f"close: arm: {e}")
        if self.publisher is not None:
            self.publisher.stop()
        self.log.close()

    # -- flags, voice, telemetry (other threads) ------------------------------------------------

    def trip_stop(self, reason: str) -> None:
        """Every stop source lands here: freeze. Never torque off (a repeated "stop" used to drop
        the arm); that is the dashboard's Release only."""
        self.stop.trip(reason)
        self.log.emit("flag", flag="stop", reason=reason)
        print(f"\nSTOP: {reason}")

    def resume(self, by: str) -> None:
        """Clears stop (the arm then goes home slowly) and continues out of HELP."""
        self.log.emit("resume", by=by)
        self.stop.clear()
        self.resume_evt.set()

    def release(self, reason: str) -> None:
        """Torque off: stop first so a running move freezes, then cut torque. The arm sags."""
        self.stop.trip(reason)
        time.sleep(0.1)  # a running move checks stop every tick
        print(f"\n{SAG} ({reason})")
        self.arm.torque(False)
        self.released = True
        self.log.emit("release", reason=reason)

    def on_utterance(self, text: str) -> None:
        self.log.emit("heard", text=text)
        self.state.set(voice=text)
        low = text.lower()
        stage = self.state.get("stage")
        if stage in ("STOPPED", "HELP") and any(w in low for w in CONTINUE):
            self.resume("voice")
        elif set(re.findall(r"[a-z']+", low)) & YES:
            self.yes_evt.set()
        if stage in ("READY", "LISTENING") and self.cur is None:
            self.heard.put(text)

    @property
    def tracking(self) -> bool:
        """The mouth can be tracked: the face camera is calibrated, or the fake world has one."""
        return "face" in self.calib.names or bool((self.fake or {}).get("mouth_xyz"))

    def mouth_xyz(self):
        """The tracked mouth (base frame mm, the median of the last second), or None while no
        still face is seen."""
        if self.fake is not None and self.fake.get("mouth_xyz"):
            return tuple(float(v) for v in self.fake["mouth_xyz"])
        fs = self.face_state()
        if fs.t != self._mouth_t:
            self._mouth_t = fs.t
            side = self.side_face.state if self.side_face is not None else None
            got = locate_mouth(self.calib, fs, side, self.eyes_mm)
            if got is not None:
                self.mouth_track.add(fs.t, *got)
        return self.mouth_track.stable(time.time())

    def food_cm(self, n: int = 1) -> float | None:
        """Ultrasonic distance from the food to what is in front of it, cm: the largest of the
        last n readings (n > 1 ignores a lone false echo), minus how far the sensor sits behind
        the food. None without readings or when one of them had no echo (nothing close)."""
        rec = list(getattr(self.esp32, "recent", []))[-n:]
        if len(rec) < n or any(d is None for d in rec):
            return None
        return max(rec) - float(self.present.get("sensor_behind_food_cm", 0))

    def dist_cm(self) -> float | None:
        return getattr(self.esp32, "dist_cm", None)

    def face_state(self) -> FaceState:
        return self.face.state or NO_FACE

    def telemetry(self) -> None:
        """5 Hz: arm, face, distance and voice into State; pause while looking away in acquire."""
        failed = False
        while self.alive:
            try:
                fs = self.face_state()
                fields = {"arm": self.arm.state(), "face": fs, "dist_cm": self.dist_cm()}
                if self.tracking and "face" in self.cameras.names:
                    self.state.annotate("face", self.mouth_annotations(fs))
                if self.listener is not None:
                    fields["voice_error"] = self.listener.error
                self.state.set(**fields)
                # Without a face camera looking_away is always true: never pause on it then.
                moving = (self.args.gaze_pause and self.state.get("goal") == "acquire"
                          and "face" in self.cameras.names and self.state.get("stage") == "MOVING")
                if moving and fs.looking_away and not self.pause.is_set():
                    self.pause.trip("looking away")
                    self.log.emit("flag", flag="pause", reason="looking away")
                elif self.pause.is_set() and not (moving and fs.looking_away):
                    self.pause.clear()
                    self.log.emit("flag", flag="pause", reason="cleared")
            except Exception as e:
                if not failed:
                    print(f"telemetry: {type(e).__name__}: {e}")
                failed = True
            time.sleep(0.2)

    def mouth_annotations(self, fs: FaceState) -> list[dict]:
        """The face stream's overlay: the mouth, and how far the food is from it."""
        if fs.mouth_uv is None:
            return []
        m = self.mouth_xyz()
        label = "mouth"
        if m is not None:
            food = self.arm.state().tip.xyz
            label = f"mouth: food {math.dist(m, food) / 10:.0f} cm away ({self.mouth_track.how})"
        if fs.chewing:
            label += ", chewing"
        return [{"type": "point", "uv": list(fs.mouth_uv), "color": "magenta", "label": label}]

    def gate(self) -> str | None:
        """present's gate: a reason to stop short (ultrasonic, face lost, stop), else None."""
        if self.stop.is_set():
            return f"stop: {self.stop.reason}"
        d = self.dist_cm()
        if d is not None and d <= float(self.present.get("ultrasonic_stop_cm", 6)):
            return f"ultrasonic {d:.1f} cm"
        fs = self.face_state()
        if not fs.present or fs.looking_away:
            self.trip_stop("face lost during present")
            return "face lost"
        return None

    # -- main thread ----------------------------------------------------------------------------

    def check(self) -> None:
        if self.stop.is_set():
            raise Stopped(self.stop.reason)

    def enter(self, stage: str, **fields) -> None:
        self.state.set(stage=stage, led=LED.get(stage, ""), **fields)
        if self.esp32 is not None:
            self.esp32.send_state(stage)
        self.log.emit("stage", stage=stage, **fields)
        shown = " ".join(f"{k}={v}" for k, v in fields.items() if v not in ("", None, {}))
        print(f"[{stage}] {shown}")

    def spent(self, cost: float | None, latency_s: float | None = None) -> None:
        self.cost += cost or 0.0
        fields = {"cost_usd": round(self.cost, 4), "vlm_cost_usd": cost}
        if latency_s:
            fields["vlm_latency_s"] = round(latency_s, 2)
        self.state.set(**fields)

    def budget_check(self) -> None:
        if self.cost >= self.budget:
            raise Abort(f"VLM budget ${self.budget:.2f} spent (${self.cost:.3f})")

    def grab(self, cams=steps.PLAN_CAMS) -> dict[str, bytes]:
        return {c: j for c in cams if c in self.cameras.names and (j := self.cameras.jpeg(c))}

    def verify(self, question: str) -> Verdict:
        self.budget_check()
        self.state.set(action=f"verify: {question}")
        v = self.planner.verify(question, self.grab())
        self.spent(v.cost_usd)
        self.log.emit("verdict", question=question, verdict=v)
        return v

    def show(self, extra: dict[str, list]) -> None:
        """The plan's overlay plus extra (the last move's tip and marker) on every stream."""
        for cam in self.cameras.names:
            self.state.annotate(cam, self.plan_anns.get(cam, []) + extra.get(cam, []))

    def begin_record(self, goal: str, obj: str, mode: str, **fields) -> None:
        n = sum(a.get("goal") == goal for a in self.state.get_attempts()) + 1
        self.results = []
        self.cur = self.state.add_attempt({"n": fields.pop("n", n), "goal": goal, "task": obj,
                                           "mode": mode, "results": [], "frames": {}, "ok": None,
                                           **fields})

    def end_record(self, verdict: Verdict | None, ok: bool, **fields) -> None:
        tag = f"a{self.cur + 1}-after"
        frames = self.state.get_attempts()[self.cur].get("frames") or {}
        for cam, jpeg in self.grab().items():
            frames[f"after {cam}"] = self.log.save_frame(cam, jpeg, tag)
        self.state.update_attempt(self.cur, verdict=verdict, ok=ok, frames=frames, **fields)
        self.log.emit("attempt", index=self.cur, ok=ok, verdict=verdict, **fields)

    def record(self, r: ActionResult) -> None:
        self.results.append(r)
        if self.cur is not None:
            self.state.update_attempt(self.cur, results=self.results)
        self.log.emit("action", attempt=self.cur, result=r)
        if r.name in ("move_to", "close") and r.predicted.get("tip_uv"):
            self.state.set(err_px=r.error.get("px"),
                           err_mm=r.error.get("mm_from_marker", r.error.get("xyz_mm")))
            self.show(steps.move_annotations(r))
        print(f"  {r.name}: {'ok' if r.ok else 'FAIL'} {r.note}".rstrip())

    def do(self, name: str, fn, *a, args: dict | None = None) -> ActionResult:
        """Runs one action and records its result. Stopped propagates (after recording);
        other errors become a failed result so the planner sees them."""
        self.check()
        self.state.set(action=name, action_args=args or {})
        t0 = time.time()
        try:
            r = fn(*a)
        except Stopped as e:
            r = getattr(e, "result", None) or ActionResult(
                name, False, note=f"stopped: {e}", t0=t0, t1=time.time(), stopped=True)
            self.record(r)
            raise
        except (NeedHelp, Abort):
            raise
        except Exception as e:
            traceback.print_exc()
            r = ActionResult(name, False, note=f"error: {type(e).__name__}: {e}", t0=t0,
                             t1=time.time())
        self.record(r)
        return r

    def capabilities(self) -> dict:
        try:
            reach = round(self.kin.reach("spoon" if self.mode == "spoon" else "jaw"))
        except Exception:
            reach = None
        return {"gripper_max_mm": self.gripper_max, "reach_mm": reach,
                "tool": "spoon" if self.mode == "spoon" else "jaw", "hover_dz": self.hover_dz}

    def help(self, reason: str) -> None:
        """HELP: ping the caregiver, wait for 'continue' or the dashboard (stop raises)."""
        reason = f"{reason} To try again, press Resume or say \"continue\"."
        self.enter("HELP", help=reason, action="waiting for the caregiver")
        self.notifier.help(f"{reason}. Say 'continue' or press Resume on the dashboard.", self.url)
        if self.args.task:  # one request from the command line: report and exit, never linger
            raise Abort(reason)
        print(f"HELP: {reason}")
        self.resume_evt.clear()
        while not self.resume_evt.wait(0.2):
            self.check()
        self.resume_evt.clear()
        self.state.set(help="")

    def _wait_continue(self, msg: str, action: str) -> None:
        self.enter("HELP", help=msg, action=action)
        print(f"HELP: {msg}")
        self.resume_evt.clear()
        while not self.resume_evt.wait(0.2):
            self.check()
        self.resume_evt.clear()

    def teach_pose(self, name: str, what: str, where: str) -> None:
        """A delivery pose is missing: teach it here instead of failing on every resume. 'continue'
        once the caregiver holds the arm (the joints go limp, the gripper keeps the food), then
        again once it is in place; the pose is saved to config/poses.yaml."""
        self._wait_continue(f"Position not set: {what}. Hold the arm, then press Resume or say "
                            "\"continue\". The arm then goes loose to be moved by hand. The gripper "
                            "keeps holding the food.", f"setting {what}")
        self.arm.torque(False, list(JOINTS))
        try:
            self._wait_continue(f"Move the gripper {where}. Then press Resume or say "
                                "\"continue\" to save the position.", f"setting {what}")
        finally:
            self.arm.torque(True, list(JOINTS))
        q = self.arm.state().q
        self.motion.poses[name] = {j: round(float(q[j]), 1) for j in JOINTS}
        self.motion._segment()
        path = CONFIG / "poses.yaml"
        poses = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
        poses[name] = self.motion.poses[name]
        path.write_text("# Taught joint poses, LeRobot degrees (tools.teach or the run's teach gate).\n"
                        + yaml.safe_dump(poses, default_flow_style=None, sort_keys=False))
        self.log.emit("taught", name=name, q=self.motion.poses[name])
        self.state.set(help="")
        print(f"Saved position: {what}.")

    def task(self, obj: str, mode: str) -> None:
        """acquire then deliver. A failed goal goes through HELP and runs again; a stop goes through
        STOPPED and, once resumed, runs the goal again from home. Release cancels the task."""
        self.object, self.mode, self.lessons = obj, mode, []
        self.arm.tool = "spoon" if mode == "spoon" else "jaw"
        self.state.set(task=obj, mode=mode)
        self.log.emit("task", object=obj, mode=mode)
        goals = (
            (lambda: steps.acquire(self), f"Pickup of {obj} failed after {self.max_attempts} tries."),
            (lambda: steps.deliver(self, obj, mode),
             f"Food still in the {self.tool_word} after delivery."),
        )
        try:
            for run_goal, failed in goals:
                while True:
                    try:
                        try:
                            if run_goal():
                                break
                            reason = failed
                        except NeedHelp as e:
                            reason = str(e)
                        self.help(reason)
                    except Stopped:
                        if not self.stopped():
                            self.log.emit("task_cancelled", object=obj)
                            return
            self.log.emit("task_done", object=obj)
            print(f"done: {obj}")
        except Abort as e:
            self.log.emit("abort", reason=str(e))
            self.notifier.info(f"Task stopped: {e}")
            print(f"abort: {e}")
            self.home()
        finally:
            self.cur = None

    def stopped(self) -> bool:
        """STOPPED: the arm froze. After a second it goes home slowly and holds there, torque on
        (nothing drops); a stop on the way freezes it where it is, and it is not tried again.
        Then it waits for Resume or "continue" before the task goes on. False when torque was
        released meanwhile (the task is cancelled)."""
        if self.cur is not None:  # close the interrupted attempt's record
            self.state.update_attempt(self.cur, ok=False, note=f"stopped: {self.stop.reason}")
            self.cur = None
        why, auto_home, at_home = self.stop.reason, True, False
        while True:
            try:
                if auto_home and not self.released:
                    auto_home = False
                    self.enter("STOPPED", reason=why, action="going home slowly in a second")
                    time.sleep(1.0)  # the freeze settles and hands get clear before it moves
                    if not self.released:
                        self.stop.clear()
                        self.home(slow=True)
                        at_home = True
                self.enter("STOPPED", reason=why, action="holding: Resume or say 'continue'")
                self.resume_evt.clear()
                while not self.resume_evt.wait(0.2):
                    pass
                self.resume_evt.clear()
                self.stop.clear()
                released, self.released = self.released, False
                if released:
                    self.arm.torque(True)  # sets goal = present first, so nothing jumps
                if released or not at_home:
                    self.home(slow=True)
                return not released
            except Stopped:  # stopped again while moving: hold where it froze
                why, at_home = self.stop.reason, False

    def home(self, slow: bool = False) -> None:
        self.enter("MOVING", action="home (slow)" if slow else "home")
        self.do("home", self.motion.home, "slow" if slow else "normal")

    def ready(self) -> None:
        self.cur = None
        self.state.set(goal="", attempt=0, action="", task="", help="", target_xyz=None)
        self.plan_anns = {}
        self.show({})
        self.enter("LISTENING" if self.listener else "READY")

    def run(self) -> int:
        self.build()
        try:
            self.start()
            self.arm.torque(True)
            try:
                self.home()
            except Stopped:
                self.stopped()
            self.ready()
            if self.args.task:
                intent = self.planner.intent(self.args.task)
                self.log.emit("intent", intent=intent)
                self.task(intent.object or self.args.task, self.args.mode)
                return 0
            while True:
                if self.stop.is_set():  # a stop while idle
                    self.stopped()
                    self.ready()
                try:
                    text = self.heard.get(timeout=0.5)
                except queue.Empty:
                    continue
                self.enter("THINKING", action=f"intent: {text}")
                intent = self.planner.intent(text)
                self.log.emit("intent", intent=intent)
                if intent.kind == "request":
                    self.task(intent.object or text, intent.mode or self.args.mode)
                self.ready()
        except KeyboardInterrupt:
            self.stop.trip("ctrl-c")
            self.log.emit("flag", flag="stop", reason="ctrl-c")
            try:
                self.arm.freeze()
                # Ctrl+C is the operator's own stop, not a danger stop: fold to rest before
                # torque goes off (the user: "it should first try to return home, not drop").
                self.stop.clear()
                print("\nctrl-c: folding to rest, then torque off")
                self.arm.disconnect()
            except Exception as e:
                print(f"ctrl-c: {e}")
            return 130
        finally:
            self.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m core.run", description=__doc__.split("\n")[0])
    p.add_argument("--task", help="one request, e.g. 'feed me a donut hole'; then exit")
    p.add_argument("--voice", action="store_true", help="listen (the default without --task)")
    p.add_argument("--mode", choices=("solid", "spoon"), default="solid")
    p.add_argument("--fake", action="store_true", help="fake arm and cameras, real VLM")
    p.add_argument("--provider", help="auto | claude-code | gemini (default: brain.yaml)")
    p.add_argument("--model", help="plan/verify model (default: brain.yaml)")
    p.add_argument("--no-dashboard", action="store_true", help="push nothing to the dashboard")
    p.add_argument("--port", type=int, default=8000, help="dashboard port (uv run dashboard)")
    p.add_argument("--esp32", metavar="PORT", help="ESP32 serial port, or auto to find it by name")
    p.add_argument("--ntfy", metavar="TOPIC", help="ntfy.sh topic for HELP (or $NTFY_TOPIC)")
    p.add_argument("--gaze-pause", action="store_true",
                   help="pause the grab while the face camera sees the user look away")
    p.add_argument("--speed", type=float, default=1.0,
                   help="multiplier for every arm and gripper speed, e.g. 1.5")
    p.add_argument("--attempts", type=int, help="tries per goal before HELP (brain.yaml max_attempts)")
    p.add_argument("--budget", type=float, help="VLM USD for the run (brain.yaml run_budget_usd)")
    args = p.parse_args(argv)
    return Runner(args, load_config()).run()


if __name__ == "__main__":
    sys.exit(main())
