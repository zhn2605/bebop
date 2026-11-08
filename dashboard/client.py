"""Publisher: push a program's State, attempts and annotated camera frames to the standalone
dashboard (dashboard.server) at MAX_FPS, and run the Stop/Resume/Release presses it hands back.

Starts the dashboard in the background when nothing answers on the port; it keeps running after
this program exits. Never raises into the caller: a missing dashboard only prints once.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable

from core.state import State, jsonable
from dashboard.server import MAX_FPS, ROOT, running


def spawn(port: int) -> bool:
    """Starts `python -m dashboard.server` detached from this terminal; True once it answers."""
    log = ROOT / "data" / "dashboard.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "ab") as out:
            subprocess.Popen(
                [sys.executable, "-m", "dashboard.server", "--port", str(port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True,
            )
    except OSError as e:
        print(f"dashboard: {e}")
        return False
    deadline = time.time() + 15
    while time.time() < deadline:
        if running(port):
            return True
        time.sleep(0.2)
    return False


class Publisher:
    """`controls` maps "stop" / "resume" / "release" / "say" to a callable taking one string (the
    reason, or for "say" the text). With
    `arm`, the arm state goes into `state` at 5 Hz; with `calib` too, every stream shows where the
    calibration puts the FK tip (hollow magenta). `cameras` may be None (no streams)."""

    def __init__(
        self,
        source: str,
        state: State,
        cameras=None,
        *,
        arm=None,
        calib=None,
        controls: dict[str, Callable[[str], None]] | None = None,
        port: int = 8000,
        own_cameras: bool = False,
    ):
        self.source, self.state, self.cameras, self.arm, self.calib = (
            source, state, cameras, arm, calib
        )
        self.controls = controls or {}
        self.port, self.own_cameras = port, own_cameras
        self.alive = False
        self.conn: http.client.HTTPConnection | None = None
        self.sent_attempts: str | None = None
        self.sent_n = 0
        self.rejected = False
        self.busy = ""
        self.thread: threading.Thread | None = None

    def start(self) -> Publisher:
        url = f"http://localhost:{self.port}"
        if running(self.port):
            print(f"dashboard: {url}")
        elif spawn(self.port):
            print(f"dashboard: {url}  (started in the background; it keeps running)")
        else:
            print("dashboard: did not start; see data/dashboard.log (or run `uv run dashboard`)")
        self.alive = True
        self.thread = threading.Thread(target=self._run, name="publisher", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.alive = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        self._post({"source": self.source, "pid": os.getpid(), "bye": True})
        if self.conn is not None:
            self.conn.close()
        if self.own_cameras and self.cameras is not None:
            self.cameras.stop()

    def _run(self) -> None:
        period, n, down = 1.0 / MAX_FPS, 0, False
        while self.alive:
            t = time.monotonic()
            try:
                if self.arm is not None and n % 2 == 0:
                    self.state.set(arm=self.arm.state())
                reply = self._post(self._body())
                if reply is None:
                    if not down:
                        print(f"dashboard: lost localhost:{self.port}; retrying (`uv run dashboard`)")
                    down = True
                    time.sleep(2.0)
                    continue
                if down:
                    print("dashboard: reconnected")
                    down = False
                busy = reply.get("busy", "")
                if busy != self.busy:
                    if busy:
                        print(f"dashboard: another {busy} is still running (it holds the page, and likely the arm and "
                              "cameras); stop it first")
                    self.busy = busy
                if busy:
                    self.sent_attempts = None
                    time.sleep(1.0)
                    continue
                if reply.get("attempts_n", self.sent_n) != self.sent_n:
                    self.sent_attempts = None  # the page lost them (restarted): resend
                for cmd, arg in reply.get("commands", []):
                    fn = self.controls.get(cmd)
                    if fn is not None:
                        fn(arg)
            except Exception as e:  # noqa: BLE001  (the dashboard must never take the arm down)
                print(f"dashboard: {type(e).__name__}: {e}")
                time.sleep(1.0)
            n += 1
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    def _body(self) -> dict:
        names = list(self.cameras.names) if self.cameras is not None else []
        body = {
            "source": self.source,
            "pid": os.getpid(),
            "state": jsonable(self.state.snapshot()),
            "controls": sorted(self.controls),
            "cameras": names,
        }
        tip = None
        if self.calib is not None:
            got = self.state.get("arm")
            tip = (got or {}).get("tip", {}).get("xyz")
        frames = {}
        for cam in names:
            anns = list(self.state.annotations.get(cam) or [])
            if tip is not None and cam in getattr(self.calib, "P", {}):
                uv = self.calib.project(cam, tip)
                anns.append({"type": "point", "uv": uv, "hollow": True, "color": "magenta",
                             "label": "gripper from joints"})
            jpeg = self.cameras.jpeg(cam, anns)
            if jpeg:
                frames[cam] = base64.b64encode(jpeg).decode("ascii")
        body["frames"] = frames
        attempts = self.state.get_attempts()
        text = json.dumps(attempts)
        if text != self.sent_attempts:
            body["attempts"] = attempts
            self.sent_attempts, self.sent_n = text, len(attempts)
        return body

    def _post(self, body: dict) -> dict | None:
        """POST /push; the reply, or None when the dashboard is not there."""
        data = json.dumps(body).encode()
        for _ in range(2):  # a kept-alive connection the server dropped fails once
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                self.conn.request("POST", "/push", data, {"Content-Type": "application/json"})
                r = self.conn.getresponse()
                payload = r.read()
                if r.status == 200:
                    return json.loads(payload)
                if not self.rejected:
                    print(f"dashboard: push rejected, HTTP {r.status}: {payload[:200]!r}")
                    self.rejected = True
                return {}
            except (OSError, http.client.HTTPException, ValueError):
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                self.sent_attempts = None  # a restarted dashboard needs them again
        return None


def for_tool(source: str, arm, stop, *, fake: bool = False, port: int = 8000) -> Publisher:
    """A started Publisher for a tool (arm_check, teach): its own State, the real cameras from
    config/cameras.yaml when they open (none with --fake), config/calib.json for the tip overlay,
    and Stop wired to `stop`. Publisher.stop() also stops the cameras."""
    import yaml

    from perception.cameras import Cameras

    cameras, calib = None, None
    if not fake:
        p = ROOT / "config" / "cameras.yaml"
        cameras = Cameras(yaml.safe_load(p.read_text()) if p.is_file() else {}).start()
        cal_path = ROOT / "config" / "calib.json"
        if cal_path.is_file():
            from perception.calib import Calibration

            calib = Calibration.load(cal_path)
    state = State({"stop": stop})
    state.set(stage=source.upper(), task=source, fake=fake)
    return Publisher(
        source, state, cameras, arm=arm, calib=calib, controls={"stop": stop.trip}, port=port,
        own_cameras=True,
    ).start()
