"""The standalone dashboard. Leave it running: `uv run dashboard` (python -m dashboard.server).

Whatever drives the arm (core.run, calibrate, arm_check, teach) pushes its state, attempts and
annotated camera frames here through dashboard.client.Publisher, and gets back the Stop/Resume/
Release presses. The cameras stay with that program: a V4L2 camera opens in one process only.
Between programs the page shows IDLE with the last run's frames and attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import os
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import Request

from core.state import RUNS_DIR

ROOT = Path(__file__).resolve().parents[1]
INDEX = Path(__file__).with_name("index.html")
MAX_FPS = 10
LIVE_S = 3.0  # a program that has not pushed for this long is gone


class Hub:
    """The latest push from the program driving the arm, and the presses waiting for it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.source, self.pid, self.t = "", None, 0.0
        self.state: dict = {}
        self.attempts: list = []
        self.names: list[str] = []
        self.frames: dict[str, tuple[int, bytes]] = {}
        self.controls: list[str] = []
        self.pending: list[str] = []
        self.seq = 0

    def push(self, body: dict) -> dict:
        """Takes one push. The reply: the presses queued since the last push and how many attempts
        the page has (the program resends them when that is off), or `busy` while another program
        is live: one program at a time, so a fake run cannot take the page from a real one."""
        with self._lock:
            pid = body.get("pid")
            if pid != self.pid:
                if body.get("bye"):
                    return {}
                if self.live():
                    return {"busy": self.source}
                # a new program: the last one's attempts and frames go
                self.attempts, self.frames, self.pending = [], {}, []
                self.source, self.pid = str(body.get("source", "?")), pid
            if body.get("bye"):
                self.t = 0.0
                return {}
            self.t = time.time()
            self.state = body.get("state") or {}
            self.controls = list(body.get("controls") or [])
            self.names = list(body.get("cameras") or [])
            if body.get("attempts") is not None:
                self.attempts = body["attempts"]
            for cam, b64 in (body.get("frames") or {}).items():
                self.seq += 1
                self.frames[cam] = (self.seq, base64.b64decode(b64))
            out, self.pending = self.pending, []
            return {"commands": out, "attempts_n": len(self.attempts)}

    def live(self) -> bool:
        return self.pid is not None and time.time() - self.t < LIVE_S

    def press(self, cmd: str) -> str | None:
        """Queues a press for the connected program; the reason when there is none to take it."""
        with self._lock:
            if not self.live():
                return "nothing is connected"
            if cmd not in self.controls:
                return f"{self.source} has no {cmd}"
            self.pending.append(cmd)
            return None

    def snapshot(self) -> dict:
        with self._lock:
            s = copy.deepcopy(self.state)
            live, source, t = self.live(), self.source, self.t
            s.update(source=source, live=live, cameras=list(self.names), controls=list(self.controls))
        if not live:
            s["stage"] = "IDLE"
            s["goal"] = (
                f"{source} ended {time.time() - t:.0f} s ago" if source and t
                else (f"{source} ended" if source else "run bebop, calibrate, arm-check or teach")
            )
        return s

    def get_attempts(self) -> list:
        with self._lock:
            return copy.deepcopy(self.attempts)

    def frame(self, cam: str) -> tuple[int, bytes] | None:
        return self.frames.get(cam)


def make_app(hub: Hub, runs_dir: str | Path = RUNS_DIR):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Byte to Bite")
    runs = Path(runs_dir).resolve()

    async def stream(cam: str):
        last = None
        while True:
            got = hub.frame(cam)
            if got is not None and got[0] != last:
                last = got[0]
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + got[1] + b"\r\n"
            await asyncio.sleep(1.0 / MAX_FPS)

    @app.get("/")
    def index():
        return FileResponse(INDEX, media_type="text/html")

    @app.get("/stream/{cam}")
    def stream_cam(cam: str):
        mime = "multipart/x-mixed-replace; boundary=frame"
        return StreamingResponse(stream(cam), media_type=mime)

    @app.get("/state")
    def get_state():
        return JSONResponse(hub.snapshot())

    @app.get("/attempts")
    def get_attempts():
        return JSONResponse(hub.get_attempts())

    @app.get("/frames/{path:path}")
    def frame(path: str):
        p = (runs / path).resolve()
        if not p.is_relative_to(runs) or not p.is_file():
            raise HTTPException(404, "no such frame")
        return FileResponse(p)

    @app.post("/push")
    async def push(request: Request):  # Request at module level: annotations are strings
        return hub.push(await request.json())

    def control(name: str):
        why = hub.press(name)
        if why:
            raise HTTPException(503, why)
        return {"ok": True, "action": name}

    @app.post("/stop")
    def stop():
        return control("stop")

    @app.post("/resume")
    def resume():
        return control("resume")

    @app.post("/release")
    def release():
        return control("release")

    return app


def running(port: int) -> bool:
    """True when a dashboard answers on localhost:port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=0.5) as r:
            return r.status == 200
    except OSError:
        return False


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(prog="dashboard", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    os.chdir(ROOT)  # data/runs/ (attempt frames) is relative to the repo root
    if running(args.port):
        print(f"dashboard already running: http://localhost:{args.port}")
        return
    print(f"dashboard: http://localhost:{args.port}  (Ctrl+C stops it)", flush=True)
    uvicorn.run(
        make_app(Hub()), host=args.host, port=args.port, log_level="warning",
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
