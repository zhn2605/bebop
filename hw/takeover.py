"""One arm program at a time: each writes its pid to data/arm.pid when it starts and stops the pid
already there first. SIGTERM folds that arm to rest and frees the arm and cameras; SIGKILL if it
hangs (torque stays on, so the arm holds)."""

from __future__ import annotations

import atexit
import os
import signal
import time
from pathlib import Path

PIDFILE = Path(__file__).resolve().parents[1] / "data" / "arm.pid"


def _alive_python(pid: int) -> bool:
    """Still running and still a python process (not a reused pid)."""
    try:
        return b"python" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def take_over(wait_s: float = 20.0) -> None:
    try:
        old = int(PIDFILE.read_text())
    except (OSError, ValueError):
        old = 0
    if old and old != os.getpid() and _alive_python(old):
        print(f"stopping the previous run (pid {old}); it folds the arm to rest first")
        os.kill(old, signal.SIGTERM)
        deadline = time.time() + wait_s
        while _alive_python(old) and time.time() < deadline:
            time.sleep(0.2)
        if _alive_python(old):
            print(f"pid {old} hung; killing it (torque stays on, the arm holds)")
            os.kill(old, signal.SIGKILL)
            time.sleep(0.5)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    atexit.register(PIDFILE.unlink, missing_ok=True)
