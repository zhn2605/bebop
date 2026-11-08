"""Run state shared with the dashboard, and the per-run event log under data/runs/<ts>/."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from core.types import Flag

RUNS_DIR = Path("data/runs")


def jsonable(x: Any) -> Any:
    """x in plain JSON types: dataclasses become dicts, tuples lists, numpy values Python numbers,
    NaN/inf None, bytes are dropped (images go to files, not JSON)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if not isinstance(v, (bytes, bytearray))}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x if not isinstance(v, (bytes, bytearray))]
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if hasattr(x, "tolist"):  # numpy arrays and scalars
        return jsonable(x.tolist())
    return str(x)


class State:
    """Thread-safe picture of the run. The orchestrator writes it, the dashboard reads snapshots.

    attempts: one dict per attempt (plan, results, verdict, lesson, frame paths), in order.
    annotations: per camera, the overlay list perception.overlay.draw takes; replaced, not mutated.
    """

    def __init__(self, flags: dict[str, Flag] | None = None):
        self._lock = threading.Lock()
        self._flags = flags or {}
        self._fields: dict[str, Any] = {
            "stage": "INIT", "goal": "", "attempt": 0, "max_attempts": 0, "action": "",
            "action_args": {}, "mode": "", "task": "", "arm": None, "face": None, "dist_cm": None,
            "led": "", "target_xyz": None, "err_px": None, "err_mm": None, "vlm_latency_s": None,
            "vlm_cost_usd": None, "cost_usd": 0.0, "budget_usd": None, "help": "", "voice": "",
            "fake": False, "t0": time.time(),
        }
        self.attempts: list[dict] = []
        self.annotations: dict[str, list] = {}

    def set(self, **fields: Any) -> None:
        fields = jsonable(fields)
        with self._lock:
            self._fields.update(fields)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._fields.get(key, default)

    def snapshot(self) -> dict:
        with self._lock:
            s = copy.deepcopy(self._fields)
            s["n_attempts"] = len(self.attempts)
        s["flags"] = {n: {"set": f.is_set(), "reason": f.reason} for n, f in self._flags.items()}
        s["elapsed_s"] = round(time.time() - s["t0"], 1)
        return s

    def add_attempt(self, rec: dict) -> int:
        """Appends an attempt record; returns its index for update_attempt."""
        rec = jsonable(rec)
        with self._lock:
            self.attempts.append(rec)
            return len(self.attempts) - 1

    def update_attempt(self, i: int, **fields: Any) -> None:
        fields = jsonable(fields)
        with self._lock:
            self.attempts[i].update(fields)

    def get_attempts(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self.attempts)

    def annotate(self, cam: str, anns: list[dict]) -> None:
        """Replaces the overlay of one camera stream."""
        with self._lock:
            self.annotations = {**self.annotations, cam: list(anns)}


class EventLog:
    """events.jsonl (one JSON object per line, with t and type) plus frames/ in run_dir."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._file = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, type: str, **data: Any) -> None:
        line = json.dumps({"t": round(time.time(), 3), "type": type, **jsonable(data)})
        with self._lock:
            if not self._file.closed:
                self._file.write(line + "\n")
                self._file.flush()

    def save_frame(self, cam: str, jpeg: bytes, tag: str) -> str:
        """Writes frames/<tag>-<cam>.jpg; returns its path relative to data/runs (the dashboard
        serves it at /frames/<path>)."""
        path = self.dir / "frames" / f"{tag}-{cam}.jpg"
        path.write_bytes(jpeg)
        try:
            return path.resolve().relative_to(RUNS_DIR.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def close(self) -> None:
        with self._lock:
            self._file.close()
