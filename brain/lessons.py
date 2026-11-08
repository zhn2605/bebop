"""Lessons per food, kept across runs in data/lessons.json: what the caregiver said when the robot
asked for help, and the planner's own lessons from pickups that succeeded. A task starts with its
food's lessons, so the next grape starts where the last one ended."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "lessons.json"
KEEP = 8  # per food, newest kept
CAREGIVER = "From the caregiver: "


def key(food: str) -> str:
    """"a donut hole", "Donut holes!" -> "donut hole"."""
    words = [w for w in re.findall(r"[a-z]+", food.lower()) if w not in ("a", "an", "the", "some")]
    last = words[-1] if words else ""
    if len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us")):
        words[-1] = last[:-1]
    return " ".join(words)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load(food: str, path: Path | None = None) -> list[str]:
    """The food's lessons as the planner sees them, oldest first; caregiver ones say so."""
    return [(CAREGIVER if e["source"] == "caregiver" else "") + e["text"]
            for e in _read(path or PATH).get(key(food), [])]


def add(food: str, text: str, source: str, path: Path | None = None) -> None:
    """Saves one lesson ("caregiver" or "planner"); a repeat moves to the end."""
    text = text.strip()
    if not text:
        return
    path = path or PATH
    data = _read(path)
    entries = [e for e in data.get(key(food), []) if e["text"] != text]
    entries.append({"text": text, "source": source, "t": time.strftime("%Y-%m-%d %H:%M:%S")})
    data[key(food)] = entries[-KEEP:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
