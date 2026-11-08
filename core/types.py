"""Shared dataclasses and flags. Every package imports these; nobody redefines them.

Units: millimetres, degrees, seconds. Frames: the base frame has its origin on the table under the
shoulder_pan axis, x forward (away from the base, toward the bowl), y left, z up.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

Vec3 = tuple[float, float, float]
UV = tuple[float, float]  # pixel column u, row v
Box = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in pixels

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
CAMERAS = ("top", "side", "face")
Mode = Literal["spoon", "solid"]
Tool = Literal["jaw", "spoon"]


class Stopped(RuntimeError):
    """A motion was interrupted by a Flag. The arm is frozen (holding position) when this is raised."""


class Unreachable(ValueError):
    """IK found no joint solution inside the limits for the requested pose."""


class Flag:
    """Thread-safe latch with a reason. `stop` freezes the arm; `pause` holds it until cleared."""

    def __init__(self, name: str):
        self.name = name
        self._event = threading.Event()
        self.reason = ""
        self.t = 0.0

    def trip(self, reason: str) -> None:
        if not self._event.is_set():
            self.reason, self.t = reason, time.time()
        self._event.set()

    def clear(self) -> None:
        self._event.clear()
        self.reason = ""

    def is_set(self) -> bool:
        return self._event.is_set()


@dataclass
class Pose:
    """Tool tip pose. pitch: angle of the tool's forward axis above horizontal, so -90 points straight
    down at the table. roll: wrist_roll relative to "spoon bowl up"; 0 for grasping."""

    xyz: Vec3
    pitch: float
    roll: float = 0.0


@dataclass
class ArmState:
    t: float
    q: dict[str, float]  # deg per joint in JOINTS, LeRobot use_degrees convention
    goal: dict[str, float]  # last commanded deg per joint
    load: dict[str, float]  # signed fraction of max load, -1..1, per joint and gripper
    gripper_mm: float  # opening between the jaw tips
    tip: Pose  # FK of q with the active tool


@dataclass
class TrackResult:
    max_err_deg: float  # worst |goal - present| over all joints at the end of the move
    max_load: float  # worst |load| seen during the move
    duration_s: float
    stopped: bool = False


@dataclass
class ActionResult:
    """What an action predicted, what was measured, and the difference. Every action returns one.
    Keys are action-specific but stable (see CORE_V2.md, Metrics)."""

    name: str
    ok: bool
    predicted: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)
    note: str = ""
    t0: float = 0.0
    t1: float = 0.0
    stopped: bool = False


ActionName = Literal[
    "move_to",
    "descend_until_contact",
    "open",
    "close",
    "scoop",
    "lift",
    "retract",
    "home",
    "stage",
    "present",
    "verify",
    "ask_operator",
]


@dataclass
class Action:
    name: ActionName
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    success_prob: float  # 0..1
    expected: str  # one sentence, what the images should show afterwards
    risk: str  # one sentence, the most likely failure


@dataclass
class Plan:
    """One attempt, as the VLM returns it. Pixels, never world coordinates: the orchestrator turns
    `target` into xyz through the calibration and fills `target` references in the actions."""

    target: dict[str, Box]  # per camera name that sees the object; may be missing a camera
    actions: list[Action]
    prediction: Prediction
    reasoning: str  # diagnosis of the last attempt plus the idea for this one
    lesson: str = ""  # one sentence kept for later attempts, empty on the first
    escalate: bool = False  # true = stop and ping the caregiver now
    escalate_reason: str = ""
    cost_usd: float | None = None
    latency_s: float = 0.0


@dataclass
class Verdict:
    ok: bool
    confidence: float
    note: str
    cost_usd: float | None = None


@dataclass
class Intent:
    kind: Literal["request", "yes", "no", "stop", "none"]
    object: str | None = None  # "a donut hole", "some yogurt"
    mode: Mode | None = None
    text: str = ""


@dataclass
class FaceState:
    t: float
    present: bool
    mouth_open: bool
    mouth_ratio: float  # lip gap / mouth width
    yaw_deg: float
    pitch_deg: float
    looking_away: bool
    center_uv: UV | None = None
    face_width_px: float = 0.0
    mouth_uv: UV | None = None  # centre between the inner lips, px
    eyes_px: float = 0.0  # distance between the iris centres, px (gives the face's depth)
    chewing: bool = False  # the lips opened and closed at least twice in the last few seconds
