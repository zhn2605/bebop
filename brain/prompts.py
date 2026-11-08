"""System prompts and the text blocks of each planner call."""

from __future__ import annotations

import dataclasses
import json

INTENT = """\
You read what a person said to the robot arm that feeds them (speech to text, so words may be
misheard) and return their intent:
- request: they ask for food or a drink. object: what, short, with an article ("a donut hole",
  "some yogurt"). mode: solid for pieces a gripper can pinch (donut holes, grapes, crackers, fruit
  cubes, cheese); spoon for food that needs a spoon (yogurt, soup, rice, cereal, pudding, beans).
- yes: agreement, ready, "go ahead", "I'm ready", "okay".
- no: refusal or "not now", "I'm full".
- stop: stop, wait, halt, "no no", or any sign of pain, choking or distress.
- none: anything else (chatter, noise, half a sentence).
object is "" and mode is "none" unless kind is request."""

VERIFY = """\
You check a feeding robot's camera images and answer one yes/no question about them.
ok is true only when the images clearly show that the answer is yes. confidence is 0 to 1.
note: one sentence on what you see that decides it."""

_PLAN = """\
You plan one attempt of a robot arm (SO-101: five joints and a two-jaw gripper) that feeds a person
who cannot use their hands. The goal now is to acquire one piece of food from the bowl on the table.

Cameras: `top` looks down at the bowl; `side` looks across the table at it. Every image has a thin
grid with lines every 100 units on a 0-1000 scale, labelled along the top and left edges; y runs
down from the top edge, x runs right from the left edge.

Pixels only. Never give joint angles, millimetre coordinates or positions in the world: you choose
where in the images and which actions; the robot does the geometry.

Answer fields:
- target: per camera, the box around the one piece of food to pick, [ymin, xmin, ymax, xmax] on the
  0-1000 grid, tight around that piece (not the bowl, not neighbouring pieces). Box the same piece
  in every camera. Leave out a camera that does not see it. The robot triangulates the box centres
  into a 3D point; move_to with at "target" goes to that point. Pick the piece easiest to reach: on
  top, not touching others.
- actions, run in order. Vocabulary (args and allowed ranges):
{vocabulary}
  descend_until_contact moves down slowly until the arm feels contact; prefer it to guessing
  heights.
  close shuts until the jaws hold something and measures the opening. scoop runs from a hover over
  the target: tilt down, dip depth_mm below the bowl rim, drag drag_mm toward the base, tilt up by
  tilt_deg, lift back to the hover.
- Default plans (use them on a first attempt unless the scene says otherwise):
  solid: move_to(at="target", dz={hover}, pitch=-90), open(width_mm = the piece's width + about 15,
  at most the gripper's max), descend_until_contact(max_dz=60), close(), lift(dz=80).
  spoon: move_to(at="target", dz={hover}, pitch=-60), scoop(depth_mm=15, drag_mm=30, tilt_deg=30),
  lift(dz=80).
- prediction: success_prob 0-1; expected: one sentence on what the images should show afterwards;
  risk: one sentence on the most likely failure.
- reasoning: on a retry, diagnose the last attempt from its measured numbers first, then the idea
  for this one; on the first attempt, what you see and why this plan.
- lesson: one sentence worth remembering for later attempts; "" on the first attempt.
- escalate: true, with escalate_reason and no actions, when the piece is wider than the gripper's
  max opening, a person or a hand is in the workspace, the bowl or the food is missing or moved, the
  food cannot be found, the scene makes no sense, or another try cannot help. Never escalate
  because a piece looks far or hard to reach: the arm tries every tool pitch itself and says
  so in its numbers when a point is truly out of reach.

Reading the previous attempts' metrics: close measured width_mm near 0 means the jaws closed on
nothing (the target was off, or the jaws stopped above the piece); descend_until_contact contact_z
far above the predicted height means it hit the rim or another piece; move_to px is how far the
camera saw the tip from where it should have been. On a retry change something: move the box onto
the piece's centre, pick another piece, open wider, descend further, change the pitch, scoop deeper
or longer.

Lessons are kept per food across runs. Those starting with "From the caregiver:" come from a person
who watched the robot fail with this food: follow them, over your own lessons when they conflict,
and say in reasoning how this plan follows them."""

_ARG_TEXT = {
    "at": '"target"',
    "dz": "mm above the target",
    "pitch": "deg, -90 = pointing straight down, 0 = horizontal",
    "max_dz": "mm, the most it goes down",
    "width_mm": "jaw opening, mm",
    "depth_mm": "mm below the bowl rim",
    "drag_mm": "mm",
    "tilt_deg": "deg",
}


def plan_system(ranges: dict, hover_dz: float) -> str:
    """The plan prompt with the action vocabulary and its ranges ({action: {arg: [lo, hi]}})."""
    lines = []
    for name, args in ranges.items():
        parts = ['at: "target"'] if name == "move_to" else []
        parts += [f"{arg}: {lo:g} to {hi:g} ({'mm up' if name == 'lift' else _ARG_TEXT[arg]})"
                  for arg, (lo, hi) in args.items()]
        lines.append(f"  {name}({', '.join(parts)})")
    return _PLAN.format(vocabulary="\n".join(lines), hover=f"{hover_dz:g}")


def _compact(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(v) for v in value]
    return value


def _plain(value):
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _result_line(result) -> str:
    r = _plain(result)
    status = "ok" if r.get("ok") else "FAILED"
    if r.get("stopped"):
        status += " (stopped)"
    parts = [f"{r.get('name')} {status}"]
    for key in ("predicted", "measured", "error"):
        if r.get(key):
            parts.append(f"{key} {json.dumps(_compact(r[key]))}")
    if r.get("note"):
        parts.append(f"note: {r['note']}")
    return "    " + "; ".join(parts)


def plan_user(ctx: dict, sizes: dict[str, tuple[int, int]]) -> str:
    """The text block of a plan call: task, mode, goal, capabilities, attempts and lessons."""
    mode = ctx.get("mode")
    tool = {"solid": "the bare gripper pinches the food", "spoon": "a spoon held in the closed "
            "gripper scoops the food"}.get(mode, "")
    caps = ", ".join(f"{k} {v}" for k, v in (ctx.get("capabilities") or {}).items())
    lines = [
        f"Task: {ctx.get('task', '')}",
        f"Mode: {mode} ({tool})" if tool else f"Mode: {mode}",
        f"Goal: {ctx.get('goal', 'acquire')}",
        f"Capabilities: {caps}",
    ]
    attempts = ctx.get("attempts") or []
    lines.append(f"This is attempt {len(attempts) + 1} of this goal.")
    for n, attempt in enumerate(attempts, 1):
        attempt = _plain(attempt)
        lines.append(f"Attempt {n}: reasoning: {attempt.get('reasoning', '')}")
        lines += [_result_line(r) for r in attempt.get("results") or []]
        verdict = _plain(attempt.get("verdict"))
        if isinstance(verdict, dict):
            verdict = f"{'ok' if verdict.get('ok') else 'not ok'}: {verdict.get('note', '')}"
        lines.append(f"    verdict: {verdict}")
    lessons = ctx.get("lessons") or []
    if lessons:
        lines.append("Lessons so far:")
        lines += [f"- {lesson}" for lesson in lessons]
    cams = ", ".join(f"{cam} ({w}x{h} px)" for cam, (w, h) in sizes.items())
    lines.append(f"Camera images follow: {cams}.")
    return "\n".join(lines)


def retry(call: str, problem: str) -> str:
    return f"That {call} is invalid: {problem}. Send the whole {call} again, fixed."
