"""The VLM planner: intent, plan and verify, each one fresh provider session closed after the call.

Stateless: everything the model needs is in the blocks, so Gemini and Claude behave the same.
"""

from __future__ import annotations

import time

from brain import grid, prompts
from brain.providers import backend, pick
from brain.providers.base import BackendError, Image, Text, Tool
from core.types import Action, Intent, Plan, Prediction, Verdict

PLAN_CAMERAS = ("top", "side")
# The acquire vocabulary and arg ranges when brain.yaml has none. width_mm None = the gripper max.
RANGES = {
    "move_to": {"dz": [0, 150], "pitch": [-90, 0]},
    "descend_until_contact": {"max_dz": [10, 120]},
    "open": {"width_mm": [5, None]},
    "close": {},
    "scoop": {"depth_mm": [0, 40], "drag_mm": [0, 60], "tilt_deg": [0, 45]},
    "lift": {"dz": [10, 150]},
    "retract": {},
    "home": {},
}
_GRIPPER_MAX_MM = 60.0  # when ctx capabilities have no gripper_max_mm
_HOVER_DZ = 40.0

_STR = {"type": "string"}
_ARGS = {
    "at": {"type": "string", "enum": ["target"], "description": 'move_to: "target"'},
    "dz": {"type": "number", "description": "move_to: mm above at; lift: mm up"},
    "pitch": {"type": "number", "description": "move_to: tool angle, deg; -90 points down"},
    "max_dz": {"type": "number", "description": "descend_until_contact: most mm down"},
    "width_mm": {"type": "number", "description": "open: jaw opening, mm"},
    "depth_mm": {"type": "number", "description": "scoop: mm below the bowl rim"},
    "drag_mm": {"type": "number", "description": "scoop: mm dragged toward the base"},
    "tilt_deg": {"type": "number", "description": "scoop: deg tilted up at the end"},
}

INTENT_TOOL = Tool("intent", "The person's intent.", {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["request", "yes", "no", "stop", "none"]},
        "object": {"type": "string", "description": 'what they asked for, "" unless a request'},
        "mode": {"type": "string", "enum": ["solid", "spoon", "none"]},
    },
    "required": ["kind", "object", "mode"],
    "additionalProperties": False,
})

VERIFY_TOOL = Tool("verdict", "The answer to the question.", {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean", "description": "true = yes"},
        "confidence": {"type": "number", "description": "0 to 1"},
        "note": {"type": "string", "description": "one sentence on what decides it"},
    },
    "required": ["ok", "confidence", "note"],
    "additionalProperties": False,
})


def plan_tool(cams: list[str], actions: list[str]) -> Tool:
    """The plan tool: reasoning first, so the model diagnoses before it chooses."""
    box = {"type": "array", "items": {"type": "number"},
           "description": "[ymin, xmin, ymax, xmax] on the 0-1000 grid"}
    action = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": actions},
            "args": {"type": "object", "properties": _ARGS, "additionalProperties": False},
        },
        "required": ["name", "args"],
        "additionalProperties": False,
    }
    prediction = {
        "type": "object",
        "properties": {"success_prob": {"type": "number"}, "expected": _STR, "risk": _STR},
        "required": ["success_prob", "expected", "risk"],
        "additionalProperties": False,
    }
    params = {
        "type": "object",
        "properties": {
            "reasoning": _STR,
            "target": {"type": "object", "properties": {cam: box for cam in cams},
                       "additionalProperties": False,
                       "description": "per camera that sees the piece"},
            "actions": {"type": "array", "items": action},
            "prediction": prediction,
            "lesson": _STR,
            "escalate": {"type": "boolean"},
            "escalate_reason": _STR,
        },
        "required": ["reasoning", "target", "actions", "prediction", "lesson", "escalate",
                     "escalate_reason"],
        "additionalProperties": False,
    }
    return Tool("plan", "One attempt's plan.", params)


def parse_plan(args: dict, ranges: dict, sizes: dict[str, tuple[int, int]]) -> Plan:
    """A Plan with pixel boxes from the model's args; ValueError naming every problem."""
    problems = []
    escalate = bool(args.get("escalate"))
    target = {}
    for cam, box in (args.get("target") or {}).items():
        if box is None or cam not in sizes:
            continue
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in box)):
            problems.append(f"target.{cam} must be 4 numbers in 0-1000, got {box}")
        elif not (box[0] < box[2] and box[1] < box[3]):
            problems.append(f"target.{cam} must be [ymin, xmin, ymax, xmax] with min < max")
        else:
            target[cam] = grid.to_pixels(box, *sizes[cam])
    actions = []
    for n, raw in enumerate([] if escalate else args.get("actions") or [], 1):
        name = raw.get("name") if isinstance(raw, dict) else None
        if name not in ranges:
            problems.append(f"action {n} {name!r} is not one of {', '.join(ranges)}")
            continue
        given = {k: v for k, v in (raw.get("args") or {}).items() if v is not None}
        kept = {}
        if name == "move_to":
            if given.get("at") != "target":
                problems.append(f'action {n} move_to needs at: "target"')
            kept["at"] = "target"
        for arg, (lo, hi) in ranges[name].items():
            value = given.get(arg)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"action {n} {name} needs {arg} (a number)")
            elif not lo <= value <= hi:
                problems.append(f"action {n} {name}.{arg} {value:g} is outside {lo:g} to {hi:g}")
            else:
                kept[arg] = float(value)
        actions.append(Action(name, kept))
    if not escalate:
        if not actions:
            problems.append("no actions (set escalate if nothing should be done)")
        if not target and any(a.name in ("move_to", "scoop") for a in actions):
            problems.append("no target box for any camera")
    if problems:
        raise ValueError("; ".join(problems))
    p = args.get("prediction") or {}
    prob = p.get("success_prob")
    prob = min(max(float(prob), 0.0), 1.0) if isinstance(prob, (int, float)) else 0.0
    return Plan(
        target=target,
        actions=actions,
        prediction=Prediction(prob, str(p.get("expected", "")), str(p.get("risk", ""))),
        reasoning=str(args.get("reasoning", "")),
        lesson=str(args.get("lesson", "")),
        escalate=escalate,
        escalate_reason=str(args.get("escalate_reason", "")),
    )


def _escalation(reason: str) -> Plan:
    return Plan({}, [], Prediction(0.0, "", reason), reasoning="", escalate=True,
                escalate_reason=reason)


class Planner:
    """cfg is brain.yaml as a dict; log is an EventLog (or None)."""

    def __init__(self, cfg: dict, log=None):
        self.cfg = cfg
        self.log = log
        self.provider = pick(cfg.get("provider") or "auto")
        models = (cfg.get("models") or {}).get(self.provider) or {}
        self.model = cfg.get("model") or models.get("plan")  # None = the backend's default
        self.intent_model = models.get("intent") or self.model

    def intent(self, text: str) -> Intent:
        """What an utterance asks for. Text only, cheap model; Intent("none") when the VLM fails."""
        def parse(args):
            if args.get("kind") not in ("request", "yes", "no", "stop", "none"):
                raise ValueError(f"kind {args.get('kind')!r} is not one of the five")
            mode = args.get("mode") if args.get("mode") in ("solid", "spoon") else None
            return Intent(args["kind"], args.get("object") or None, mode, text)

        try:
            intent, problem, *_ = self._ask("intent", self.intent_model, prompts.INTENT,
                                            INTENT_TOOL, [Text(f"They said: {text!r}")], parse)
        except BackendError as e:
            self._emit("vlm_error", call="intent", error=str(e))
            return Intent("none", text=text)
        return intent or Intent("none", text=text)

    def plan(self, ctx: dict) -> Plan:
        """One attempt's Plan with pixel boxes. ctx: task, mode, goal, capabilities
        (gripper_max_mm, reach_mm, tool, hover_dz), attempts (reasoning, results as ActionResult
        dicts, verdict), lessons, images {cam: jpeg}. A VLM failure or a plan still invalid after
        one retry comes back as an escalation with the reason."""
        images = {cam: ctx["images"][cam] for cam in PLAN_CAMERAS
                  if (ctx.get("images") or {}).get(cam)}
        if not images:
            return _escalation("no top or side camera image")
        caps = ctx.get("capabilities") or {}
        ranges = {
            name: {arg: [lo, caps.get("gripper_max_mm", _GRIPPER_MAX_MM) if hi is None else hi]
                   for arg, (lo, hi) in args.items()}
            for name, args in (self.cfg.get("ranges") or RANGES).items()
        }
        sizes = {cam: grid.size(jpeg) for cam, jpeg in images.items()}
        blocks = [Text(prompts.plan_user(ctx, sizes))]
        for cam, jpeg in images.items():
            shown = grid.with_grid(jpeg) if self.cfg.get("grid", True) else jpeg
            blocks += [Text(f"Camera {cam}:"), Image(shown, label=cam)]
        system = prompts.plan_system(ranges, caps.get("hover_dz", _HOVER_DZ))
        try:
            plan, problem, cost, latency = self._ask(
                "plan", self.model, system, plan_tool(list(images), list(ranges)), blocks,
                lambda args: parse_plan(args, ranges, sizes))
        except BackendError as e:
            self._emit("vlm_error", call="plan", error=str(e))
            return _escalation(f"the VLM failed: {e}")
        if plan is None:
            plan = _escalation(f"the plan was invalid twice: {problem}")
        plan.cost_usd, plan.latency_s = cost, latency
        return plan

    def verify(self, question: str, images: dict[str, bytes]) -> Verdict:
        """A yes/no check of the images ({cam: jpeg}); not ok when the VLM fails."""
        blocks = [Text(f"Question: {question}")]
        for cam, jpeg in images.items():
            if jpeg:
                blocks += [Text(f"Camera {cam}:"), Image(jpeg, label=cam)]

        def parse(args):
            conf = args.get("confidence")
            conf = min(max(float(conf), 0.0), 1.0) if isinstance(conf, (int, float)) else 0.0
            return Verdict(bool(args.get("ok")), conf, str(args.get("note", "")))

        try:
            verdict, _, cost, _ = self._ask("verify", self.model, prompts.VERIFY, VERIFY_TOOL,
                                            blocks, parse)
        except BackendError as e:
            self._emit("vlm_error", call="verify", error=str(e))
            return Verdict(False, 0.0, f"the VLM failed: {e}")
        verdict.cost_usd = cost
        return verdict

    def _ask(self, call, model, system, tool, blocks, parse):
        """(result, problem, cost_usd, latency_s). Sends blocks; when parse raises ValueError,
        re-sends once with the problem; result is None and problem set when that fails too."""
        options = {k: self.cfg[k] for k in ("timeout_s", "max_budget_usd", "effort")
                   if self.cfg.get(k) is not None}
        if model:
            options["model"] = model
        t0, cost, result, problem = time.monotonic(), None, None, None
        session = backend(self.provider, **options).start(system, [tool])
        try:
            for _ in range(2):
                turn = session.send(blocks)
                if turn.cost_usd is not None:
                    cost = (cost or 0.0) + turn.cost_usd
                try:
                    result, problem = parse(turn.args), None
                    break
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    problem = str(e)
                    self._emit("vlm_invalid", call=call, problem=problem, args=turn.args)
                    blocks = [Text(prompts.retry(call, problem))]
        finally:
            session.close()
        latency = time.monotonic() - t0
        self._emit("vlm", call=call, provider=self.provider, model=model, note=turn.note,
                   args=turn.args, cost_usd=cost, latency_s=round(latency, 2), problem=problem)
        return result, problem, cost, latency

    def _emit(self, kind: str, **data) -> None:
        if self.log is not None:
            self.log.emit(kind, **data)
