# Byte to Bite — CORE v2 (2026-11-07, supersedes the grid design in CORE.md)

Self-correcting feeding arm on an SO-101. A VLM finds the food in two fixed cameras, geometry turns
pixels into a 3D target, analytic IK moves the arm, every action reports predicted vs measured, and
the VLM re-plans from those metrics. Two gated stages: acquire, then deliver. Two modes: `solid`
(bare gripper pinches) and `spoon` (a spoon strapped into the closed gripper scoops). The person
being fed never touches anything: voice and face are their only controls.

CORE.md stays for hardware (SO-101, STS3215, Waveshare bus adapter, ESP32 pinout), the ESP32 serial
protocol (telemetry every 50 ms, `{"event":"ESTOP"}`, `STATE <NAME>` commands, 500 ms watchdog), the
LED meanings and the safety requirements S1–S10. Everything about cells, corners, taught scoop
deltas and `brain/search.py` is gone.

## Rules

- Python 3.10+. Arch Linux on the arm laptop. Run everything as `uv run python -m <pkg>.<tool>` (or the short `uv run` names in pyproject.toml) from the repo
  root. Config is YAML under `config/`. Hackathon: no tests, minimal abstractions, small files.
- `core/types.py` is the only place dataclasses live. Import, never redefine.
- Nothing from the VLM is ever a joint angle or a world coordinate. It returns pixels and choices.
- Every arm command passes through the fence (`motion/fence.py`) and the speed cap.
- The arm is on a teammate's laptop. Everything must run here with `--fake` (fake arm, fake cameras)
  against the real VLM.

## Layout

```
config/   robot.yaml cameras.yaml workspace.yaml brain.yaml poses.yaml(taught) calib.json(fit) fake.yaml
core/     types.py state.py run.py notify.py
hw/       arm.py esp32.py
motion/   kinematics.py fence.py primitives.py
perception/ cameras.py marker.py calib.py face.py overlay.py
brain/    providers/{base,claude_code,gemini,__init__}.py planner.py prompts.py grid.py
voice/    listener.py
dashboard/ server.py index.html
tools/    arm_check.py teach.py calibrate.py list_cameras.py face_check.py voice_check.py
data/     runs/<ts>/ (events.jsonl + frames)   gitignored
```

## Config

- `robot.yaml`: `port`, `id: feeder_arm`, `calibration_dir: calibration`, per-joint soft limits (deg),
  `speed_deg_s` (cap, default 60), `slow_factor` 0.4, `hz` 50, torque caps per servo (register value),
  `contact_load` (fraction, default 0.25), `track_err_deg` (default 6), `tool_offsets` {jaw, spoon}
  as xyz in the gripper frame (mm), `gripper` {open_mm_at_100, closed_mm_at_0, max_mm}, `geometry`
  (link parameters for kinematics, with the source URDF named).
- `cameras.yaml`: `cameras: {top: {source, width, height}, side: {...}, face: {...}}`, `fps` 15,
  `marker: {hsv_low, hsv_high, min_area}`.
- `workspace.yaml`: `fence` box (mm, where the tip may go in acquire), `head_zone` box (forbidden
  except during `stage`/`present`), `hover_dz` 40, `table_z` 0, `bowl_rim_z` default, `retract_dz` 60,
  `present` {stop_short_mm 50, ultrasonic_stop_cm 6, speed slow}.
- `poses.yaml`: taught joint poses, deg: `home`, `stage` (15 cm in front of the mouth), `mouth`
  (spoon at the lips). Written by `tools.teach`.
- `brain.yaml`: `provider: auto|claude-code|gemini`, `model`, `timeout_s` 60, `max_budget_usd`, grid
  overlay on/off, `max_attempts` 3.
- `fake.yaml`: `food_xyz`, `food_radius_mm`, `bowl_rim_z`, fake camera placement.

## Module contracts

### hw/arm.py — `Arm(cfg, stop: Flag, fake=False)`
- `connect()` sets `use_degrees=True`, `max_relative_target` float, points `calibration_dir` at the
  repo's `calibration/`, writes the torque caps, reads once. `disconnect()` torque off.
- `state() -> ArmState` (FK via a `Kinematics` given at construction; tool switchable with
  `arm.tool = "jaw"|"spoon"`).
- `move_joints(q: dict, duration_s, pause: Flag|None=None) -> TrackResult`: min-jerk interpolation at
  `hz`, every tick checks `stop` (→ `freeze()` then raise `Stopped`) and `pause` (hold position until
  cleared; after `pause_timeout_s` raise `Stopped`). Duration is raised to respect `speed_deg_s`.
- `set_gripper_mm(mm, duration_s) -> float measured`, `close_gripper(load_limit) -> float measured`
  (closes until the gripper load exceeds the limit or it is shut; returns the opening in mm).
- `freeze()`: goal = present position on all servos. `torque(on)`.
- Fake mode: instant moves, loads 0, gripper tracks the command, `close_gripper` returns
  `fake.food_width_mm` when the tip is within `food_radius_mm` of `food_xyz`, else 0.

### hw/esp32.py — `Esp32(port, stop: Flag)` or `None` when absent
- Thread reading JSON lines; `.dist_cm`, `.estop`; `{"event":"ESTOP"}` or `estop: true` trips `stop`.
- `send_state(name)` writes `STATE <NAME>` at least every 300 ms (watchdog). States: READY, LISTENING,
  THINKING, MOVING, PRESENTING, STOPPED, HELP.

### motion/kinematics.py — `Kinematics(geometry, tool_offsets)`
- `fk(q: dict, tool) -> Pose`, `ik(pose: Pose, tool, q_seed=None) -> dict` or raise `Unreachable`.
- Analytic: shoulder_pan = yaw from (x, y); the other three pitch joints form a planar chain; the
  wrist position is tip minus the tool vector at the requested pitch; two-link law of cosines for
  shoulder/elbow; wrist_flex closes the pitch; wrist_roll = roll. Geometry and the zero convention
  come from the research file (`so101_new_calib.urdf`); if the URDF zero does not match LeRobot's
  degrees mode, a per-joint `zero_offset_deg` in `robot.yaml` fixes it, and `tools.arm_check`
  prints the FK of the current pose so the teammate can check with a ruler.

### motion/fence.py — `Fence(cfg)`
- `check(xyz, stage: str) -> str|None` (None = allowed, else the reason). `acquire` allows the fence
  box only; `deliver` also allows the segment stage→mouth with a 40 mm radius.

### motion/primitives.py — `Motion(arm, kin, fence, cfg, poses)`
Each method returns an `ActionResult` with the keys in Metrics, and raises `Stopped` when the arm
was frozen (the result is still logged by the caller).
- `move_to(pose, speed="normal"|"slow")`: fence-check the target and 5 points along the straight
  tip path, IK, joint interpolation.
- `descend_until_contact(max_dz, load=None)`: step down 5 mm at a time at slow speed until a joint
  load exceeds `contact_load` or `max_dz` is reached; measured `contact_z`.
- `open(width_mm)`, `close()`, `lift(dz)`, `retract()` (up by `retract_dz`, then `home`), `home()`,
  `stage()`, `present(gate: Callable[[], str|None])`: from `stage` toward `mouth` at slow speed,
  stopping `stop_short_mm` before `mouth` or when `gate()` returns a reason.
- `scoop(depth_mm, drag_mm, tilt_deg)`: from a hover over the target: pitch down, dip `depth_mm`
  below `bowl_rim_z`, drag `drag_mm` toward the base, tilt up by `tilt_deg`, lift to hover.
- `tip -> Pose` property.

### perception/cameras.py — `Cameras(cfg, fake: FakeWorld|None=None)`
- One grabber thread per camera holding the latest frame; `start()`, `stop()`, `names`,
  `latest(name) -> (bgr, t)|None`, `jpeg(name, annotations=None, quality=70) -> bytes|None`.
- `FakeWorld(tip_fn, calib, food_xyz, food_radius_mm)`: renders 640×480 frames: grey table, a brown
  disc at the projected food, a green disc at the projected tip, a white "fake" label; face camera
  renders a drawn open mouth so `FaceSignals` has something to detect, or returns a constant
  `FaceState` when fake.

### perception/marker.py — `find_marker(bgr, marker_cfg) -> UV|None`
HSV threshold, largest blob above `min_area`, centroid.

### perception/calib.py — `Calibration`
- `fit(samples, size) -> Calibration`: per camera a pinhole with square pixels, centred principal
  point and one radial term (cv2.calibrateCamera, ≥ 6 non-coplanar samples), `P: dict[str, 3x4]`,
  `K`, `dist`, `rms_px`. `implausible(table_z)` lists why a fit can't be a real camera.
- `along_ray(name, uv, other, other_uv, z_range)`: the height on `name`'s ray that `other` sees
  closest to other_uv (top fixes x and y, side only the height).
- `project(name, xyz) -> UV`, `triangulate({name: UV}) -> Vec3` (two or more cameras, least squares),
  `at_height(name, uv, z) -> Vec3` (one camera, known height), `width_mm(name, box, z) -> float`.
- `save(path)`, `load(path)`. JSON.

### perception/face.py — `FaceSignals(cameras, name="face", cfg)` thread
- MediaPipe FaceLandmarker. `.state -> FaceState`. `mouth_open` = ratio > `mouth_open_ratio` (0.35)
  for 300 ms; `looking_away` = |yaw| > 30° or face absent for 500 ms. `start()`, `stop()`.

### perception/overlay.py — `draw(bgr, annotations) -> bgr`
Annotation dicts: `{"type": "point"|"box"|"text"|"cross", "uv"|"box"|"text", "color", "label"}`.
Convention: predicted tip = hollow circle, detected marker = filled circle, target = cross,
VLM box = rectangle with its label.

### brain/ — `Planner(cfg, log)`
- Uses `brain.providers.backend(name, model=...)`; each call is `start(system, [tool]).send(blocks)`
  and the `Turn.args` are the result; the session is closed after. Stateless: all context is text in
  the blocks, so Gemini and Claude behave the same.
- `intent(text) -> Intent`. Text-only, cheap model.
- `plan(ctx) -> Plan`. `ctx`: task text, mode, goal, capabilities (gripper max mm, reach mm, tool),
  the previous attempts of this goal (each: plan reasoning, per-action predicted/measured/error,
  verdict), lessons, and the current `top`/`side` frames as JPEG with the grid overlay
  (`brain/grid.py`: 0–1000 normalized axes drawn every 100, labelled) when `grid: true`. Boxes come
  back as `[ymin, xmin, ymax, xmax]` in 0–1000 (Gemini's convention; the prompt tells Claude to use
  the same) and are converted to pixel `Box` before returning. Actions are validated against
  `ActionName` and the arg schema; an invalid plan is retried once with the validator's message.
- `verify(question, images) -> Verdict`.
- `providers/gemini.py`: `google-genai`, temperature 0, JSON response schema built from the tool's
  params, `GEMINI_API_KEY`/`GOOGLE_API_KEY`; `auto` picks Gemini when a key is set, else claude-code.
- Prompts in `prompts.py`. The plan prompt lists the action vocabulary with arg ranges and says:
  pixels only; `target` may be referenced in `move_to.at`; prefer `descend_until_contact` over
  guessing heights; report `escalate` when something is unsafe or impossible (object too wide, bowl
  moved, person in the workspace).

### voice/listener.py — `Listener(cfg, on_utterance, on_stop)` thread
Vosk small model + sounddevice at 16 kHz. Partial results scanned for `stop_words` ("stop", "wait",
"no no") → `on_stop(word)` immediately; final results → `on_utterance(text)`. `last_text`, `level`.

### core/state.py — `State`, `EventLog`
- `State`: thread-safe; `set(**fields)`, `snapshot() -> dict` (stage, goal, attempt, action,
  arm: ArmState as dict, face, dist_cm, led, flags, last VLM latency/cost, totals, mode, task),
  `attempts: list[dict]` (each attempt's plan, results, verdict, lesson, frame paths),
  `annotations: dict[cam, list]`.
- `EventLog(run_dir)`: `emit(type, **data)` appends JSONL with `t`; `save_frame(cam, jpeg, tag) ->
  path`.

### core/notify.py — `Notifier(topic)`
`help(reason, url)`, `info(msg)` via ntfy.sh POST with urllib; no-op when `topic` is empty.

### core/run.py — the orchestrator and CLI
```
python -m core.run [--task TEXT | --voice] --mode solid|spoon [--fake] [--provider] [--model]
                   [--no-dashboard] [--port] [--esp32 PORT] [--ntfy TOPIC]
```
State machine (ESP32 state in brackets):
1. READY: home, torque on, listening [READY/LISTENING]. Input: `--task` once, or `Listener`
   utterances → `intent()` → a `request` starts a Task.
2. Task → Goal `acquire`: up to `max_attempts` Attempts. Each: frames → `plan()` [THINKING] →
   `target_xyz = triangulate` (or `at_height` with `bowl_rim_z` when only `top` sees it) →
   `width_mm` → refuse before moving if `width > gripper.max_mm` (escalate) → run actions in order
   [MOVING], each `ActionResult` logged and shown → `verify("Is <object> held in the <tool>?")` +
   the `close` metric decide `ok`. On failure the next attempt gets the metrics and a lesson. After
   `max_attempts` → HELP: `Notifier.help`, red LED, wait for a voice "stop"/"continue" or the
   dashboard button.
3. Goal `deliver` (fixed sequence, VLM only verifies): `stage()` → wait for `FaceState.mouth_open`
   or a voice "yes" (timeout 20 s → retract, home, HELP) → `present(gate)` where gate returns a
   reason when `dist_cm <= ultrasonic_stop_cm`, `looking_away`, or `stop` [PRESENTING] → solid:
   `open()` once the mouth has been open 1 s at the present point; spoon: hold until `mouth_open`
   goes false (bite) or 8 s → `retract()` → `verify("Is the <tool> empty?")`.
4. Flags: `stop` (voice stop word, ESP32 e-stop, space in the terminal, dashboard button, face lost
   during `present`) → arm frozen, state STOPPED; after 1 s it goes home slowly and holds there
   (torque on; a stop on the way freezes it where it is); voice "continue" or the dashboard button
   goes on with the task. `pause` (looking away during acquire) → hold, then
   resume when the face is back.
5. Every stage change, plan, action, verdict and flag is an event. Frames at plan time and after
   each attempt are saved and linked in `attempts`.

### dashboard/ — `make_app(state, cameras) -> FastAPI`, served by uvicorn in a thread
Plain HTML, one file, no framework. `/` page; `/stream/<cam>` MJPEG with overlays from
`state.annotations`; `/state` JSON (polled every 500 ms); `/attempts` JSON; `/frames/<path>`;
`/stop` and `/resume` POST. Page: three streams, the numbers table, the attempts strip with
prev/next (arrow keys) and a Live toggle, each attempt's plan/prediction/per-action table/lesson/
frames, and one small Chart.js chart (position error per attempt; predicted success prob vs
outcome). `--fake` runs it the same way.

## Metrics (ActionResult keys)
- `move_to`: predicted `tip_xyz`, `tip_uv:{cam}`; measured `tip_xyz` (FK), `marker_uv:{cam}`,
  `track_err_deg`, `max_load`; error `xyz_mm` (|FK−target|), `px:{cam}` (|predicted−marker|),
  `mm_from_marker` (triangulated marker vs target, when both cameras see it).
- `descend_until_contact`: predicted `contact_z`; measured `contact_z`, `max_load`; error `z_mm`.
- `open`: predicted `width_mm`; measured `width_mm`; error `mm`.
- `close`: predicted `width_mm` (object width); measured `width_mm`, `load`; error `mm`;
  `ok` = measured ≥ 3 mm and |error| ≤ 12 mm.
- `scoop`: predicted `depth_mm`; measured `peak_wrist_load`, `contact`; `ok` from `verify`.
- `present`: predicted `stop_short_mm`; measured `dist_cm`, `stopped_by`.
- Any action: `track_err_deg > track_err_deg_cfg` with `max_load > contact_load` → `ok=False`,
  note "contact", and the orchestrator freezes and retracts.

## Safety (S-numbers carry over from CORE.md)
Torque caps written at connect; speed cap and slow factor on `present`, `descend_until_contact`,
`close`; fence and head zone; contact detection from tracking error + load; `stop` = freeze, then
home slowly and hold (torque on; a stop on the way freezes it there), only the dashboard "release" =
torque off; ultrasonic gates `present` only; the person's controls are voice and face only, the
caregiver has the ESP32 button and the laptop keyboard;
HELP notifications go to a phone via ntfy.

## Tools
- `tools.list_cameras`: saves one frame per /dev/video* index to `data/cameras/`.
- `tools.arm_check`: prints joints, FK tip, loads at 2 Hz; REPL: `xyz X Y Z [PITCH]` moves,
  `q NAME DEG` moves one joint, `open MM`, `close`, `home`, `torque off`. Space = stop.
- `tools.teach`: torque off, move by hand, Enter saves `home`, `stage`, `mouth` into `poses.yaml`;
  `--goto NAME`.
- `tools.calibrate`: carries a ball held in the jaws through a 3x3x3 grid at pitch -90 (from
  `workspace.yaml` calib_box), finds it in `top` and `side`, fits, prints the held-out px and mm,
  saves `config/calib.json` unless the fit is implausible;
  `--check` projects the current tip into each stream window for a visual check.
- `tools.face_check`: prints `FaceState` at 5 Hz. `tools.voice_check`: prints partials/finals and
  stop-word hits.

## Fake end-to-end
`uv run bebop --fake --mode solid --task "feed me a donut hole"` runs the whole loop here:
fake arm, fake cameras rendering the fake world through a built-in `Calibration`, the real VLM,
dashboard on :8000. This is the integration check after every change.
