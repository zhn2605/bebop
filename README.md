# Byte to Bite

A self-correcting feeding arm on an SO-101. A VLM finds the food in two fixed cameras, geometry turns
the pixels into a 3D target, analytic IK moves the arm, every action reports predicted vs measured,
and the VLM re-plans from those numbers. Two gated stages: acquire, then deliver. Two modes: `solid`
(the bare gripper pinches) and `spoon` (a spoon strapped into the closed gripper scoops). The person
being fed controls it by voice and face only.

The spec is `.config/docs/CORE_V2.md` (hardware, the ESP32 protocol and the safety list S1-S10 are in
`.config/docs/CORE.md`).

## Setup (Arch Linux)

```sh
sudo pacman -S --needed uv portaudio v4l-utils     # portaudio: microphone; v4l-utils: v4l2-ctl
uv sync          # Python 3.10 and every dependency into .venv, from pyproject.toml and uv.lock
```

If cv2 fails to import in an old `.venv` (an earlier install had opencv-python-headless next to
opencv-contrib-python), run `uv sync --reinstall-package opencv-contrib-python` once.

- Planner backend: either the `claude` CLI on `PATH` and logged in (`--provider claude-code`), or
  `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) set (`--provider gemini`). `auto` picks Gemini when a key is
  set. Models are in `config/brain.yaml`.
- Models download into `data/models/` on first use (MediaPipe face landmarker, Vosk, Whisper).
- Your user needs the serial ports: `sudo usermod -aG uucp $USER` (log out and in), or for one session
  `sudo chmod a+rw /dev/ttyACM0` (the arm's adapter; it is `root:uucp`).

Run everything from the repo root with `uv run`: `uv run bebop` runs `core.run`, and each
tool has a short name (`uv run arm-check`, `teach`, `calibrate`, `list-cameras`, `face-check`,
`voice-check`, `esp32-check`). `uv run python -m <package>.<tool>` works too.

## Bring-up order (teammate, on the arm laptop)

1. `uv run list-cameras`: saves a frame per `/dev/video*` into `data/cameras/`. Put the
   right sources (better: `/dev/v4l/by-id/...` paths) for `top`, `side`, `face` into
   `config/cameras.yaml`.
2. `uv run arm-check`: prints joints, the FK tip and loads at 2 Hz. Hold the arm in the
   poses described in `config/robot.yaml` and fix `zero_offset_deg`/`sign` until the FK tip matches
   a ruler. REPL: `xyz X Y Z [PITCH]`, `q NAME DEG`, `open MM`, `close`, `home`, `torque off`;
   space = stop.
3. `uv run teach`: torque goes off, move the arm by hand, Enter saves `home`, `stage`
   (15 cm in front of the mouth) and `mouth` (tool at the lips) into `config/poses.yaml`.
   `--goto NAME` replays one.
4. `uv run calibrate`: yellow tape on the gripper, cameras rigid. The arm visits ~15
   tip poses, fits both cameras and saves `config/calib.json` with the RMS px per camera.
   `--check` draws the predicted tip into `data/calib_check/<cam>.jpg`.
   Poses where the tape is hidden are rejected.
   Optional: `uv run face-check`, `uv run voice-check`.
5. Fake end-to-end (no hardware, real VLM):
   `uv run bebop --fake --mode solid --task "feed me a donut hole"`
6. Real: `uv run bebop --mode solid --task "feed me a donut hole" --esp32 /dev/ttyUSB0 --ntfy TOPIC`,
   then the demo loop without `--task` (it listens): `uv run bebop --mode spoon --esp32 ... --ntfy ...`

Other flags: `--provider`, `--model`, `--no-dashboard`, `--port`, `--budget USD` (VLM spend for the
run; default `run_budget_usd` in `brain.yaml`, else $1). Each run logs to
`data/runs/<timestamp>/events.jsonl` with the frames next to it.

## Live-tested setup notes

From the hardware session of 2026-11-08 (00:00-01:00), on the real SO-101 through LeRobot 0.4.4, in
the order you hit them:

- Arm port: the adapter is `/dev/ttyACM0`, `root:uucp`. Join the `uucp` group or run
  `sudo chmod a+rw /dev/ttyACM0`.
- Arm accuracy: a fully stretched move and an IK move to (200, 0, 120) mm landed within about 12 mm.
- Servo gain: LeRobot's default P of 16 let the shoulder sag 24 deg short under gravity at the 5.3 V
  supply. `config/robot.yaml` sets P=32.
- Load sign: LeRobot 0.4.4 returns the servo load already signed; `hw/arm.py` handles both encodings.
- Cameras: a webcam held by Kamoso or a browser tab makes V4L2 captures fail with "busy"; close
  them first. The Logitech C270 needs about 1 s of frames before its exposure settles, so the first
  frames are black.
- Top camera: the C270 has no zoom (about 50 x 38 deg in 4:3), so mount it 60-70 cm above the table.
- Marker: yellow tape (HSV H 16-38). The first calibration used one camera and gave 12 px RMS after
  rejecting poses where the tape was hidden.
- Microphone: works through the PipeWire device (`voice.device: pipewire`). "stop" fires the voice
  e-stop. "halt" was removed from the stop words because it fired on room chatter.
- Shutdown: the arm folds into a rest pose before torque goes off, so nothing drops at the end of a
  run. After a stop or Ctrl+C it stays where it froze.

## Dashboard

`http://localhost:8000/` (the run prints the LAN address too). Three camera streams with overlays
(VLM box, target cross, predicted tip as a hollow circle, detected marker as a filled circle), the
run numbers, the attempts with their plan, prediction, per-action predicted/measured/error, verdict
and lesson (← → or the strip; Live follows the newest), a chart of position error and predicted vs
actual success per attempt, and the Stop, Resume and Release torque buttons.

## Controls

The person being fed (voice and face only):
- Say what you want: "can I have a donut hole", "some yogurt please". That starts a task.
- "stop", "wait" (the `stop_words` in `config/brain.yaml`): the arm freezes at once (spotted while you are still speaking).
- At the stage pose the arm waits for an open mouth or a "yes", then comes slowly toward the mouth.
  It stops short, or where the ultrasonic sensor says you are close. Solid food: keep the mouth open
  for a second and the gripper lets go. Spoon: close your mouth on it and the arm pulls back.
- Looking away while it picks food pauses it; turning away while it presents stops it.

The caregiver: the ESP32 button, the space bar in the run's terminal, and the dashboard. Stop freezes
the arm (it holds); a second stop, or Release, turns torque off. Resume (or saying "continue") clears
the stop and the arm goes home slowly. After three failed attempts, or when the planner says it is
unsafe, the robot goes to HELP: red LED, a phone notification, and it waits for Resume or "continue".

## Caregiver phone notifications (ntfy)

1. Pick an unguessable topic, e.g. `bebop-help-7f3k2q` (anyone who knows it can read and post).
2. Install the ntfy app (Play Store, F-Droid or App Store), tap +, enter the topic, server `ntfy.sh`.
3. Run with `--ntfy bebop-help-7f3k2q` (or `export NTFY_TOPIC=...`). Test: `curl -d hi ntfy.sh/<topic>`.
   A HELP notification opens the dashboard when tapped (phone and laptop on the same network).

## Safety

- Never be in front of the arm while `tools.calibrate`, `tools.teach --goto` or `arm_check` moves
  it, and keep the person out of reach until the poses and the fence (`config/workspace.yaml`) are
  checked.
- Torque off makes the arm sag and drop. A normal end of a run first folds the arm slowly into
  `rest_pose` (`config/robot.yaml`), so nothing drops. Ctrl+C, Release and a second stop cut torque
  where the arm froze: keep a hand under the arm, and keep the delivery pose low so it falls away
  from the face.
- Stop = freeze and hold; nothing moves again until someone resumes, and then only home, slowly.
- The ESP32 must be connected for a real demo: its button is the e-stop and silence for 500 ms
  counts as one. The final approach is slow, gated by the face camera and the ultrasonic sensor.
- The VLM only returns pixels and choices; every move goes through the fence and the speed cap.
- Hygiene: a fresh spoon per person, gloves; a teammate does the default eating.
