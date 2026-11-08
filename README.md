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
4. `uv run calibrate`: close the jaws on a small ball in a colour nothing else in view has
   (`ball:` in `config/cameras.yaml`, its centre in `tool_offsets.ball` of `config/robot.yaml`),
   cameras rigid. The arm carries it through 27 points from the food's height up (`calib_box`),
   fits both cameras and prints the held-out error (aim: top < 3 px, side < 6 px, < 5 mm). A fit
   no real camera could have (under the table, a wild focal length) is not saved.
   `--check` draws the predicted ball into `data/calib_check/<cam>.jpg`.
   Optional: `uv run face-check`, `uv run voice-check`.
5. Fake end-to-end (no hardware, real VLM):
   `uv run bebop --fake --mode solid --task "feed me a donut hole"`
6. Real: `uv run bebop --mode solid --task "feed me a donut hole" --esp32 /dev/ttyUSB0 --ntfy TOPIC`,
   then the demo loop without `--task` (it listens): `uv run bebop --mode spoon --esp32 ... --ntfy ...`

Other flags: `--provider`, `--model`, `--no-dashboard`, `--port`, `--budget USD` (VLM spend for the
run; default `run_budget_usd` in `brain.yaml`, else $1). Each run logs to
`data/runs/<timestamp>/events.jsonl` with the frames next to it.

## Demo layout and the end-to-end run

The judge sits in a chair. To one side, a pole holds the `top` camera over the plate. The plate is
in front of the judge, the arm beside it. The laptop sits behind the plate, its webcam (`face`)
looking at the judge. The second Logitech (`side`) is off to the side and sees the plate and the
judge's head from the side. The ultrasonic sensor sits on top of the wrist, pointing along the
gripper. ESP32: e-stop button GPIO27 to GND, HC-SR04 TRIG GPIO5 and ECHO GPIO18 through the 1k/2k
divider (VCC on VIN), LEDs green GPIO25, blue GPIO26 (the "yellow" slot), red GPIO33. There's no
OLED, contact switch or piezo; `PIEZO_ENABLED` is 0 in the firmware. Reflash after pulling.

How delivery works with a calibrated `face` camera (`core/steps.py` `deliver_tracked`):
1. The mouth's 3D position comes from the webcam's mouth pixel. Its distance comes from the side
   camera when MediaPipe finds the face there (up to about a three-quarter view), or from the eye
   spacing otherwise (`face.eyes_mm`, about 3 cm off when the guess is 5 mm wrong). The dashboard's
   face stream shows the mouth and how far the food is from it.
2. The arm goes slowly to a point 15 cm out from the mouth (`stage_mm`), on the line back toward
   its base. There it waits for an open mouth or a "yes".
3. It comes in slowly and stops `stop_short_mm` (6 cm) short. It never goes into the mouth: the
   judge leans in the rest. While it moves, the ultrasonic (minus `sensor_behind_food_cm`) slows it
   to a creep within 15 cm and stops it 6 cm away. It e-stops within 3 cm on two readings in a row,
   so one false echo does nothing; the arm then goes home. The face leaving the webcam also stops
   it, and if the head moves 3 cm it backs out and aims again.
4. Holding still, the ultrasonic no longer stops anything, because the judge is meant to come
   close. A bite (mouth open, then closed at the food) makes the gripper let go. The arm backs out
   and goes home. "Eaten" means the VLM sees the gripper empty, plus chewing on the webcam.

Without a calibrated `face` camera, delivery uses the taught `stage`/`mouth` poses as before.

Setup rules:
- The arm base, the cameras and the plate stay fixed. Clamp them, tape the plate down, and don't
  move the laptop or its lid after calibrating. Anything that moves means calibrating again.
- Keep the mouth within about 40 cm of the arm's base (horizontally), at a height the arm reaches
  level. Further out, the food stops at the edge of reach and the judge has to lean further.
- Mount the ultrasonic so its cone (about 15 deg) clears the jaws and the food. Otherwise it reads
  them as a face. Check with `uv run esp32-check`: nothing in front should read far or no echo.
  Then hold a card flat against the food and put that reading into `sensor_behind_food_cm`.
- Light the judge's face from the front. Nothing yellow or ball-coloured should be in the cameras'
  view.

End-to-end bring-up:
1. Check the heights: `uv run arm-check`, `torque off`, touch the jaw tips to the plate and read
   z. It must match `table_z` in `workspace.yaml` (now -95). If it doesn't, fix that first, and
   move `calib_box` z with it (its bottom layer sits just above the plate).
2. Seat someone, `uv run teach` and save `stage` (15 cm in front of their mouth). That only tells
   the calibration where the face will be. Or set `face_box` in `workspace.yaml`.
3. Empty the chair. Run `uv run calibrate` with the ball: 27 poses over the plate, then 27 in front
   of the face. Every camera should show up in "samples per camera", and the face camera needs at
   least 6.
4. `uv run bebop --mode solid --task "feed me a donut hole" --esp32 /dev/ttyUSB0`, with the judge
   seated and a hand on the e-stop.

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
the arm, then after a second it goes home slowly and holds there with torque on (a stop on the way
freezes it where it is). Only Release turns torque off. Resume (or saying "continue") goes on with
the task. After three failed attempts, or when the planner says it is
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
  `rest_pose` (`config/robot.yaml`), so nothing drops. Ctrl+C folds to rest too. Release cuts torque
  where the arm is: keep a hand under the arm, and keep the delivery pose low so it falls away
  from the face.
- Stop = freeze, then home slowly and hold there; nothing else moves until someone resumes.
- The ESP32 must be connected for a real demo: its button is the e-stop and silence for 500 ms
  counts as one. The final approach is slow, gated by the face camera and the ultrasonic sensor.
- The VLM only returns pixels and choices; every move goes through the fence and the speed cap.
- Hygiene: a fresh spoon per person, gloves; a teammate does the default eating.
