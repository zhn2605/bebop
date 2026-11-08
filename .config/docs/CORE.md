Byte to Bite: Build Spec
Nov 7, 2026 · @Zihan
Overview
Byte to Bite is an adaptive feeding robot for people who can't feed themselves, built at MakeCU 2026 for the Healthcare track. An SO-101 arm with a spoon scoops food from a bowl and brings it to the user's mouth. Its key ability: it learns each new food from its own failed scoops. Before each scoop it predicts the outcome, measures what actually happened, and adjusts the next attempt.
This spec is the single source of truth for building it. It supersedes the hardware and sensor details in the earlier team plan. An agent should be able to start at the build phases section and work upward from power-on to the finished demo.
What's on hand right now: the SO-101 arm (6 STS3215 servos, assembled) and a Waveshare Bus Servo Adapter (A). Everything else comes from the MakeCU hardware inventory, or software.
Locked decisions:
• Name: Byte to Bite. The robot can be called Nibble.
• Track: Healthcare (assistive robots for people with disabilities).
• Brain: a fast multimodal model (Gemini Flash; Claude Haiku as an alternative) makes ONE structured decision per attempt. It never commands joints directly.
• Motion: poses taught by hand plus interpolation. No IK in v1.
• Perception: OpenCV color thresholds and a homography for the bowl; MediaPipe for the mouth.
• Safety: fixed geometry first; then speed caps, an ultrasonic distance check, servo-load contact detection and an e-stop.
• Out of stock, already substituted: Facmogu supply, TouchTronix sensor, thin-film pressure sensor, VL53L5CX, Pseudo-LiDAR, NeoPixel ring, lamp.
Constraints
The build has about 18 hours left: submission is due Sun Nov 8 at 12:00 pm. Every decision below favors something that works over something clever.
Deadlines:
Event
When
Hardware ordering windows
Sat 5–7 pm, Sat 11 pm–1 am, Sun 8–10 am
Feature freeze (our rule)
Sun 9:00 am
Devpost submission
Sun 12:00 pm
Expo, judges walk up
Sun 1:00–3:00 pm
Hardware return, Mudd 829
Sun 3:00 pm
Rules from the hacker handbook:
• Our own hardware (the arm) needs organizer approval, case by case.
• The project must be built during the event. Disclose that the arm was assembled before it.
• Credit every library, dataset and design we didn't make.
• Only use makerspace equipment you've been trained on. 3D printing: one printer per group, 4 hours max per print.
• Borrowed hardware must come back in the condition we got it. Never repair MLH components.
Environment:
• One laptop runs everything. Venue Wi-Fi may be unreliable, so keep a phone hotspot ready and have offline fallbacks.
• No lamp or depth camera available. Lighting will change between the build table and the expo table.
• The expo is loud, so voice input needs a keyboard fallback.
Architecture
The VLM is called once per attempt, only while the arm is still. Servo control, perception and every safety check run locally at 20–50 Hz. A full attempt takes about 10–15 s: perceive ~0.1 s, decide 1–3 s, scoop 3–4 s, check ~0.2 s, deliver and return ~5 s. Planning the next bite while the user chews hides the decision time.
Hardware
Every part is either already on hand or comes from the MakeCU inventory under its handbook name. Power depends on the servo voltage, which is still unconfirmed (see Open questions).
On hand:
Part
Notes
SO-101 arm, 6× STS3215 servos
Assembled before the event; disclose it in Devpost
Waveshare Bus Servo Adapter (A)
Power input 9–12.6 V nominal, also 5–8.4 V; DC 5.5×2.1 mm jack and screw terminal; USB-C to host; jumper A = UART, B = USB
Order from the inventory:
Part (handbook name)
Qty
Source
Use
USB-C Cable
1
CURC
Servo adapter to laptop
DMM
1
MLH
Check voltage and polarity before every first power-up
Wire Spool
1
MLH
Power leads into the screw terminal
Wire Strippers K604-ND
1
MLH
Strip leads
Alligator Clip Test Lead
2
CURC
Temporary connections
Battery Bank
2
CURC
5 V power if the servos are 7.4 V
USB 3.1 Type male plug breakout
1
CURC
Takes 5 V from a battery bank's USB-C port
3S Lipo Battery
2
CURC
11.1–12.6 V power if the servos are 12 V; ask CURC for a mating lead
Resistor Kit
1
MLH
5.1 kΩ USB-C CC pull-down, ultrasonic echo divider, LED resistors
Logitech Webcams
2
MLH
Face cam and overhead cam
Small Camera Tripod
1
CURC
Face cam mount
ESP32 ESP-WROOM-32 Development Board
1
CURC
Sensor and status hub
Ultrasonic Distance Sensor (CURC)
1
CURC
Face distance at the delivery point
SSD1306 0.96-inch OLED Display
1
CURC
Status text for the user
Misc. LEDs
3
MLH
Green ready, yellow thinking, red stopped
Tactile Push Buttons (Assorted)
2
CURC
E-stop; optional compliant-mount contact switch
Solderless Breadboard (Half Size)
1
CURC
ESP32 wiring
41mm Piezo Disc
1
CURC
Optional contact tap sensor
Endoscope Inspection Camera
1
CURC
Optional spoon cam, only if the overhead spoon check fails
Out of stock, and what replaces it:
Out of stock
Replacement
Facmogu Adjustable DC Power Supply
Battery bank + USB-C breakout (7.4 V servos), or 3S LiPo (12 V servos)
TouchTronix tactile sensor, thin-film pressure sensor
Servo load reading in software; optional compliant mount + tactile button, or piezo disc
VL53L5CX, Pseudo-LiDAR
Ultrasonic sensor + camera face-size estimate
NeoPixel ring
OLED + 3 LEDs
Lamp
Locked camera exposure, white paper backdrop, phone flashlight
Not in the inventory, get elsewhere: clamps (ask a makerspace superuser), a powered USB hub, a small screwdriver, jumper wires (untracked, ask the hardware desk), a cardboard box for the overhead camera mount, a white plastic spoon, a wide heavy bowl, gloves, napkins, and the test foods (cheese cubes, grapes, banana).
At pickup, check: which USB connector the ESP32 has (the inventory lists no micro-USB cable), and what connector the 3S LiPo uses.
Power and wiring
The servo bus and the sensors are two separate electrical systems. Each talks to the laptop over its own USB cable, so they never share power and need no level shifting between them.
Servo power
The adapter's input voltage must match the servo voltage. Read the sticker on one servo first.
Servo version
Power source
Connection
7.4 V
Battery Bank → USB-C male breakout (5 V)
Breakout VBUS and GND to the adapter's screw terminal. Same voltage the stock SO-101 kit uses for these servos
12 V
3S LiPo (11.1–12.6 V)
Pigtail lead to the screw terminal. Stop using a pack once it reads about 10.5 V
First power-up, every time a source changes:
1. Nothing connected to the adapter. Measure the source with the DMM: correct voltage, correct polarity.
2. If a USB-C breakout reads 0 V, the source wants a sink resistor: add 5.1 kΩ from CC to GND on the breakout.
3. Wire red to + and black to − on the screw terminal, matching the board's markings. Measure again at the terminal.
4. Set the adapter's jumper to B (USB) and connect USB-C to the laptop.
5. Power on, find the port in LeRobot, and move one joint slowly before anything else.
If servos stutter or LeRobot drops the connection mid-move, that's a brownout: slow the motion down or use a stronger source.
ESP32 hub pin map
Device
ESP32 pin
Wiring
E-stop button
GPIO27
Button to GND; INPUT_PULLUP; pressed = LOW
Ultrasonic TRIG
GPIO5
Direct
Ultrasonic ECHO
GPIO18
Through a divider: ECHO → 1 kΩ → GPIO18, and GPIO18 → 2 kΩ → GND (5 V → about 3.3 V)
Ultrasonic VCC / GND
VIN (5 V from USB) / GND
The sensor needs 5 V
OLED SDA / SCL
GPIO21 / GPIO22
I2C, powered from 3V3 and GND
LED green (ready)
GPIO25
220 Ω in series, to GND
LED yellow (thinking)
GPIO26
220 Ω in series, to GND
LED red (stopped)
GPIO33
220 Ω in series, to GND
Compliant-mount button (optional)
GPIO32
Button to GND; INPUT_PULLUP
Piezo disc (optional)
GPIO34
1 MΩ across the disc to GND; GPIO34 is input-only, ADC1
Avoid GPIO0, 2, 12 and 15 (boot-strapping pins), and never use GPIO34–39 as outputs.
ESP32 ↔ laptop serial protocol
USB serial at 115200 baud, one JSON object per line.
• ESP32 → laptop, every 50 ms: {"t": 12345, "estop": 0, "dist_cm": 23.4, "contact": 0}. dist_cm is null when there's no echo. contact is 1 if the compliant-mount button or the piezo fired.
• ESP32 → laptop, immediately on e-stop: {"event": "ESTOP"}. The laptop turns servo torque off on receipt.
• Laptop → ESP32: STATE <name> sets the OLED text and the LEDs. Names: READY, THINKING, SCOOPING, DELIVERING, OPEN_WIDE, STOPPED, HELP.
• Watchdog: if the laptop hasn't heard from the ESP32 for 500 ms, it treats that as an e-stop.
Physical layout and mounting
Everything that matters for calibration (arm, bowl, overhead camera) is fixed to one baseplate, so the rig can move to the expo table without recalibrating.
                    USER (seated on a taped mark)
                           mouth
   ---------------------- table edge ----------------------
   [face cam on tripod]     <> delivery point    [ultrasonic]
    aimed at the face         (spoon stops)       aimed at the face,
                                                  not in the spoon's path
   [OLED + LEDs]            +---------+
    facing the user         |  BOWL   |  <- overhead cam, angled down 45-60 deg
                            +---------+     from behind, opposite the arm
                        [ SO-101 arm base ]
   [ESP32 + breadboard] [Waveshare adapter] [power source]
   ======================== baseplate =======================
   [laptop + USB hub]  [e-stop on a lead]  [monitor facing judges]
Part
Placement
Notes
Baseplate
Rigid board, clamped (makerspace clamps) or weighted
Laser-cut if the makerspace has stock; any rigid board works
SO-101 arm
Bolted or zip-tied to the baseplate
Bowl within easy reach; delivery point at the edge of its reach toward the user
Bowl
Taped inside a drawn outline, on white paper
Contrasting color to the foods; 4 corner markers for the homography
Overhead cam
30–40 cm up behind the bowl, angled down 45–60°
Clip it on a taped cardboard box first; replace with a printed or laser-cut bracket on the baseplate by M1. Must see the whole bowl and the spoon check pose
Face cam
On the tripod beside the delivery point, slightly off-axis
Mouth height, 40–60 cm from the face; the arm must never block it
Ultrasonic sensor
Fixed beside the delivery point, aimed at the face
Mounted fixed so no sensor wires run along the moving arm
OLED + LEDs
Near the bowl, facing the user
Readable from the user's seat
Spoon
White plastic, in a zip-tied or printed mount
White makes food easy to count; optional compliant mount with a tactile button behind it
E-stop button
On a long lead, in the operator's hand

Two checks before mounting anything:
1. Reach. Torque off, move the spoon by hand to where a seated person's mouth would be. If the arm can't reach, raise its base on a sturdy block or seat the user lower.
2. Visibility. Pick the spoon check pose just above the bowl and confirm the overhead camera sees it clearly before you teach it.
Software stack and repo layout
Python on one laptop, plus one Arduino-framework sketch on the ESP32. Use libraries for everything that already exists; write only the glue and the loop.
Need
Use
Notes
Arm control, calibration
LeRobot (SO-101 follower, Feetech bus)
Read positions and load; write goal positions; torque on/off
Cameras, color threshold, homography
OpenCV
Lock exposure and white balance on both webcams
Mouth tracking, face size
MediaPipe Face Landmarker
Mouth openness and face width from landmarks
VLM decisions
Gemini Flash with a JSON response schema
Claude Haiku is a drop-in alternative; temperature 0
Numeric baseline
A small random or Bayesian search over the same parameters
Also the offline fallback
ESP32 firmware
Arduino framework; Adafruit SSD1306 library
Implements the serial protocol
Dashboard and caregiver page
Streamlit
Reads the live log file
Voice input (last)
Browser speech API or Whisper
Keyboard always works
IK (only if needed)
ikpy with the SO-101 URDF
Not used in v1
byte-to-bite/
  config/
    poses.yaml          # taught joint poses: home, corners, dip deltas, check, delivery
    calibration.json    # bowl homography, check-pose crop, face-size constant
    foods.yaml          # per-food HSV ranges and default params
    safety.yaml         # speed caps, distance and load thresholds, param ranges
  firmware/esp32_hub/   # serial protocol, e-stop, ultrasonic, OLED, LEDs
  hw/
    arm.py              # LeRobot wrapper: connect, read_joints, read_load, move_to, torque
    esp32.py            # serial reader thread + watchdog; send_state()
  perception/
    cameras.py          # open both webcams, locked exposure, grab frames
    bowl.py             # homography, food segmentation, grid overlay, candidate cells
    spoon_check.py      # crop at the check pose, food-pixel ratio
    face.py             # mouth open ratio, face width, distance estimate
  skills/
    motion.py           # time-based interpolation between joint poses, speed caps
    scoop.py            # parameterized scoop built from taught keyframes
    deliver.py          # delivery, wait for open mouth, retract
  brain/
    vlm.py              # one call per attempt: schema, validation, clamping, timeout, fallback
    memory.py           # per-food attempts and lessons
    search.py           # numeric baseline / offline fallback
    prompts/
  core/
    state_machine.py    # the bite loop
    safety.py           # monitor thread: e-stop, distance, load, face present
  tools/
    teach_pose.py       # torque off, move by hand, save the pose by name
    calibrate_bowl.py   # click 4 markers -> homography
    calibrate_colors.py # pick HSV ranges per food from a live frame
  experiments/run_trials.py  # modes: fixed, search, vlm
  dashboard/app.py
  data/logs/ and data/images/
Interfaces and data
These contracts let four people build in parallel. Change them only by agreement, and update this section when you do.
Taught poses (config/poses.yaml)
All poses are joint angles recorded with torque off by tools/teach_pose.py.
• home, check (spoon just above the bowl, in the overhead camera's view), delivery (a few cm short of the mouth)
• corner_1, corner_3, corner_7, corner_9: the spoon hovering about 3 cm over each corner cell of a 3×3 grid on the bowl
• dip_delta, drag_delta, tilt_delta, lift_delta: joint-angle differences taught once at the center cell, then added to any cell's hover pose
A cell's hover pose is a bilinear blend of the four corner poses. No IK.
Scoop parameters
Param
Range (clamped in safety.yaml)
Meaning
cell
1–9
Grid cell, row-major from the far-left corner
depth
−1.0 to +1.0
Scales dip_delta; 0 = taught depth
wrist_deg
−20 to +20
Wrist-pitch offset during the dip and drag
drag
0.0 to 1.0
Fraction of drag_delta to apply
speed
slow, medium, fast
Motion-duration multiplier 1.5, 1.0, 0.7
Module functions
Module
Function
Returns
hw/arm.py
connect(), read_joints(), read_load(), move_to(pose, duration_s), torque(on)
Joints and loads as dicts keyed by joint name
hw/esp32.py
latest(), send_state(name)
{estop, dist_cm, contact, age_ms}
skills/scoop.py
scoop(params)
Duration and peak wrist load (a bowl-contact signal)
skills/deliver.py
go_check(), deliver(), home()

perception/bowl.py
candidates(food), grid_image()
Cells containing food; overhead frame with the numbered grid drawn on
perception/spoon_check.py
check(food)
{got_food, food_ratio, amount, crop}
perception/face.py
read()
{present, mouth_open, face_width_px, dist_est_cm}
brain/vlm.py
decide(context)
A validated, clamped Decision, or the fallback
brain/search.py
propose(food, history)
Params from numeric search
brain/memory.py
load(food), add_attempt(record), add_lesson(food, text)

core/safety.py
ok()
(bool, reason)
VLM decision (one call per attempt)
Input: the food name, attempt number, the overhead image with the grid and candidate cells marked, the last attempt's spoon crop, the last params, prediction and measured outcome, the food's lessons, and a compact table of past attempts.
Output schema:
{
  "diagnosis": "string, why the last attempt went as it did; empty on attempt 1",
  "lesson": "string or null, a reusable rule for this food",
  "params": {"cell": 5, "depth": 0.3, "wrist_deg": 10, "drag": 0.5, "speed": "slow"},
  "prediction": {"success_prob": 0.6, "expected": "one piece, may roll to the wall", "risk": "none | rolled | broke | slid_off | missed"}
}
Rules: temperature 0; validate against the schema; clamp every number to its range; cell must be a candidate cell; a 5 s timeout or any error falls back to the best params so far for that food, or to foods.yaml defaults.
Outcome (deterministic first)
{"got_food": true, "food_ratio": 0.18, "amount": "none | small | good", "failure": "none | missed | rolled | broke | slid_off | dropped"}. got_food and amount come from pixel counts. The failure label comes from the VLM's diagnosis on the next call.
Log record (data/logs/<session>.jsonl, one line per attempt)
trial_id, mode (fixed / search / vlm), food, episode, attempt, timestamp, params, prediction, outcome, lesson, latency (vlm_s, scoop_s), images (paths to the overhead and spoon crop), safety_events. This file feeds the dashboard and the dataset export.
State machine
1. HOME → WAIT_REQUEST (voice or keyboard picks a food)
2. PLAN (VLM or search) → SCOOP → CHECK
3. On failure: back to PLAN. After 3 failures on one food: HELP (the caregiver page offers a fix, saved as a lesson), then PLAN.
4. On success: DELIVER → WAIT_BITE (mouth open, user leans in) → CONFIRM_EMPTY (spoon check again) → HOME
5. Any state → STOPPED on a safety event. Only the operator can leave STOPPED, and the arm goes back to HOME slowly.
Every state change also calls send_state() so the OLED and LEDs match.
Build phases
Work bottom up. Each phase ends with an acceptance test that must pass before the next phase depends on it. Git-tag every passed phase, and never break the last tag. After P0, phases P1, P2 and P3 can run in parallel on different people.
P0: Bring-up (target Sat 7 pm)
[ ] Read the servo voltage sticker; pick the power source from the Power section
[ ] Get the arm approved by an organizer
[ ] Measure the source with the DMM, wire it to the screw terminal, set the jumper to B
[ ] Install LeRobot; find the port; calibrate the SO-101 follower
[ ] Clamp or weigh down the baseplate; mount the arm
Done when: a script moves each joint 10° and back, reads back positions and loads, and turns torque off on Ctrl-C.
P1: Motion (target Sat 9 pm)
[ ] tools/teach_pose.py: torque off, move by hand, save named poses
[ ] Teach home, the four corners, the deltas, check and delivery
[ ] skills/motion.py: time-based interpolation with a speed cap; slower near delivery
[ ] skills/scoop.py with all five parameters; deliver.py
[ ] Keyboard e-stop (space bar = torque off)
Done when: a keypress runs home → scoop at any cell → check pose → delivery → home, 10 times in a row with the bowl full, without a collision.
P2: Perception (target Sat 11 pm)
[ ] Both webcams open with locked exposure and white balance
[ ] Mount the overhead cam; tools/calibrate_bowl.py (click 4 markers)
[ ] tools/calibrate_colors.py: HSV range per food, saved to foods.yaml
[ ] bowl.candidates() and grid_image()
[ ] spoon_check.check() at the check pose
[ ] face.read(): mouth open, face width; calibrate the distance constant at a known distance
Done when: across 20 manual scoops, the spoon check agrees with a human judge at least 18 times; mouth open/closed is right on 10 tries; candidate cells match where the food actually is.
P3: ESP32 hub (target Sat 11 pm)
[ ] Wire per the pin map on a breadboard
[ ] Firmware: 50 ms telemetry, immediate ESTOP event, STATE commands to the OLED and LEDs
[ ] hw/esp32.py: reader thread, 500 ms watchdog
Done when: pressing the button turns arm torque off within 200 ms; unplugging the ESP32 also stops the arm; dist_cm tracks a hand moving 5–50 cm in front of the sensor.
P4: Brain (target Sun 2 am)
[ ] brain/vlm.py: one call per attempt, schema, clamping, 5 s timeout, fallback
[ ] brain/memory.py and brain/search.py
[ ] core/state_machine.py: the full bite loop with logging
[ ] Caregiver HELP page after 3 failures
Done when: asked for a food, the system runs the loop end to end, retries on failure, delivers on success, and writes a complete log line for every attempt. Pulling the network cable falls back to search without crashing.
P5: Safety integration (target Sun 3 am)
[ ] core/safety.py monitor at 20–50 Hz: e-stop, ESP32 watchdog, distance, servo load, face present
[ ] Log wrist and elbow load for 10 normal scoops and 10 free-air moves; set the spike threshold between them
[ ] Every requirement in the Safety section implemented and tested
Done when: each safety trigger, tested one at a time, stops the arm and enters STOPPED.
P6: Experiments and dashboard (target Sun 7 am)
[ ] experiments/run_trials.py for the fixed, search and vlm modes
[ ] Run the protocol in the Experiments section
[ ] Streamlit dashboard: live prediction vs. outcome, lesson log, learning curve per mode
[ ] Voice input, if time allows
Done when: the dashboard shows real trial data with all modes on one chart.
P7: Ship (freeze Sun 9 am, submit by 11 am)
[ ] 30-minute reliability run; record the success rate
[ ] Backup video of a full run
[ ] Everything in the Demo and submission section
Done when: Devpost is submitted with the video and all tags, before noon.
Safety requirements
Geometry is the primary safety layer: the spoon can only go to a taught delivery point a few centimeters short of the mouth, and the user leans in. Every other layer is a backup. All thresholds live in config/safety.yaml.
#
Requirement
Mechanism
Starting value
S1
The spoon never passes the delivery point
Software workspace limit; delivery is the furthest pose toward the user
Taught pose
S2
Slow final approach
Speed cap on the last segment into delivery
Half of normal speed
S3
No motion toward the face unless a face is seen
face.read().present must be true before and during DELIVER
Required
S4
Stop if the face is too close while moving
Ultrasonic dist_cm, cross-checked with the face-size estimate
Hold under 5 cm
S5
Stop on unexpected contact
Wrist or elbow load above threshold outside the scoop dip; optional compliant-mount button or piezo
Calibrated in P5
S6
Physical e-stop
Button → ESP32 → ESTOP event → torque off
Within 200 ms
S7
Lost sensor link = stop
No ESP32 message for 500 ms counts as an e-stop
500 ms
S8
Hardware cut-off
Unplug the power source at the screw terminal side
Operator
S9
The VLM never moves the arm directly
It returns parameters only; every value is clamped to its range
Always
S10
Recovery is deliberate
Only the operator leaves STOPPED; the arm returns home at slow speed
Always
Notes:
• Cutting torque makes the arm sag. Keep delivery low enough that a sagging arm falls away from the face.
• Test every requirement on its own before the expo, and write the results in the Devpost writeup. This is the safety panel for the Clarapath judge.
• Hygiene: a fresh plastic spoon per person, gloves, and judges opt in. A teammate does the default eating.
Experiments and metrics
The experiment answers one question: does the VLM-guided loop learn a new food in fewer attempts than a fixed scoop or blind numeric search? The resulting chart is the project's main evidence.
Modes:
Mode
How params are chosen
fixed
The same taught scoop every time; what existing feeders do
search
Numeric random or Bayesian search over the five params, using only success or failure
vlm
The full loop: prediction, diagnosis, lessons
Protocol:
1. Foods: cheese cubes (easy), grapes (roll), banana slices (squish). Refill the bowl to the same level before each episode.
2. An episode = fresh memory, then 6 consecutive scoop attempts of one food. Record success for every attempt; don't stop at the first success.
3. Run 3 episodes per food per mode. Priority order if time runs short: vlm and fixed first, search last.
4. Full run: 3 foods × 3 modes × 3 episodes × 6 attempts = 162 attempts, about 1.5 hours with resets. The minimum is vlm + fixed: 108 attempts, about an hour.
Metrics:
• Headline: success rate by attempt number (1–6), averaged over episodes, one line per mode.
• Attempts to first success, per food and mode.
• Prediction quality: the gap between predicted success_prob and the actual outcome, falling over attempts. This is the world-model angle for the Midcentury judges.
• Latency: median VLM seconds and scoop seconds per attempt.
• Reliability (P7): success rate over a continuous 30-minute run.
Report the numbers honestly. If vlm doesn't beat search on some food, say so, and show where the diagnosis helped.
Demo and submission requirements
Judges score five criteria, and the 3-minute Devpost video must cover each one, in this order.
Criterion
What we show
Problem solving
Caregiver dependence for every bite; picking up a bite as an open research problem; existing feeders replay fixed motions
Innovation
Per-food learning from failure; predict, then measure
Technology
Three-layer architecture, grid prompting, two cameras + ultrasonic + servo load + ESP32 hub, the baseline experiment
Design
Voice or keyboard request, OLED and LED status, lean-in delivery, caregiver help page, live dashboard
Completion
Reliability-run success rate, a live demo, the backup video
Submission checklist:
[ ] 3-minute video covering the five criteria
[ ] Healthcare track tagged
[ ] Notability tagged, 2+ canvas screenshots, 1–2 sentences on how we used it (keep sketching in it during the build; it's judged on real use)
[ ] MLH Best Use of Gemini API tagged
[ ] Trace PCB files, only if entered
[ ] Architecture diagram, learning-curve chart, safety test results
[ ] Disclosure that the arm was assembled before the event
[ ] Credits: SO-101 design (TheRobotStudio, Hugging Face), LeRobot, MediaPipe, OpenCV, Gemini
[ ] Repo link
Expo setup (Sun 1–3 pm): one person pitches, one runs the robot; the presenter never types. The dashboard goes on a monitor facing judges. Between visitors, run attract mode so the chart keeps moving.
Pitch (about 45 seconds): "People who can't use their arms depend on someone else for every bite. Feeding robots exist, but they replay fixed motions, and every food behaves differently: grapes roll, banana squishes. Byte to Bite learns each new food from its own mistakes. Before every scoop it predicts what will happen, measures how wrong it was, and uses that next time. Watch it miss the grape, then figure it out. Want to pick the next food?"
One line per judge after the pitch:
• Midcentury Labs (Ali Al-Bustami, Brandon Samaroo): "Every bite is a logged prediction and outcome, with a baseline."
• Tutor Intelligence (David Watkins): "When it's stuck it asks the caregiver, and the fix becomes a lesson."
• Clarapath (Aniket Dhole): "Here are our safety layers and the test result for each."
• BotBlox (Josh Elijah): "The safety loop is local. Nothing near the face waits on the network."
Open questions, risks and fallbacks
Open questions (answer these first):
[ ] Servo voltage: 7.4 V or 12 V? This decides the power source.
[ ] Has an organizer approved the arm?
[ ] Which USB connector does the ESP32 board have, and is there a matching cable?
[ ] What connector does the 3S LiPo use, and can CURC supply a lead?
[ ] Can the arm reach a seated person's mouth height from the baseplate?
[ ] Is there an expo monitor we can use (for the dashboard, and possibly as the overhead camera mount)?
Risks:
Risk
Fallback
No usable power for the servos
Battery bank at 5 V for slow moves only; ask the hardware desk about a returned Facmogu every window
Arm not approved
Build a 3-DOF feeder from the inventory's goBILDA 2000-series and DS3218MG servos on the same software stack
Brownouts during motion
Slower speeds, shorter moves, a second battery bank
Venue Wi-Fi drops Gemini calls
Phone hotspot; automatic fallback to search and the best known params; backup video
Spoon check unreliable in expo lighting
Re-run calibrate_colors.py at the expo table; white backdrop; phone flashlight; the endoscope as a dedicated spoon cam
Overhead camera gets bumped
calibrate_bowl.py takes 30 seconds; move to the baseplate bracket by P2
Grapes too hard for the arm's precision
Swap to blueberries or cereal; keep one food that clearly fails at first
No visible learning in the chart
Choose foods where the default scoop clearly fails; run more episodes
Servo dies
Ask on Discord for a spare STS3215; run slower to cut load
Time runs out
Demo the last passed phase tag and say honestly what's next
