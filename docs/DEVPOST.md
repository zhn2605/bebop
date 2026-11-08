# Byte to Bite - Devpost draft

Track: Healthcare (MakeCU 2026). Team: [TODO team: names]. Repo: [TODO team: link]. Video: [TODO team: link].

## Inspiration

People who can't use their arms depend on someone else for every bite. Feeding robots exist, but they replay fixed motions, and every food behaves differently: grapes roll, banana squishes. We wanted a feeder that reports what it expected, measures what happened, and uses the difference on the next try. The person being fed should control it with their voice and face only, and a caregiver should be pinged when it is stuck.

## What it does

An SO-101 arm feeds a person who asks for food by voice. A vision-language model finds the food in two fixed cameras and returns a plan. Geometry turns its pixel boxes into a 3D target, and every action reports predicted against measured numbers. If the grab misses, the model re-plans from those numbers and writes a lesson. Delivery waits for an open mouth, approaches slowly, and releases. After three failed attempts the robot asks for help on a phone.

## How we built it

The model returns pixels and choices, never joint angles. Marker-based camera calibration, analytic IK and a fence box turn those into moves. Two modes: `solid` (the bare gripper pinches) and `spoon` (a spoon strapped into the gripper scoops). Gemini or the `claude` CLI plans. MediaPipe reads the mouth, Vosk catches stop words, and an ESP32 provides the e-stop button and an ultrasonic distance check. A FastAPI dashboard shows the cameras, overlays, attempts and metrics. The whole loop also runs with a fake arm and cameras against the real model.

## Challenges we ran into

- LeRobot's default servo gain (P=16) let the shoulder sag 24 degrees short under gravity at the 5.3 V supply. P=32 fixed it.
- A webcam held by Kamoso or a browser tab made captures fail with "busy", and the C270 returns black frames for about 1 s.
- The C270 has no zoom, so the top camera has to sit 60-70 cm above the table.
- The first calibration gave 12 px RMS with one camera, after we rejected poses where the tape marker was hidden.
- "halt" fired on room chatter, so we removed it from the stop words.

## Accomplishments that we're proud of

- The real SO-101 runs through LeRobot 0.4.4. A fully stretched move and an IK move to (200, 0, 120) mm landed within about 12 mm.
- The arm folds into a rest pose before torque goes off, so nothing drops at shutdown.
- Voice stop works through the microphone.
- Every action logs predicted and measured values to `events.jsonl` with its frames: "Every bite is a logged prediction and outcome."
- The safety layers S1-S10 are written down.
- Results on real food: [TODO team: success rate, number of runs].

## What we learned

Hardware findings were mostly small and physical: servo gain, supply voltage, camera exposure, and whether a device was already open. Keeping the model out of the control loop made the safety story simple: it returns pixels, and the fence and speed cap check every move. Predicted-versus-measured numbers gave the planner something concrete to correct. Stop words need to be short and tested in the actual room. [TODO team: anything else learned at the expo]

## What's next

- Finish calibration with both cameras (the first fit used one).
- Measure the placeholder values in `config/robot.yaml`: the gripper opening at 100, the spoon tip offset, and the marker offset.
- Teach and test the `stage` and `mouth` poses and the head zone before a person sits in front of the arm.
- Compare against a fixed scoop and a numeric search baseline, as planned in CORE.md. [TODO team: status]

## Built with

- SO-101 arm (TheRobotStudio, Hugging Face), STS3215 servos, Waveshare Bus Servo Adapter
- LeRobot 0.4.4
- Python
- OpenCV
- MediaPipe
- Gemini API
- Claude (via the `claude` CLI)
- Vosk
- faster-whisper
- FastAPI and uvicorn
- Chart.js
- ESP32 with an ultrasonic sensor, e-stop button and LEDs
- ntfy.sh
- Logitech C270 webcams
- [TODO team: Notability, if used and tagged]

Disclosure: the arm was assembled before the event. Sponsor angles: metrics and data (a logged prediction and outcome for every action), safe medical robotics (the safety layers and the test result for each), caregiver ping (HELP goes to a phone).

## 60-second demo script

- 0:00 We say "can I have a donut hole"; the dashboard shows the task start.
- 0:08 The plan appears with boxes on the top and side streams.
- 0:18 The arm grabs. The attempt shows predicted against measured values and the error.
- 0:30 [TODO team: staged miss] The grab misses; the next attempt shows a lesson and a changed plan.
- 0:38 The arm waits at the stage pose until our mouth opens.
- 0:44 It approaches slowly, stops short, and releases after the mouth has been open for a second.
- 0:52 [TODO team: forced third failure] The HELP ping arrives on a phone and opens the dashboard.
