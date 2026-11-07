import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="feeder_arm",              # must match the calibration id
    max_relative_target=5,        # safety: clamp each command to a small step
)
robot = SO101Follower(cfg)
robot.connect()

obs = robot.get_observation()
pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
print(pos)                        # e.g. {'shoulder_pan.pos': 1.2, ...}

# Wiggle the wrist slowly: small steps, held targets
start = pos["wrist_flex.pos"]
for target in [start + 10, start - 10, start]:
    for _ in range(40):           # ~2 s at 20 Hz; max_relative_target limits each step
        robot.send_action({**pos, "wrist_flex.pos": target})
        time.sleep(0.05)

robot.bus.disable_torque()        # leave it limp
robot.disconnect()
