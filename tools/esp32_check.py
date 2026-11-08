"""ESP32 hub check: prints telemetry, drives the OLED/LEDs, and tests the e-stop button and its reset.

    python -m tools.esp32_check [--port /dev/ttyUSB0] [--seconds 5] [--press-s 15]

1. Prints every line for --seconds while sending STATE THINKING (yellow steady), then STATE READY
   (green steady), re-sent every 250 ms like hw/esp32.py. Reports the line rate and the longest gap
   (the laptop treats 500 ms of silence as an e-stop).
2. Asks for a button press and reports whether {"event": "ESTOP"} and "estop": 1 arrive.
3. Sends STATE STOPPED (as core.run does) and checks "estop" drops back to 0 once the button is up.

Raw pyserial, not hw/esp32.Esp32: that class consumes the lines on its own thread and folds the
event into `estop`, so it can't show either.
"""

from __future__ import annotations

import argparse
import json
import time

import serial  # pyserial

RESEND_S = 0.25


class Link:
    def __init__(self, port: str):
        self.ser = serial.Serial(port, 115200, timeout=0.05)
        self.state = "READY"
        self.t_sent = 0.0
        self.t_last = 0.0
        self.max_gap = 0.0

    def poll(self) -> dict | None:
        """One line (or None after 50 ms), and the current STATE re-sent when due."""
        now = time.monotonic()
        if now - self.t_sent >= RESEND_S:
            self.send(self.state)
        raw = self.ser.readline().decode(errors="replace").strip()
        if not raw:
            return None
        try:
            msg = json.loads(raw)
        except ValueError:
            print(f"  (not JSON) {raw}")
            return None
        if not isinstance(msg, dict):
            return None
        now = time.monotonic()
        if self.t_last:
            self.max_gap = max(self.max_gap, now - self.t_last)
        self.t_last = now
        return msg

    def send(self, name: str) -> None:
        self.state = name
        self.ser.write(f"STATE {name}\n".encode())
        self.t_sent = time.monotonic()

    def wait(self, seconds: float, until) -> dict | None:
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            msg = self.poll()
            if msg is not None and until(msg):
                return msg
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", default="auto", help="serial port, or auto to find the CP2102 by name")
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--press-s", type=float, default=15, help="how long to wait for the button")
    args = ap.parse_args()
    from hw.esp32 import find_port

    args.port = find_port(args.port)
    link = Link(args.port)

    # Opening the port resets most dev boards (DTR/RTS auto-reset): wait for the first telemetry.
    if link.wait(4, lambda m: "t" in m) is None:
        print(f"no telemetry from {args.port} in 4 s: wrong port, not flashed, or wrong baud")
        return
    link.max_gap = 0.0

    print(f"-- telemetry for {args.seconds:g} s: THINKING (yellow), then READY (green)")
    link.send("THINKING")
    t0 = time.monotonic()
    n, dists = 0, []
    while (t := time.monotonic() - t0) < args.seconds:
        if t >= args.seconds / 2 and link.state != "READY":
            link.send("READY")
        msg = link.poll()
        if msg is None:
            continue
        print(f"  {json.dumps(msg)}")
        if "t" in msg:
            n += 1
            if isinstance(msg.get("dist_cm"), (int, float)):
                dists.append(float(msg["dist_cm"]))
    rate = n / args.seconds
    span = f"{min(dists):.1f}-{max(dists):.1f} cm" if dists else "always null (no echo)"
    gap_ok = "OK" if link.max_gap < 0.5 else "TOO LONG, the laptop would e-stop"
    gap_ms = link.max_gap * 1000
    print(f"{n} lines, {rate:.1f}/s (want ~20); longest gap {gap_ms:.0f} ms ({gap_ok})")
    print(f"dist_cm: {span}")

    print(f"-- press the e-stop button (waiting {args.press_s:g} s)")
    t_press = time.monotonic()
    ev = link.wait(args.press_s, lambda m: m.get("event") == "ESTOP")
    if ev is None:
        print("ESTOP event: NO (check GPIO27 to GND)")
        return
    print(f"ESTOP event: yes, {(time.monotonic() - t_press):.1f} s after the prompt")
    latched = link.wait(1, lambda m: "estop" in m)
    print(f'telemetry "estop": {latched.get("estop") if latched else "no telemetry"} (want 1)')

    print("-- sending STATE STOPPED; release the button")
    link.send("STOPPED")
    cleared = link.wait(10, lambda m: m.get("estop") == 0)
    print(f'"estop" back to 0 after STOPPED: {"yes" if cleared else "NO"}')
    link.send("READY")
    link.wait(0.3, lambda m: False)
    link.ser.close()


if __name__ == "__main__":
    main()
