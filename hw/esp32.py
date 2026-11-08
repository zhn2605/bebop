"""ESP32 hub over USB serial (CORE.md protocol): JSON telemetry in, `STATE <NAME>` out.

Telemetry every 50 ms: {"t", "estop", "dist_cm", "contact"}; {"event": "ESTOP"} on the button.
Either e-stop form, or 500 ms of silence (S7), trips `stop`. The current state is re-sent every
250 ms so the ESP32 can run its own watchdog.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time

from core.types import Flag

STATES = ("READY", "LISTENING", "THINKING", "MOVING", "PRESENTING", "STOPPED", "HELP")


class Esp32:
    def __init__(
        self, port: str, stop: Flag, baud: int = 115200, silence_s: float = 0.5, boot_s: float = 3.0
    ):
        import serial  # pyserial

        self.stop = stop
        self.dist_cm: float | None = None
        self.estop = False
        self.contact = False
        self.state = "READY"
        self.last_rx = 0.0
        self._silence, self._boot = silence_s, boot_s
        self._ser = serial.Serial(port, baud, timeout=0.1)
        self._wlock = threading.Lock()
        self._alive = True
        self._t_open = time.monotonic()
        threading.Thread(target=self._reader, name="esp32-rx", daemon=True).start()
        threading.Thread(target=self._sender, name="esp32-tx", daemon=True).start()

    def _reader(self) -> None:
        while self._alive:
            try:
                line = self._ser.readline()
            except OSError as e:  # unplugged (SerialException is an OSError)
                self.stop.trip(f"esp32 serial error: {e}")
                return
            now = time.monotonic()
            if line:
                try:
                    msg = json.loads(line.decode(errors="replace"))
                except ValueError:
                    msg = None
                if isinstance(msg, dict):
                    self.last_rx = now
                    if msg.get("event") == "ESTOP" or msg.get("estop"):
                        self.estop = True
                        self.stop.trip("esp32 e-stop")
                    elif "estop" in msg:
                        self.estop = False
                    if "dist_cm" in msg:
                        d = msg["dist_cm"]
                        self.dist_cm = float(d) if isinstance(d, (int, float)) else None
                    if "contact" in msg:
                        self.contact = bool(msg["contact"])
            quiet = now - (self.last_rx or self._t_open)
            if quiet > (self._silence if self.last_rx else self._boot):
                self.stop.trip(f"esp32 silent for {quiet * 1000:.0f} ms")

    def _sender(self) -> None:
        while self._alive:
            self._write(f"STATE {self.state}\n")
            time.sleep(0.25)

    def _write(self, text: str) -> None:
        with self._wlock, contextlib.suppress(OSError):  # the reader trips stop on a dead port
            self._ser.write(text.encode())

    def send_state(self, name: str) -> None:
        self.state = name.upper()
        self._write(f"STATE {self.state}\n")

    def close(self) -> None:
        self._alive = False
        with contextlib.suppress(OSError):
            self._ser.close()


def open_esp32(port: str | None, stop: Flag) -> Esp32 | None:
    """`Esp32(port, stop)`, or None when no port is given. A port that fails to open raises."""
    return Esp32(port, stop) if port else None
