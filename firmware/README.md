# ESP32 hub firmware

`esp32/esp32.ino`: one Arduino sketch for the ESP-WROOM-32 dev board. E-stop button, HC-SR04
distance, optional contact inputs, SSD1306 OLED and three LEDs, talking JSON lines to
`hw/esp32.py` over USB serial at 115200. Pin map and protocol: `.config/docs/CORE.md`
("ESP32 hub pin map", "ESP32 ↔ laptop serial protocol").

## Flash it (Arch Linux)

Either the CLI (`sudo pacman -S arduino-cli`) or Arduino IDE 2 (AUR `arduino-ide-bin`, or the
AppImage from arduino.cc). Both need the esp32 core and two libraries:

- Board manager URL: `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
  (IDE: File → Preferences → Additional boards manager URLs; then Boards Manager → "esp32" by
  Espressif)
- Libraries: **Adafruit SSD1306** and **Adafruit GFX Library** (IDE: Library Manager; accept
  "install all" for Adafruit BusIO)
- Board: **ESP32 Dev Module** (`esp32:esp32:esp32`)

Quickest: flash the prebuilt images in `firmware/esp32/build` (only needs `uv`; esptool comes
through `uvx`):

```sh
firmware/flash.sh                 # finds the CP2102 under /dev/serial/by-id
firmware/flash.sh /dev/ttyUSB0    # or name the port
```

After changing `esp32.ino`, rebuild with arduino-cli (below) and copy the four images
(`esp32.ino.bootloader.bin`, `esp32.ino.partitions.bin`, `boot_app0.bin`, `esp32.ino.bin`) from
`--build-path` into `firmware/esp32/build`.


```sh
arduino-cli config init
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index && arduino-cli core install esp32:esp32
arduino-cli lib install "Adafruit SSD1306" "Adafruit GFX Library"

arduino-cli compile --fqbn esp32:esp32:esp32 firmware/esp32
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/esp32
```

The port is usually `/dev/ttyUSB0` (CP2102 or CH340 USB-serial chip; `ls /dev/ttyUSB*` before and
after plugging in). It is `root:uucp`: `sudo usermod -aG uucp $USER` and log out and in, or for one
session `sudo chmod a+rw /dev/ttyUSB0`. If the upload hangs at `Connecting....`, hold the board's
BOOT button until it starts writing. Close any serial monitor first; only one program can hold the
port.

## Check it

```sh
uv run esp32-check --port /dev/ttyUSB0
```

Prints telemetry for 5 s while the OLED/LEDs go THINKING (yellow) → READY (green), reports the line
rate and longest gap, then asks you to press the button and checks for `{"event": "ESTOP"}`, the
latched `"estop": 1`, and that it clears after `STATE STOPPED` once the button is up. Then run the
robot with `uv run bebop ... --esp32 /dev/ttyUSB0`.

## Behaviour

| Shown | Green GPIO25 | Yellow GPIO26 | Red GPIO33 |
|---|---|---|---|
| READY | on | | |
| LISTENING | slow blink | | |
| THINKING | | on | |
| MOVING, SCOOPING, DELIVERING | | blink | |
| PRESENTING, OPEN_WIDE | on | on | |
| STOPPED | | | on |
| HELP | | | fast blink |
| E-STOP (button latched) | | | on |
| NO LINK (no STATE for 2 s) | | | slow blink |
| BOOT (until the first STATE) | on | on | on |

CORE.md's LEDs are green, yellow and red (no blue), matching `LED` in `core/run.py`. Unknown state
names are shown as text with all LEDs off; lines that aren't `STATE <NAME>` are ignored.

- **E-stop**: a press (debounced 30 ms) sends `{"event": "ESTOP"}` once and latches `"estop": 1`.
  The latch clears when the laptop has sent `STATE STOPPED` since the press and the button is up.
  Not on `STATE READY`: `hw/esp32.py` trips stop on every `"estop": 1`, so the laptop could never
  get back to READY.
- **dist_cm**: median of the last 3 pings (one every 60 ms, 25 ms echo timeout); `null` when 2 of
  the 3 had no echo.
- **contact**: 1 for 200 ms after the GPIO32 switch closes or the GPIO34 piezo reads above
  `PIEZO_THRESHOLD`. Without the disc and its 1 MΩ wired, set `PIEZO_ENABLED` to 0 (a floating
  GPIO34 reads noise).
- **OLED**: probed at 0x3C and 0x3D at boot; when it isn't there the sketch runs without it.
