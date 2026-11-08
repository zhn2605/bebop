#!/usr/bin/env bash
# Flash the ESP32 hub with the prebuilt images in firmware/esp32/build (no Arduino toolchain needed).
#   firmware/flash.sh [PORT]     default: the CP2102 adapter under /dev/serial/by-id
# Rebuild the images after changing esp32.ino: see firmware/README.md.
set -euo pipefail
cd "$(dirname "$0")/esp32/build"
PORT=${1:-$(ls /dev/serial/by-id/*CP2102* 2>/dev/null | head -n 1 || true)}
if [ -z "$PORT" ]; then
  echo "no ESP32 found under /dev/serial/by-id; pass the port, e.g. firmware/flash.sh /dev/ttyUSB0" >&2
  exit 1
fi
uvx esptool --chip esp32 --port "$PORT" --baud 921600 write-flash \
  0x1000 esp32.ino.bootloader.bin \
  0x8000 esp32.ino.partitions.bin \
  0xe000 boot_app0.bin \
  0x10000 esp32.ino.bin
