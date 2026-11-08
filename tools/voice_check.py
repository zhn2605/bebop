"""Prints the microphone level, Vosk partials, transcribed utterances and stop-word hits.

python -m tools.voice_check [--device N|NAME] [--wav FILE]

--wav feeds a 16 kHz mono 16-bit WAV instead of the microphone. Ctrl+C quits.
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import yaml

from voice.listener import BLOCK, RATE, Listener

CONFIG = Path(__file__).resolve().parents[1] / "config" / "brain.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", help="sounddevice input index or name substring")
    parser.add_argument("--wav", help="16 kHz mono 16-bit WAV to feed instead of the microphone")
    args = parser.parse_args()
    cfg = (yaml.safe_load(CONFIG.read_text()) or {}).get("voice") or {}
    if args.device is not None:
        cfg["device"] = int(args.device) if args.device.isdigit() else args.device
    t0 = time.monotonic()

    def say(kind, text):
        print(f"{time.monotonic() - t0:6.2f}s {kind:9} {text}", flush=True)

    listener = Listener(cfg, on_utterance=lambda text: say("utterance", text),
                        on_stop=lambda word: say("STOP", word),
                        on_partial=lambda text: say("partial", text))
    if args.wav:
        with wave.open(args.wav) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
                raise SystemExit("the WAV must be 16 kHz mono 16-bit: "
                                 "ffmpeg -i IN -ar 16000 -ac 1 -sample_fmt s16 OUT.wav")
            listener.load()
            t0 = time.monotonic()
            while data := w.readframes(BLOCK):
                listener.feed(data)
        listener.drain()
        return
    import sounddevice as sd

    print(f"input device: {sd.query_devices(cfg.get('device'), 'input')['name']}")
    listener.start()
    try:
        while listener.is_alive():
            time.sleep(1)
            say("level", f"{listener.level:.3f} {'#' * min(int(listener.level * 400), 60)}")
    except KeyboardInterrupt:
        pass
    listener.stop()
    if listener.error:
        raise SystemExit(f"listener died: {listener.error}")


if __name__ == "__main__":
    main()
