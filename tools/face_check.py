"""Print FaceState at 5 Hz from the `face` camera, to check the mouth threshold and yaw/pitch signs.

    python -m tools.face_check [--fake] [--seconds N]

--fake runs MediaPipe on the fake world's drawn face.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from perception.cameras import Cameras, FakeWorld
from perception.face import FaceSignals

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = until Ctrl+C)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "cameras.yaml").read_text()) or {}
    cams = Cameras(cfg, fake=FakeWorld(lambda: None) if args.fake else None).start()
    if "face" not in cams.names:
        print("no `face` camera; check config/cameras.yaml with python -m tools.list_cameras")
        return
    fs = FaceSignals(cams, cfg=cfg).start()
    c = fs.cfg
    print(
        f"yaw_sign={c['yaw_sign']} pitch_sign={c['pitch_sign']} yaw_offset_deg={c['yaw_offset_deg']}"
        f" mouth_open_ratio={c['mouth_open_ratio']} away_yaw_deg={c['away_yaw_deg']}\n"
        "Turn your head to your left: yaw should go positive. Look up: pitch should go positive.\n"
        "Otherwise flip yaw_sign / pitch_sign in the face: section of config/cameras.yaml."
    )
    t_end = time.time() + args.seconds if args.seconds else float("inf")
    try:
        while time.time() < t_end:
            s = fs.state
            print(
                f"present={s.present!s:5} mouth_open={s.mouth_open!s:5} ratio={s.mouth_ratio:.2f}"
                f" yaw={s.yaw_deg:+6.1f} pitch={s.pitch_deg:+6.1f} looking_away={s.looking_away}"
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        fs.stop()
        cams.stop()


if __name__ == "__main__":
    main()
