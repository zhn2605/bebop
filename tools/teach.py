"""Teach the joint poses home, stage and mouth by hand, or go to one.

python -m tools.teach [NAME ...]          torque off, move by hand, Enter saves each into poses.yaml
python -m tools.teach --goto NAME         slow joint move to a taught pose (nobody in front of it)
--fake uses the fake arm and prints instead of saving unless --out is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from core.types import JOINTS, Stopped
from dashboard.client import for_tool
from tools.arm_check import ROOT, SAG, fmt, setup

HINT = {
    "home": "tip above the bowl, inside the fence box",
    "stage": "tool level, 15 cm in front of the mouth",
    "mouth": "tool at the lips",
}
HEADER = """# Taught joint poses, LeRobot degrees. Written by `python -m tools.teach`; check one
# with `python -m tools.teach --goto NAME`. home: above the bowl inside the fence box;
# stage: tool level, 15 cm in front of the mouth; mouth: tool at the lips (present stops
# stop_short_mm before it).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="teach or visit taught poses")
    ap.add_argument("names", nargs="*", default=["home", "stage", "mouth"])
    ap.add_argument("--goto", metavar="NAME")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", help="poses file (default config/poses.yaml; none with --fake)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (None if args.fake else ROOT / "config" / "poses.yaml")
    arm, motion, stop = setup(args.fake)
    pub = for_tool("teach", arm, stop, fake=args.fake)
    try:
        if args.goto:
            pub.state.set(action=f"goto {args.goto}")
            if args.goto == "mouth":
                print("Going to the mouth pose: nobody in front of the arm.")
            r = motion.goto(args.goto, "slow")
            print(fmt(r))
            return
        poses = (yaml.safe_load(out.read_text()) or {}) if out and out.is_file() else {}
        input(f"{SAG}\nEnter to turn torque off... ")
        arm.torque(False)
        for name in args.names:
            pub.state.set(action=f"move the arm to {name} by hand")
            ans = input(
                f"Move the arm to '{name}' ({HINT.get(name, 'by hand')}); Enter saves, s skips: "
            )
            if ans.strip().lower() == "s":
                continue
            st = arm.state()
            poses[name] = {j: round(st.q[j], 1) for j in JOINTS}
            print(
                f"{name}: {poses[name]}  tip ({', '.join(f'{v:.0f}' for v in st.tip.xyz)})"
                f" pitch {st.tip.pitch:+.0f}"
            )
            if out:
                out.write_text(
                    HEADER + yaml.safe_dump(poses, default_flow_style=None, sort_keys=False)
                )
                print(f"saved {out}")
        print("Done. Rest the arm; torque stays off.")
    except (KeyboardInterrupt, EOFError, Stopped):
        arm.freeze()
        stop.clear()  # so disconnect() still folds to rest instead of dropping the arm
        print("\nfrozen; folding to rest")
    finally:
        arm.disconnect()  # folds to rest_pose slowly when torque is on, then torque off
        pub.stop()


if __name__ == "__main__":
    main()
