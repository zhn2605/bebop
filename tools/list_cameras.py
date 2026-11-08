"""Open every /dev/video* node, save one frame each to data/cameras/, and print what it is.

    python -m tools.list_cameras

Then set the `source` of top, side and face in config/cameras.yaml (the index or a by-id path).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from perception.cameras import open_capture

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cameras"


def main() -> None:
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    by_id: dict[str, list[str]] = {}
    for link in sorted(glob.glob("/dev/v4l/by-id/*")):
        by_id.setdefault(os.path.realpath(link), []).append(link)
    nodes = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
    if not nodes:
        print("no /dev/video* devices")
    for node in nodes:
        n = int(re.sub(r"\D", "", node))
        try:
            name = Path(f"/sys/class/video4linux/video{n}/name").read_text().strip()
        except OSError:
            name = "?"
        ids = " ".join(by_id.get(node, []))
        cap = open_capture(n)
        if cap is None:
            print(f"{node:13} {name:32} no frame (metadata node or busy)  {ids}")
            continue
        ok, frame = cap.read()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not ok:
            print(f"{node:13} {name:32} opened, read failed  {ids}")
            continue
        path = OUT / f"video{n}.jpg"
        cv2.imwrite(str(path), frame)
        h, w = frame.shape[:2]
        rel = path.relative_to(ROOT)
        print(f"{node:13} {name:32} {w}x{h} {fourcc} {fps:.0f}fps -> {rel}  {ids}")
    print("Set top/side/face `source` in config/cameras.yaml (index, or the by-id path).")


if __name__ == "__main__":
    main()
