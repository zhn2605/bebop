"""Predictions against outcomes over every run: the learning loop against the no-learning baseline.

    uv run report [--food NAME] [--fake] [--csv PATH]

Reads the `outcome` events in data/runs/*/events.jsonl (one per acquire try: food, policy, task,
try number, the planner's success_prob, ok). An episode is one task's tries up to its first
success. Per food and over all foods, per policy:
  got it by try k   episodes whose first success came at try k or earlier
  tries to success  mean over the episodes that succeeded; "never" counts the rest
  Brier             mean (success_prob - outcome)^2, lower is better (0.25 = always saying 50 %);
                    per try number too, to see whether the predictions get better within a task
  predicted/actual  mean success_prob against the success rate (calibration)
Baseline runs: `uv run bebop --no-learning ...`. Fake runs are left out unless --fake.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "data" / "runs"
POLICIES = (("learning", True), ("baseline", False))
TRIES = 4  # rows of "got it by try k"


def load(runs: Path = RUNS, fake: bool = False) -> list[dict]:
    rows = []
    for path in sorted(runs.glob("*/events.jsonl")):
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "outcome" and (fake or not e.get("fake")):
                rows.append(e)
    return rows


def stats(rows: list[dict]) -> dict:
    """The numbers of one policy's rows (see the module doc)."""
    episodes: dict[str, list[dict]] = {}
    for r in rows:
        episodes.setdefault(r["task"], []).append(r)
    first = []  # try number of each episode's first success, None = never
    for tries in episodes.values():
        ok = [r["try_n"] for r in sorted(tries, key=lambda r: r["try_n"]) if r["ok"]]
        first.append(ok[0] if ok else None)
    done = [k for k in first if k is not None]

    def brier(rs):
        rs = [r for r in rs if r.get("p") is not None]
        return sum((r["p"] - r["ok"]) ** 2 for r in rs) / len(rs) if rs else None

    by_try = {k: brier([r for r in rows if min(r["try_n"], 3) == k]) for k in (1, 2, 3)}
    ps = [r["p"] for r in rows if r.get("p") is not None]
    return {
        "episodes": len(first),
        "tries": len(rows),
        "by_try": [sum(1 for k in done if k <= n) / len(first) for n in range(1, TRIES + 1)],
        "mean_first": sum(done) / len(done) if done else None,
        "never": first.count(None),
        "brier": brier(rows),
        "brier_try": by_try,
        "predicted": sum(ps) / len(ps) if ps else None,
        "actual": sum(r["ok"] for r in rows) / len(rows),
    }


def table(title: str, rows: list[dict]) -> str:
    cols = {name: stats(rs) for name, flag in POLICIES
            if (rs := [r for r in rows if bool(r.get("learning", True)) == flag])}
    if not cols:
        return ""
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"
    num = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    lines = [(f"{title}", list(cols))]
    lines.append(("episodes (tries)", [f"{s['episodes']} ({s['tries']})" for s in cols.values()]))
    for k in range(TRIES):
        lines.append((f"got it by try {k + 1}", [pct(s["by_try"][k]) for s in cols.values()]))
    lines.append(("tries to success", [num(s["mean_first"], 1) for s in cols.values()]))
    lines.append(("never got it", [str(s["never"]) for s in cols.values()]))
    lines.append(("Brier (lower = better)", [num(s["brier"]) for s in cols.values()]))
    for k, label in ((1, "try 1"), (2, "try 2"), (3, "try 3+")):
        lines.append((f"  {label}", [num(s["brier_try"][k]) for s in cols.values()]))
    lines.append(("predicted / actual", [f"{pct(s['predicted'])} / {pct(s['actual'])}"
                                         for s in cols.values()]))
    w = max(len(a) for a, _ in lines) + 2
    return "\n".join(f"{a:<{w}}" + "".join(f"{v:>16}" for v in vals) for a, vals in lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--food", help="only this food (e.g. 'donut hole')")
    ap.add_argument("--fake", action="store_true", help="include --fake runs")
    ap.add_argument("--csv", type=Path, help="also write every try as a CSV row here")
    args = ap.parse_args()
    from brain.lessons import key

    rows = load(fake=args.fake)
    if args.food:
        rows = [r for r in rows if r["food"] == key(args.food)]
    if not rows:
        print("no outcomes yet: run `uv run bebop` (and `--no-learning` for the baseline)")
        return
    foods = sorted({r["food"] for r in rows})
    blocks = [table(f, [r for r in rows if r["food"] == f]) for f in foods]
    if len(foods) > 1:
        blocks.append(table("all foods", rows))
    print("\n\n".join(b for b in blocks if b))
    if args.csv:
        fields = ["task", "food", "learning", "try_n", "p", "ok", "err_mm", "escalated", "fake"]
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} tries -> {args.csv}")


if __name__ == "__main__":
    main()
