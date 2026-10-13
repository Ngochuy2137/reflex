"""Summarise per-throw evaluation JSONs with the paper's metrics.

Counts the first observed throw of every environment.  Catch success uses the paper's rule:
  - regular / small / large box: sustained hold -- the box rests against the torso (twin criterion),
    is never dropped, the robot does not fall, and the box is slow (< 1 m/s) at the end of the episode;
  - every other condition: the chest criterion (box at the chest, wrists on both sides, not on the
    head or shoulders, slow for 0.6 s).
Interception: the box slows below 1 m/s near the torso, whether or not it is kept.

usage: python -m reflex.eval.summarize outputs/eval/visual_regular_s8.json [more.json ...] [--condition regular]
"""
import argparse
import json
import math

SUSTAINED = {"regular", "small", "large"}


def sustained_hold(t):
    return bool(t["caught_twin"] and not t["dropped_after"] and not t["fell"]
                and t.get("end_v") is not None and t["end_v"] < 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--condition", default=None, help="condition name (default: parsed from the file name)")
    a = p.parse_args()
    cond = a.condition or a.files[0].split("/")[-1].split("_", 1)[1].rsplit("_s", 1)[0]
    caught = icpt = fell = n = 0
    for f in a.files:
        throws = [t for t in json.load(open(f))["throw_log"] if t.get("idx", 1) == 1]
        n += len(throws)
        caught += sum(sustained_hold(t) if cond in SUSTAINED else bool(t["caught"]) for t in throws)
        icpt += sum(bool(t.get("intercepted")) for t in throws)
        fell += sum(bool(t["fell"]) for t in throws)
    rate = caught / max(n, 1)
    ci = 1.96 * math.sqrt(rate * (1 - rate) / max(n, 1))
    rule = "sustained hold" if cond in SUSTAINED else "chest criterion"
    print(f"[summary] condition={cond} throws={n} | catch success ({rule}) {100 * rate:.1f}% +/- {100 * ci:.1f} (95% CI)"
          f" | interception {100 * icpt / max(n, 1):.1f}% | falls {fell}")


if __name__ == "__main__":
    main()
