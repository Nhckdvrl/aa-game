"""Dump rate of eval runs: how often a reply hands over the whole remaining answer at once.

A stage-0 reply counts as a dump when it releases >= 99% of the answer information still missing (the same label
the probes in RESULTS section 14 use, `hds.infometrics.schedule`). Prints, per file: episodes, share of episodes with
at least one dump, share of stage-0 replies that dump, solved and violation rates.

python scripts/dump_rate.py runs/val/arm_E_step75.jsonl runs/val/arm_Esh10_step50.jsonl ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402


def dumps(row):
    sp = row["spec"]
    sch = im.schedule(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), row["transcript"], 0)
    h0 = (sch[0]["h0"] if sch else 1.0) or 1.0
    return [x["release"] / h0 >= 0.99 for x in sch]


def main():
    print(f"{'file':40s} {'n':>4s} {'ep_dump':>8s} {'turn_dump':>9s} {'solved':>7s} {'viol':>6s}")
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
        ep, turns, nturn, solved, viol = 0, 0, 0, 0, 0
        for r in rows:
            d = dumps(r)
            ep += any(d)
            turns += sum(d)
            nturn += len(d)
            solved += bool(r["metrics"].get("solved"))
            viol += any(any(t["audit"].values()) for t in r["transcript"] if t["role"] == "companion")
        n = max(len(rows), 1)
        print(f"{os.path.basename(path)[:40]:40s} {len(rows):4d} {100 * ep / n:7.1f}% {100 * turns / max(nturn, 1):8.1f}% "
              f"{100 * solved / n:6.1f}% {100 * viol / n:5.1f}%")


if __name__ == "__main__":
    main()
