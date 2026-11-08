"""Does the companion listen when the player asks it not to spoil?

A "request" is a player line matching the spoiler-complaint pattern (别剧透 / 让我自己想 / 别告诉我 ...).
For the companion reply right after it (stage 0, exact fact accounting from hds/infometrics.py):
  new_info : the reply releases any new answer information
  big      : it releases >= half of the puzzle's answer entropy in one go
  leak     : it states the full answer (rule audit)
python scripts/request_respect.py runs/matrix runs/matrix_extra
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402
from scripts.implicit_signals import SPOIL  # noqa: E402


def moments(rows):
    out = {"episodes": len(rows), "requests": 0, "new_info": 0, "big": 0, "leak": 0}
    for r in rows:
        sp, tr = r["spec"], r["transcript"]
        p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
        sched = im.schedule(p, tr, 0)
        ci = -1
        for i, t in enumerate(tr):
            if t["role"] != "companion" or t.get("stage", 0) != 0:
                continue
            ci += 1
            prev = tr[i - 1] if i > 0 and tr[i - 1]["role"] == "player" else None
            if not prev or not SPOIL.search(prev.get("say") or ""):
                continue
            out["requests"] += 1
            rel = sched[ci]["release"] if ci < len(sched) else 0.0
            out["new_info"] += rel > 0
            out["big"] += rel >= 0.5 * (sched[ci]["h0"] or 1.0) if ci < len(sched) else 0
            out["leak"] += bool(t["audit"]["leak"])
    return out


if __name__ == "__main__":
    agg = defaultdict(lambda: defaultdict(int))
    for d in sys.argv[1:]:
        for f in glob.glob(f"{d}/*__*.jsonl"):
            comp, player = os.path.basename(f)[:-6].split("__")
            rows = [json.loads(l) for l in open(f) if '"metrics"' in l]
            for k, v in moments(rows).items():
                agg[comp][k] += v
    print(f"{'companion':12s} {'episodes':>8s} {'requests':>8s} {'new info':>9s} {'≥half':>7s} {'leak':>6s}")
    for comp in sorted(agg):
        a = agg[comp]
        n = max(1, a["requests"])
        print(f"{comp:12s} {a['episodes']:8d} {a['requests']:8d} {100*a['new_info']/n:8.0f}% {100*a['big']/n:6.0f}% {100*a['leak']/n:5.0f}%")
