"""LoRe-style per-player-type weights, fitted from implicit signals in the interaction logs.

Basis rewards per episode (rule-computed):
  eff  = progress with the turn-efficiency bonus (hds.rewards.progress)
  scaf = 1 - max_unprompted_release (how much answer information she held back unless asked)
Implicit satisfaction per episode (from the player's own lines, scripts/implicit_signals.py):
  sat  = -(impatience rate + spoiler-complaint rate + quit)
For each player type we fit sat ≈ a + w_eff*eff + w_scaf*scaf with w >= 0 and report the weights
normalized to the simplex (LoRe's constraint). Unlike LoRe the basis is fixed and tiny (2 rule metrics)
and the 'users' are simulated player types; the point is whether the logs alone separate the types.

python scripts/fit_type_weights.py runs/matrix/*__q35.jsonl
"""
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402
from hds.rewards import progress  # noqa: E402
from scripts.implicit_signals import IMPATIENT, SPOIL  # noqa: E402


def episodes(files):
    out = []
    for f in files:
        for l in open(f):
            if '"metrics"' not in l:
                continue
            r = json.loads(l)
            sp, m = r["spec"], r["metrics"]
            p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
            sched = im.schedule(p, r["transcript"], 0)
            h0 = (sched[0]["h0"] if sched else 1.0) or 1.0
            unprompted = [x["release"] / h0 for j, x in enumerate(sched) if not (x["stuck"] and j > 0)]
            scaf = 1 - (max(unprompted) if unprompted else 0.0)
            lines = [t["say"] or "" for t in r["transcript"] if t["role"] == "player"]
            n = max(1, len(lines))
            sat = -(sum(bool(IMPATIENT.search(x)) for x in lines) / n + sum(bool(SPOIL.search(x)) for x in lines) / n
                    + float(m["quit"]))
            out.append((sp["ptype"], progress(m), scaf, sat))
    return out


def fit(rows):
    """Least squares with w >= 0 by grid over the simplex direction + closed-form scale and intercept."""
    best = None
    ys = [r[3] for r in rows]
    my = sum(ys) / len(ys)
    for t in [i / 100 for i in range(101)]:
        x = [t * r[1] + (1 - t) * r[2] for r in rows]
        mx = sum(x) / len(x)
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, ys))
        k = max(0.0, sxy / sxx) if sxx else 0.0
        sse = sum((yi - my - k * (xi - mx)) ** 2 for xi, yi in zip(x, ys))
        if best is None or sse < best[0]:
            best = (sse, t, k)
    sst = sum((yi - my) ** 2 for yi in ys)
    return {"w_eff": best[1], "w_scaf": 1 - best[1], "scale": best[2], "r2": 1 - best[0] / sst if sst else 0.0,
            "n": len(rows)}


if __name__ == "__main__":
    eps = episodes(sys.argv[1:])
    for ptype in ["急躁型", "卡关型", "闲聊型", "探索型", "沉默型", "捣乱型"]:
        rows = [e for e in eps if e[0] == ptype]
        if len(rows) < 20:
            continue
        r = fit(rows)
        print(f"{ptype}: w_eff={r['w_eff']:.2f} w_scaf={r['w_scaf']:.2f} scale={r['scale']:.2f} R2={r['r2']:.3f} n={r['n']}")
