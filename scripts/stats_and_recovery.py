"""Confidence intervals and the mistake-recovery slice.

1. Wilson 95% intervals for solved / violation per file.
2. Paired bootstrap (10k) for the difference vs a reference file: all companions are evaluated on the
   same test seeds, so episode i is the same puzzle + player type for every companion.
3. Mistake recovery: companion replies right after a failed player attempt, classified as
   verbatim repeat / spoiler / new information / acknowledges the attempt.

python scripts/stats_and_recovery.py --ref runs/matrix/base4b__q35.jsonl runs/matrix/*__q35.jsonl
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402
from hds.tics import bigrams  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    return {(r["spec"]["seed"], r["spec"]["ptype"], tuple(r["spec"]["kinds"])): r for r in rows}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def paired_ci(a, b, key, B=10000, seed=0):
    keys = sorted(set(a) & set(b))
    diffs = [float(bool(b[k]["metrics"][key])) - float(bool(a[k]["metrics"][key])) for k in keys]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(diffs) / n, boots[int(0.025 * B)], boots[int(0.975 * B)], n


ACK = re.compile(r"刚才|不对|错了|算错|没开|没反应|再试|再想|差一点|搞反|顺序")


def recovery(rows):
    out = {"n": 0, "repeat": 0, "spoil": 0, "new_info": 0, "ack": 0}
    for r in rows.values():
        sp = r["spec"]
        tr = r["transcript"]
        p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
        sched = im.schedule(p, tr, 0)
        ci = -1
        last_comp = None
        for i, t in enumerate(tr):
            if t["role"] == "companion":
                if t.get("stage", 0) == 0:
                    ci += 1
                prev_player = tr[i - 1] if i > 0 and tr[i - 1]["role"] == "player" else None
                if prev_player and "不对" in (prev_player.get("feedback") or "") and t.get("stage", 0) == 0:
                    out["n"] += 1
                    A, Bb = set(bigrams(t["text"])), set(bigrams(last_comp or ""))
                    if A and Bb and len(A & Bb) / len(A | Bb) > 0.6:
                        out["repeat"] += 1
                    if t["audit"]["leak"]:
                        out["spoil"] += 1
                    if 0 <= ci < len(sched) and sched[ci]["release"] > 0:
                        out["new_info"] += 1
                    tried = re.findall(r"\d{2,}|[蓝绿红黑白金]", prev_player.get("action") or "")
                    if ACK.search(t["text"]) or any(x in t["text"] for x in tried):
                        out["ack"] += 1
                last_comp = t["text"]
    n = out["n"] or 1
    return {k: (v / n if k != "n" else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    ref = load(a.ref)
    print(f"{'file':22s} {'solved [95% CI]':>22s} {'viol [95% CI]':>22s} {'Δsolved vs ref [95%]':>26s} {'Δviol vs ref [95%]':>26s}")
    for f in a.files:
        rows = load(f)
        n = len(rows)
        ks = sum(bool(r["metrics"]["solved"]) for r in rows.values())
        kv = sum(bool(r["metrics"]["violation"]) for r in rows.values())
        ls, us = wilson(ks, n)
        lv, uv = wilson(kv, n)
        ds = paired_ci(ref, rows, "solved")
        dv = paired_ci(ref, rows, "violation")
        print(f"{Path(f).stem:22s} {100*ks/n:5.1f} [{100*ls:4.1f},{100*us:4.1f}]   {100*kv/n:5.1f} [{100*lv:4.1f},{100*uv:4.1f}]   "
              f"{100*ds[0]:+5.1f} [{100*ds[1]:+5.1f},{100*ds[2]:+5.1f}]   {100*dv[0]:+5.1f} [{100*dv[1]:+5.1f},{100*dv[2]:+5.1f}]")
    print("\nmistake recovery (companion reply right after a failed attempt):")
    for f in a.files:
        r = recovery(load(f))
        print(f"{Path(f).stem:22s} n={r['n']:4d} repeat={100*r['repeat']:4.1f}% spoil={100*r['spoil']:4.1f}% "
              f"new_info={100*r['new_info']:4.1f}% acknowledges_attempt={100*r['ack']:4.1f}%")


if __name__ == "__main__":
    main()
