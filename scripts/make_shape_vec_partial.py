"""Shaping direction "whole answer at once" vs "a partial hint", from one model's own hidden states.

The first dump direction (make_shape_vec.py) contrasts dump turns with every other turn of the hacked model; those
other turns are mostly chit-chat after the dump, so the direction also means "about to give information", and
penalizing it produced a withholder. Here the negatives are only turns that released part of the answer
(PART_LO <= release / H0 < PART_HI), so the direction separates all-at-once from some.

Release fractions are recomputed from the probe's source runs in the same order probe_data.py used, and the dump
labels are checked against the stored ones before anything is written.

python scripts/make_shape_vec_partial.py runs/probe/dump_base4b.npz LAYER OUT.pt RUNS.jsonl [RUNS.jsonl ...] [--n 240]
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402

PART_LO, PART_HI = 0.05, 0.9


def fractions(row):
    sp = row["spec"]
    sch = im.schedule(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), row["transcript"], 0)
    h0 = (sch[0]["h0"] if sch else 1.0) or 1.0
    return [x["release"] / h0 for x in sch]


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("layer", type=int); ap.add_argument("out"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--min-auroc", type=float, default=0.0, help="write nothing if held-out dump-vs-partial AUROC is lower")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    layers = list(z["layers"]); j = layers.index(a.layer)
    H, y, g = z["H"][:, j].astype(np.float32), z["y"].astype(int), z["group"]
    # rebuild per-item release fractions in probe_data order: per file, per row, per stage-0 turn with a schedule entry
    frac = {}
    for path in a.runs:
        rows = [json.loads(l) for l in open(path) if '"metrics"' in l][: a.n]
        for gi, row in enumerate(rows):
            frac[f"{os.path.basename(path)}:{gi}"] = fractions(row)
    seen, f = {}, np.zeros(len(y))
    for i, grp in enumerate(g):
        path, gi = str(grp).rsplit(":", 1)
        key = f"{os.path.basename(path)}:{gi}"
        k = seen.get(key, 0); seen[key] = k + 1
        f[i] = frac[key][k]
    assert np.array_equal((f >= 0.99).astype(int), y), "release fractions do not line up with stored dump labels"
    dump, part = f >= 0.99, (f >= PART_LO) & (f < PART_HI)
    if dump.sum() < 10 or part.sum() < 10:
        sys.exit(f"too few examples to fit (dump {dump.sum()}, partial {part.sum()}); nothing written")
    tr = np.array([(zlib.crc32(str(x).encode()) % 1000) < 700 for x in g])
    v = H[dump & tr].mean(0) - H[part & tr].mean(0)
    v = v / np.linalg.norm(v)
    p = H @ v
    print(f"layer {a.layer}: dump {dump.sum()} partial {part.sum()} other {(~dump & ~part).sum()} (n={len(y)})")
    print(f"  projection mean  dump {p[dump].mean():.2f}  partial {p[part].mean():.2f} (sd {p[part].std():.2f})  "
          f"no-info {p[~dump & ~part].mean():.2f}")
    print(f"  held-out AUROC  dump vs partial {auroc(p[dump & ~tr], p[part & ~tr]):.3f}  "
          f"dump vs no-info {auroc(p[dump & ~tr], p[~dump & ~part & ~tr]):.3f}")
    auc = auroc(p[dump & ~tr], p[part & ~tr])
    if auc < a.min_auroc:
        sys.exit(f"held-out AUROC {auc:.3f} below {a.min_auroc}; nothing written")
    tau = float(p[part].mean() + p[part].std())
    torch.save({"v": torch.tensor(v), "tau": tau, "layer": a.layer, "src": a.npz, "neg": "partial hints"}, a.out)
    print(f"  tau = partial mean + 1 sd = {tau:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
