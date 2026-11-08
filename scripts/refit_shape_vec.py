"""Refit the shaping direction on the policy's own recent episodes (a "refit monitor", level L2).

A class-balanced logistic probe (as in scripts/probe_eval.py) at one layer, on dump-concept hidden states from
scripts/probe_data.py run on the current checkpoint's own episodes; its weight vector, mapped back to raw activation
space and normalized, is the new direction. Threshold = mean + 1 sd of the non-dump projections, so ordinary turns
mostly stay below it. The new direction is written only if it is a usable monitor: held-out AUROC >= MIN_AUROC and
not worse than the previous direction on the same contexts; otherwise nothing is written and the caller keeps the
previous one.

python scripts/refit_shape_vec.py runs/probe/dump_Eshr_25.npz 19 OUT.pt [PREV.pt]
"""
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_eval import fit  # noqa: E402

MIN_AUROC = 0.65


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    npz, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = np.load(npz, allow_pickle=True)
    H = z["H"][:, list(z["layers"]).index(layer)].astype(np.float32)
    y = z["y"].astype(int)
    if y.sum() < 10 or (1 - y).sum() < 10:
        sys.exit(f"too few examples to fit (dump {y.sum()}, other {(1 - y).sum()}); nothing written")
    tr = np.array([(zlib.crc32(str(g).encode()) % 1000) < 700 for g in z["group"]])
    mu, sd, W, b = fit(H[tr], y[tr], 2)
    v = (W[:, 1] - W[:, 0]) / sd
    v /= np.linalg.norm(v)
    p = H @ v
    yb = y.astype(bool)
    new_auc = auroc(p[yb & ~tr], p[~yb & ~tr])
    tau = float(p[~yb].mean() + p[~yb].std())
    msg = f"{npz}: n={len(y)} dumps={y.sum()} new direction held-out AUROC {new_auc:.3f} tau {tau:.2f}"
    old_auc = -1.0
    if len(sys.argv) > 4:
        old = torch.load(sys.argv[4])["v"].float().numpy()
        po = H @ old
        old_auc = auroc(po[yb & ~tr], po[~yb & ~tr])
        msg += f" | previous direction AUROC {old_auc:.3f}, cosine {float(old @ v):.2f}"
    print(msg, flush=True)
    if new_auc < MIN_AUROC or new_auc < old_auc - 0.02:
        sys.exit("new direction is not a better monitor; nothing written")
    torch.save({"v": torch.tensor(v), "tau": tau, "layer": layer, "src": npz, "auroc": new_auc}, out)


if __name__ == "__main__":
    main()
