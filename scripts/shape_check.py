"""Did shaping remove the dump decision, or only its projection on the penalized direction?

For each model's dump-concept hidden states (scripts/probe_data.py), project the layer-L contexts on the shaping
direction and report the mean for dump and non-dump turns, the share above the penalty threshold, and the AUROC of
that single projection. Pair with scripts/probe_eval.py (L1 reuse of the reference probe, L2 refit).

python scripts/shape_check.py runs/probe/shape_dump_L19.pt runs/probe/dump_E75.npz:E75 runs/probe/dump_Eshg5_50.npz:Eshg5_50 ...
"""
import sys

import numpy as np
import torch


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    spec = torch.load(sys.argv[1])
    v, tau, L = spec["v"].float().numpy(), float(spec["tau"]), int(spec["layer"])
    print(f"direction {sys.argv[1]} layer {L} tau {tau:.2f}")
    print(f"{'model':12s} {'n':>5s} {'dump':>5s} {'proj dump':>10s} {'proj other':>11s} {'>tau dump':>10s} {'>tau other':>11s} {'AUROC':>6s}")
    for arg in sys.argv[2:]:
        path, name = arg.rsplit(":", 1)
        z = np.load(path, allow_pickle=True)
        j = list(z["layers"]).index(L)
        p = z["H"][:, j].astype(np.float32) @ v
        y = z["y"].astype(bool)
        print(f"{name:12s} {len(y):5d} {y.sum():5d} {p[y].mean():10.2f} {p[~y].mean():11.2f} "
              f"{(p[y] > tau).mean():10.0%} {(p[~y] > tau).mean():11.0%} {auroc(p[y], p[~y]):6.2f}")


if __name__ == "__main__":
    main()
