"""Refit the shaping direction on the policy's own recent episodes (a "refit monitor", level L2).

Same rule as the original direction: unit(mean of dump contexts - mean of the other contexts) at one layer, threshold
= mean projection of the other contexts. Input: dump-concept hidden states from scripts/probe_data.py run on the
current checkpoint's own val episodes. Prints the held-out AUROC of the new direction and of the previous one.

python scripts/refit_shape_vec.py runs/probe/dump_Eshr_25.npz 19 OUT.pt [PREV.pt]
"""
import sys
import zlib

import numpy as np
import torch


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = np.empty(len(s)); r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    npz, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = np.load(npz, allow_pickle=True)
    H = z["H"][:, list(z["layers"]).index(layer)].astype(np.float32)
    y = z["y"].astype(bool)
    if y.sum() < 10 or (~y).sum() < 10:
        sys.exit(f"too few examples to fit (dump {y.sum()}, other {(~y).sum()}); nothing written")
    tr = np.array([(zlib.crc32(str(g).encode()) % 1000) < 700 for g in z["group"]])
    v = H[y & tr].mean(0) - H[~y & tr].mean(0)
    v /= np.linalg.norm(v)
    p = H @ v
    tau = float(p[~y].mean())
    msg = f"{npz}: n={len(y)} dumps={y.sum()} new direction held-out AUROC {auroc(p[y & ~tr], p[~y & ~tr]):.3f} tau {tau:.2f}"
    if len(sys.argv) > 4:
        old = torch.load(sys.argv[4])["v"].float().numpy()
        po = H @ old
        msg += f" | previous direction AUROC {auroc(po[y & ~tr], po[~y & ~tr]):.3f}, cosine {float(old @ v):.2f}"
    print(msg, flush=True)
    torch.save({"v": torch.tensor(v), "tau": tau, "layer": layer, "src": npz}, out)


if __name__ == "__main__":
    main()
