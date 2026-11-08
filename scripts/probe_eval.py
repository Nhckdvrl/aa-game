"""Linear probes across checkpoints: does a concept stay decodable after post-training, and does the old probe still work?

Following the monitorability hierarchy of Goodfire's "Shaping and Monitoring Models Through Their Internal
Representations" (2026): L1 = reuse the probe fit on a reference model; L2 = refit a linear probe on the new model.
Per layer: logistic regression on standardized hidden states, split 70/30 by episode, balanced accuracy (and AUROC
for binary concepts); also the class-separation ratio (between-class / within-class variance along the probe axis).

python scripts/probe_eval.py runs/probe/persona_base4b.npz:base4b runs/probe/persona_F2w150.npz:F2w150 ... [--ref base4b]
"""
import argparse
import zlib

import numpy as np
import torch


def split(groups, frac=0.7):
    tr = np.array([(zlib.crc32(str(g).encode()) % 1000) < frac * 1000 for g in groups])
    return tr, ~tr


def fit(X, y, k, steps=300, wd=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    w_cls = torch.tensor(len(y) / (k * np.bincount(y, minlength=k).clip(1)), dtype=torch.float32)  # balance classes
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt, weight=w_cls) + wd * (W ** 2).sum()
        loss.backward()
        opt.step()
    return mu, sd, W.detach().numpy(), b.detach().numpy()


def predict(probe, X):
    mu, sd, W, b = probe
    return ((X - mu) / sd) @ W + b


def bal_acc(y, logits):
    p = logits.argmax(1)
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separation(X, y):
    """Fisher-style ratio along the top discriminant directions: mean between-class over within-class variance."""
    mu = X.mean(0)
    cls = np.unique(y)
    between = sum((y == c).sum() * np.sum((X[y == c].mean(0) - mu) ** 2) for c in cls) / len(y)
    within = sum(np.sum((X[y == c] - X[y == c].mean(0)) ** 2) for c in cls) / len(y)
    return float(between / within)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default=None, help="model name whose probe is reused (L1)")
    a = ap.parse_args()
    data = {}
    for f in a.files:
        path, name = f.rsplit(":", 1)
        z = np.load(path, allow_pickle=True)
        data[name] = (z["H"].astype(np.float32), z["y"].astype(int), z["group"], list(z["layers"]))
    ref = a.ref or list(data)[0]
    k = int(max(d[1].max() for d in data.values())) + 1
    layers = data[ref][3]
    print(f"concept classes={k}; reference model for L1 = {ref}")
    for name, (H, y, g, _) in data.items():
        print(f"  {name}: n={len(y)} class shares {np.round(np.bincount(y, minlength=k) / len(y), 3).tolist()}")
    hdr = "layer " + " ".join(f"{n:>22s}" for n in data)
    print("\nL2 (refit on each model): balanced accuracy" + (" / AUROC" if k == 2 else "") + " / separation ratio")
    print(hdr)
    probes = {}
    for j, L in enumerate(layers):
        row = []
        for name, (H, y, g, _) in data.items():
            tr, te = split(g)
            p = fit(H[tr, j], y[tr], k)
            probes[(name, L)] = p
            lg = predict(p, H[te, j])
            extra = f"/{auroc(y[te], lg[:, 1] - lg[:, 0]):.2f}" if k == 2 else ""
            row.append(f"{bal_acc(y[te], lg):.2f}{extra}/{separation(H[:, j] / (H[:, j].std() + 1e-6), y):.3f}")
        print(f"{L:5d} " + " ".join(f"{c:>22s}" for c in row))
    print(f"\nL1 (reuse the {ref} probe on each model's held-out split): balanced accuracy" + (" / AUROC" if k == 2 else ""))
    print(hdr)
    for j, L in enumerate(layers):
        row = []
        for name, (H, y, g, _) in data.items():
            tr, te = split(g)
            lg = predict(probes[(ref, L)], H[te, j])
            extra = f"/{auroc(y[te], lg[:, 1] - lg[:, 0]):.2f}" if k == 2 else ""
            row.append(f"{bal_acc(y[te], lg):.2f}{extra}")
        print(f"{L:5d} " + " ".join(f"{c:>22s}" for c in row))
    print("\nCosine between each model's refit probe and the reference probe (class-mean-difference directions, layer by layer)")
    print(hdr)
    for j, L in enumerate(layers):
        Wr = probes[(ref, L)][2] / probes[(ref, L)][1][:, None]
        row = []
        for name in data:
            W = probes[(name, L)][2] / probes[(name, L)][1][:, None]
            a1, b1 = Wr - Wr.mean(1, keepdims=True), W - W.mean(1, keepdims=True)
            cos = float(np.sum(a1 * b1) / (np.linalg.norm(a1) * np.linalg.norm(b1) + 1e-9))
            row.append(f"{cos:.2f}")
        print(f"{L:5d} " + " ".join(f"{c:>22s}" for c in row))


if __name__ == "__main__":
    main()
