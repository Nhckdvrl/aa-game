"""Concept direction for expression shaping: unit class-mean difference at one layer from a probe dataset, with tau
set from the reference model's negative-class projections. python scripts/make_shape_vec.py SRC.npz REF.npz LAYER OUT"""
import sys
import numpy as np
import torch
src, ref, layer, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
a, b = np.load(src, allow_pickle=True), np.load(ref, allow_pickle=True)
j = list(a["layers"]).index(layer)
H, y = a["H"][:, j].astype(np.float32), a["y"]
v = H[y == 1].mean(0) - H[y == 0].mean(0)
v /= np.linalg.norm(v)
Hr, yr = b["H"][:, list(b["layers"]).index(layer)].astype(np.float32), b["y"]
pr = Hr @ v; ps = H @ v
tau = float(pr[yr == 0].mean())
print(f"layer {layer}: src proj pos {ps[y==1].mean():.2f} neg {ps[y==0].mean():.2f} | ref proj pos {pr[yr==1].mean():.2f} neg {pr[yr==0].mean():.2f} (sd {pr[yr==0].std():.2f}) -> tau {tau:.2f}")
torch.save({"v": torch.tensor(v), "tau": tau, "layer": layer, "src": src, "ref": ref}, out)
