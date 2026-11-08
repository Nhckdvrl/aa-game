"""Average two checkpoints of the same architecture shard by shard (bf16 out): python scripts/soup.py A B OUT [w]"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
a, b, out = sys.argv[1:4]; w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
os.makedirs(out, exist_ok=True)
idx = json.load(open(f"{a}/model.safetensors.index.json"))
for shard in sorted(set(idx["weight_map"].values())):
    ta, tb = load_file(f"{a}/{shard}"), load_file(f"{b}/{shard}")
    save_file({k: ((1 - w) * ta[k].float() + w * tb[k].float()).to(ta[k].dtype) for k in ta}, f"{out}/{shard}", metadata={"format": "pt"})
for f in os.listdir(a):
    if not f.endswith(".safetensors") and f not in ("trainer_state.json", "training_args.bin", "args.json"):
        shutil.copy(f"{a}/{f}", out)
print("wrote", out)
