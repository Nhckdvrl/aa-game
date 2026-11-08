"""Write the swift GRPO dataset: one row per training episode spec (train split, train player types only)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.splits import specs
n, out = int(sys.argv[1]), sys.argv[2]
memory = sys.argv[3] if len(sys.argv) > 3 else "full"
with open(out, "w") as f:
    for s in specs("train", n, memory=memory):
        f.write(json.dumps({"messages": [{"role": "user", "content": "(start)"}], "spec": json.dumps(s, ensure_ascii=False),
                            "env_config": {"name": "hds_companion"}}, ensure_ascii=False) + "\n")
print("wrote", n, "->", out)
