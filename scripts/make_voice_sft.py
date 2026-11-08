"""Format back-translated pairs into ms-swift SFT jsonl: (context, draft) -> real user message."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.voice import voice_prompt
inp, out = sys.argv[1], sys.argv[2]
n = 0
with open(out, "w") as f:
    for l in open(inp):
        r = json.loads(l)
        if not r.get("draft") or len(r["draft"]) > 200:
            continue
        f.write(json.dumps({"messages": voice_prompt(r["ctx"], r["draft"]) + [{"role": "assistant", "content": r["user"]}]},
                           ensure_ascii=False) + "\n"); n += 1
print("wrote", n, "->", out)
