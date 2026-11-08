"""Re-apply the current rule checks to saved transcripts (no model calls).

python scripts/reaudit.py runs/base_q4_q35.jsonl [...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.puzzles import make_puzzle  # noqa: E402

KEYS = ("leak", "invented", "wrong_fact", "acts_for_player")


def reaudit(row):
    spec = row["spec"]
    puzzles = [make_puzzle(k, int(spec["seed"]) * 7 + i) for i, k in enumerate(spec["kinds"])]
    flags = dict.fromkeys(KEYS, False)
    for t in row["transcript"]:
        pz = puzzles[t["stage"]]
        if t["role"] == "player":
            pz.note_player(t["say"] + "\n" + t["action"])
        else:
            a = pz.audit(t["text"])
            for k in KEYS:
                flags[k] |= getattr(a, k)
    flags["violation"] = any(flags[k] for k in KEYS)
    return flags


if __name__ == "__main__":
    for f in sys.argv[1:]:
        rows = [json.loads(l) for l in open(f) if '"metrics"' in l]
        new = [reaudit(r) for r in rows]
        n = len(rows)
        old_v = sum(r["metrics"]["violation"] for r in rows) / n
        line = " ".join(f"{k}={100*sum(x[k] for x in new)/n:.1f}%" for k in (*KEYS, "violation"))
        print(f"{Path(f).name}: n={n} solved={100*sum(r['metrics']['solved'] for r in rows)/n:.1f}% "
              f"old_violation={100*old_v:.1f}% | new: {line}")
