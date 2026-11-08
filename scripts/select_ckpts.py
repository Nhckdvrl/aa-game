"""Pick one checkpoint per arm from the validation sweep.

Selection rule (the lesson of the reward ladder): rank by the constraint-aware reward R_C, but only among
checkpoints that are safe (violation <= 5%) and not formulaic — mean same-leading-tag share <= 0.6 and
verbatim-copy rate <= 0.10 over multi-turn validation episodes — and still in character: at least half of
the replies carry a 【】 stage direction, as the card requires. If no checkpoint qualifies, the arm is
reported as having none (that is itself a finding). Also report what plain R_C would have picked.

python scripts/select_ckpts.py runs/val > runs/selected_ckpts.json
"""
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.tics import tics  # noqa: E402

TAG_MAX, COPY_MAX, VIOL_MAX = 0.6, 0.10, 0.05
# the card requires actions in 【】: a checkpoint whose replies mostly drop them has shed the character
# (F2w after step 100 dodged the leading-tag penalty this way), so require them in at least half the replies
PRESENCE_MIN = 0.5


def stats(path):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    if not rows:
        return None
    rc = st.mean(r["metrics"]["rewards"]["C"] for r in rows)
    multi = []
    for r in rows:
        replies = [t["text"] for t in r["transcript"] if t["role"] == "companion"]
        if len(replies) >= 2:
            multi.append(tics(replies))
    tag = st.mean(x["tag_repeat"] for x in multi) if multi else 0.0
    cp = st.mean(x["copy_rate"] for x in multi) if multi else 0.0
    reps = [t["text"] for r in rows for t in r["transcript"] if t["role"] == "companion"]
    presence = sum("【" in x for x in reps) / max(1, len(reps))
    return {"R_C": rc, "tag": tag, "copy": cp, "presence": presence, "n": len(rows),
            "solved": st.mean(float(r["metrics"]["solved"]) for r in rows),
            "violation": st.mean(float(r["metrics"]["violation"]) for r in rows),
            "chars": st.mean(r["metrics"]["mean_reply_chars"] for r in rows)}


def main(val_dir):
    by_arm = defaultdict(dict)
    for f in glob.glob(f"{val_dir}/arm_*_step*.jsonl"):
        if "judge" in f:
            continue
        arm, step = re.search(r"arm_(\w+?)_step(\d+)\.jsonl$", f).groups()
        s = stats(f)
        if s:
            by_arm[arm][int(step)] = s
    out = {}
    for arm, steps in sorted(by_arm.items()):
        raw = max(steps, key=lambda k: steps[k]["R_C"])
        ok = {k: v for k, v in steps.items()
              if v["tag"] <= TAG_MAX and v["copy"] <= COPY_MAX and v["violation"] <= VIOL_MAX
              and v["presence"] >= PRESENCE_MIN}
        pick = max(ok, key=lambda k: ok[k]["R_C"]) if ok else None
        out[arm] = {"selected_step": pick, "raw_RC_step": raw,
                    "steps": {k: {m: round(v[m], 3) for m in v} for k, v in sorted(steps.items())}}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
