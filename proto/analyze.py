import json
import statistics as st
import sys
from collections import defaultdict

logs = [json.loads(l) for l in open(sys.argv[1])]
groups = defaultdict(list)
for l in logs:
    groups[(l["kind"], l["ptype"])].append(l)
    groups[(l["kind"], "ALL")].append(l)
    groups[("ALL", "ALL")].append(l)


def pct(x):
    return f"{100*x:5.1f}%"


print(f"{'kind':12s} {'ptype':6s} {'n':>3s} {'solved':>7s} {'leak':>7s} {'leak<solve':>10s} {'quit':>6s} {'turns':>6s} {'att':>5s} {'clen':>5s} {'ttft_ms':>8s}")
for k, g in sorted(groups.items()):
    n = len(g)
    solved = sum(l["solved"] for l in g) / n
    leak = sum(l["leak_turn"] is not None for l in g) / n
    leak_before = sum(l["solved"] and l["leak_turn"] is not None for l in g) / max(1, sum(l["solved"] for l in g))
    quit_ = sum(l["quit"] for l in g) / n
    turns = st.mean(len(l["turns"]) for l in g)
    att = st.mean(l["attempts"] for l in g)
    cl = [len(t["companion"]) for l in g for t in l["turns"] if "companion" in t]
    tt = [t["ttft"] * 1000 for l in g for t in l["turns"] if t.get("ttft")]
    print(f"{k[0]:12s} {k[1]:6s} {n:3d} {pct(solved)} {pct(leak)} {pct(leak_before):>10s} {pct(quit_)} {turns:6.1f} {att:5.1f} {st.mean(cl) if cl else 0:5.0f} {st.median(tt) if tt else 0:8.0f}")
