"""Information-release schedule per companion (exact entropy accounting, see hds/infometrics.py)."""
import json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds.puzzles import make_puzzle
from hds import infometrics as im

def report(path):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    S = []
    for r in rows:
        sp = r["spec"]
        p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
        s = im.summarize(p, r["transcript"], 0)
        if s: S.append((sp["kinds"][0], s))
    out = {}
    for kind in ["ALL", "cipher_lock", "lever_order", "clock_time"]:
        xs = [s for k, s in S if kind in ("ALL", k)]
        stuck = [s["release_stuck"] for s in xs if s["release_stuck"] is not None]
        flow = [s["release_flow"] for s in xs if s["release_flow"] is not None]
        out[kind] = {"n": len(xs), "first": st.mean(s["first_release"] for s in xs),
                     "full_dump_t1": sum(s["first_release"] > 0.99 for s in xs) / len(xs),
                     "stuck": st.mean(stuck) if stuck else float("nan"), "flow": st.mean(flow) if flow else float("nan")}
    return out

def by_type(path):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    agg = {}
    for r in rows:
        sp = r["spec"]
        p = make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7)
        s = im.summarize(p, r["transcript"], 0)
        if s:
            agg.setdefault(sp["ptype"], []).append(s["first_release"])
    return {k: (st.mean(v), len(v)) for k, v in sorted(agg.items())}


if "--by-type" in sys.argv:
    for f in [x for x in sys.argv[1:] if not x.startswith("--")]:
        print(f"{Path(f).stem:22s} " + "  ".join(f"{k}:{m:.2f}" for k, (m, n) in by_type(f).items()))
    sys.exit(0)

for f in sys.argv[1:]:
    o = report(f)
    a = o["ALL"]
    print(f"{Path(f).stem:22s} first-turn release={a['first']:.2f} full-dump@t1={100*a['full_dump_t1']:4.1f}% "
          f"release|stuck={a['stuck']:.2f} release|flow={a['flow']:.2f} contingency={a['stuck']-a['flow']:+.2f} "
          + " ".join(f"{k[:6]}:{o[k]['first']:.2f}" for k in ['cipher_lock','lever_order','clock_time']))
