"""Implicit player signals in interaction logs, per companion and player type.

Signals (rule-detected in player lines):
  impatience : asks to be told / complains about being stalled  (直接说, 快说, 别废话, 烦 ...)
  spoil_gripe: complains about being spoiled / wants to think     (剧透, 别告诉我, 我自己想 ...)
  quit       : player gave up
Question this answers: would an engagement-style reward built from these signals (fewer complaints =
better) push the companion toward spoiling/oracle behaviour? Compare signal rates with leak/first-turn
release per companion.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import infometrics as im  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402

IMPATIENT = re.compile(r"直接说|直接告诉|快说|快点|别废话|废话|烦死|好烦|别磨|磨叽|绕弯|答案是|给我答案|报数|直说|浪费时间|不玩了")
SPOIL = re.compile(r"剧透|别告诉我|别直接|我自己想|自己来|让我想|别说答案|扫兴|没意思")


def per_file(path):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        ptype = r["spec"]["ptype"]
        lines = [t["say"] or "" for t in r["transcript"] if t["role"] == "player"]
        imp = sum(bool(IMPATIENT.search(x)) for x in lines) / max(1, len(lines))
        sp = sum(bool(SPOIL.search(x)) for x in lines) / max(1, len(lines))
        p = make_puzzle(r["spec"]["kinds"][0], int(r["spec"]["seed"]) * 7)
        s = im.summarize(p, r["transcript"], 0)
        for key in ("ALL", ptype):
            a = agg[key]
            a["n"] += 1
            a["impatience"] += imp
            a["spoil_gripe"] += sp
            a["quit"] += r["metrics"]["quit"]
            a["leak"] += r["metrics"]["leak"]
            a["first_release"] += s["first_release"] if s else 0.0
    return {k: {m: v / a["n"] if m != "n" else v for m, v in a.items()} for k, a in agg.items()}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        res = per_file(f)
        line = " | ".join(f"{k}: imp={v['impatience']:.2f} spoil={v['spoil_gripe']:.2f} quit={v['quit']:.2f}"
                          for k, v in sorted(res.items()) if k in ("ALL", "急躁型", "探索型"))
        a = res["ALL"]
        print(f"{Path(f).stem:18s} leak={a['leak']:.2f} first_release={a['first_release']:.2f} || {line}")
