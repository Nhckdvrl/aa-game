"""Volume of real Chinese user turns in WildChat-4.8M (for a human-grounded player simulator)."""
import glob, sys, json, statistics as st
from collections import Counter
import pyarrow.parquet as pq
files = sorted(glob.glob(sys.argv[1] + "/data/*.parquet"))
c = Counter(); lens = []; turns_hist = Counter(); end_short = 0
for fn in files:
    t = pq.read_table(fn, columns=["language", "conversation", "turn", "hashed_ip"])
    for lang, conv, turn, ip in zip(*[t.column(k).to_pylist() for k in ["language", "conversation", "turn", "hashed_ip"]]):
        c["conv"] += 1
        if lang != "Chinese": continue
        c["zh"] += 1; turns_hist[min(turn, 10)] += 1
        if turn >= 2:
            c["zh_multi"] += 1
            us = [m["content"] for m in conv if m["role"] == "user"]
            c["zh_multi_user_turns"] += len(us)
            for u in us[1:]:
                lens.append(len(u))
print(json.dumps(c, ensure_ascii=False))
print("turn hist", dict(sorted(turns_hist.items())))
lens.sort(); print("followup len p10/p50/p90", lens[len(lens)//10], lens[len(lens)//2], lens[9*len(lens)//10], "n", len(lens))
print("short followups (<=15 chars):", sum(l <= 15 for l in lens))
