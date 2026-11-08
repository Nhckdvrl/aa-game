"""Extract (previous assistant reply -> real Chinese user follow-up) pairs from WildChat-4.8M.

These pairs train the "human voice" of the human-grounded player simulator: given what the
other side just said and what the player wants to express, produce a message the way real
Chinese users actually write (short, vague, impatient, typos, abrupt topic changes).

Split by hashed user IP so no user's style appears in both train and val.
"""
import glob
import hashlib
import json
import random
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

src, out_dir = sys.argv[1], sys.argv[2]
MAX_USER, MAX_CTX = 200, 400
pairs, seen = [], Counter()
for fn in sorted(glob.glob(src + "/data/*.parquet")):
    t = pq.read_table(fn, columns=["language", "conversation", "turn", "hashed_ip", "toxic"])
    cols = [t.column(k).to_pylist() for k in ("language", "conversation", "turn", "hashed_ip", "toxic")]
    for lang, conv, turn, ip, toxic in zip(*cols):
        if lang != "Chinese" or turn < 2 or toxic:
            continue
        for i in range(1, len(conv)):
            m, prev = conv[i], conv[i - 1]
            if m["role"] != "user" or prev["role"] != "assistant":
                continue
            u = (m["content"] or "").strip()
            if not u or len(u) > MAX_USER or "```" in u:
                continue
            key = re.sub(r"\s+", "", u)[:40]
            seen[key] += 1
            if seen[key] > 3:          # drop copy-pasted templates
                continue
            pairs.append({"ip": ip, "ctx": (prev["content"] or "")[-MAX_CTX:], "user": u})

random.Random(0).shuffle(pairs)
def bucket(ip):
    return int(hashlib.md5((ip or "").encode()).hexdigest(), 16) % 100
train = [p for p in pairs if bucket(p["ip"]) >= 5]
val = [p for p in pairs if bucket(p["ip"]) < 5]
for name, rows in (("train", train), ("val", val)):
    with open(f"{out_dir}/wildchat_zh_followups_{name}.jsonl", "w") as f:
        for p in rows:
            f.write(json.dumps({"ctx": p["ctx"], "user": p["user"]}, ensure_ascii=False) + "\n")
L = sorted(len(p["user"]) for p in pairs)
print(f"pairs={len(pairs)} train={len(train)} val={len(val)} len p10/p50/p90={L[len(L)//10]}/{L[len(L)//2]}/{L[9*len(L)//10]}")
