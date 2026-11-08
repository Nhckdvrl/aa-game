"""Blind persona identification: does the companion sound like the card it was given?

For each episode in an eval run, the 35B judge reads only Rin's lines (stage directions kept) and picks which of
the five character cards they match. Accuracy against the intended card (chance = 20%) measures persona
following; the confusion matrix shows which personalities bleed into each other.

python scripts/persona_id.py runs/x__q35.jsonl:kuudere runs/y__q35.jsonl:tsundere ...
"""
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import character  # noqa: E402
from hds.llm import NO_THINK, client  # noqa: E402

JUDGE = ["http://172.31.3.227:8101/v1", "http://172.31.3.227:8102/v1", "http://172.31.3.227:8103/v1",
         "http://172.31.3.227:8104/v1", "http://172.31.3.224:8101/v1", "http://172.31.3.224:8102/v1"]
ORDER = ["tsundere", "gentle", "kuudere", "chuuni", "lazy"]
LABEL = "ABCDE"
LINES = {"tsundere": character.TSUNDERE, "gentle": character.GENTLE, "kuudere": character.KUUDERE,
         "chuuni": character.CHUUNI, "lazy": character.LAZY}
PROMPT = """下面是解谜游戏里同伴角色「铃」在一局里说的话（只有她的台词）。请判断她最符合下面哪一种性格设定，只输出一个字母。

{options}

铃的台词：
{lines}"""


async def judge(lines, i):
    options = "\n".join(f"{LABEL[k]}. {LINES[p]}" for k, p in enumerate(ORDER))
    r = await client(JUDGE[i % len(JUDGE)]).chat.completions.create(
        model="q35", messages=[{"role": "user", "content": PROMPT.format(options=options, lines=lines[:1500])}],
        max_tokens=4, temperature=0.0, extra_body=NO_THINK)
    m = re.search(r"[A-E]", r.choices[0].message.content or "")
    return ORDER[LABEL.index(m.group())] if m else None


async def run(items, k=40):
    sem = asyncio.Semaphore(32)

    async def one(i, lines, want):
        async with sem:
            try:
                return want, await judge(lines, i)
            except Exception:
                return want, None
    jobs = []
    for path, want in items:
        rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
        random.Random(0).shuffle(rows)
        for r in rows[:k]:
            lines = "\n".join(t["text"] for t in r["transcript"] if t["role"] == "companion")
            if lines.strip():
                jobs.append((len(jobs), lines, want))
    return await asyncio.gather(*[one(i, l, w) for i, l, w in jobs])


if __name__ == "__main__":
    items = [(a.rsplit(":", 1)[0], a.rsplit(":", 1)[1]) for a in sys.argv[1:]]
    res = asyncio.run(run(items))
    by = defaultdict(Counter)
    for want, got in res:
        by[want][got] += 1
    print(f"{'card':9s} {'n':>3s} {'accuracy':>8s}  confusion")
    tot = cor = 0
    for want in ORDER:
        if want not in by:
            continue
        c = by[want]; n = sum(c.values()); tot += n; cor += c[want]
        print(f"{want:9s} {n:3d} {100 * c[want] / n:7.0f}%  " + " ".join(f"{p}:{c[p]}" for p in ORDER + [None] if c[p]))
    print(f"overall accuracy {100 * cor / max(1, tot):.0f}% (chance 20%)")
