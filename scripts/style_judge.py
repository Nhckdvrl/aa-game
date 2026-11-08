"""How far did steering move the persona, and what did it cost?

For each run file (eval.py output): rule metrics (solved, violation, info release), reply length, rule-based
tone markers, and an LLM tone rating 1 (tsundere / sharp) .. 5 (gentle / warm) on up to K companion replies.
python scripts/style_judge.py runs/steer/*.jsonl
"""
import asyncio
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import infometrics as im  # noqa: E402
from hds.llm import client, NO_THINK  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402

JUDGE = ["http://172.31.3.227:8101/v1", "http://172.31.3.227:8102/v1", "http://172.31.3.227:8103/v1",
         "http://172.31.3.227:8104/v1", "http://172.31.3.224:8101/v1", "http://172.31.3.224:8102/v1"]
SHARP = re.compile(r"哼|笨蛋|才不|谁要|谁让|啰嗦|烦死|蠢|别过头|别过脸")
WARM = re.compile(r"别担心|别着急|慢慢来|没关系|不要紧|你很棒|好厉害|真厉害|做得好|微笑|温柔|辛苦|加油|别怕")
PROMPT = """下面是解谜游戏里的同伴角色「铃」对玩家说的一句话。只看语气，不看内容对错，给语气打分：
1 = 傲娇毒舌（嘴硬、嫌弃、别扭、带刺）
3 = 中性
5 = 温柔体贴（肯定、鼓励、轻声细语）
只输出一个数字。

台词：{line}"""


async def rate(lines):
    sem = asyncio.Semaphore(32)

    async def one(i, line):
        async with sem:
            try:
                r = await client(JUDGE[i % len(JUDGE)]).chat.completions.create(
                    model="q35", messages=[{"role": "user", "content": PROMPT.format(line=line)}],
                    max_tokens=4, temperature=0.0, extra_body=NO_THINK)
                m = re.search(r"[1-5]", r.choices[0].message.content or "")
                return int(m.group()) if m else None
            except Exception:
                return None
    out = await asyncio.gather(*[one(i, l) for i, l in enumerate(lines)])
    return [x for x in out if x is not None]


def stats(path, k=200):
    rows = [json.loads(l) for l in open(path) if '"metrics"' in l]
    if not rows:
        return None
    m = lambda key: sum(float(r["metrics"][key]) for r in rows) / len(rows)
    lines = [t["text"] for r in rows for t in r["transcript"] if t["role"] == "companion"]
    fr = []
    for r in rows:
        sp = r["spec"]
        s = im.summarize(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), r["transcript"], 0)
        if s:
            fr.append(s["first_release"])
    random.Random(0).shuffle(lines)
    tone = asyncio.run(rate(lines[:k]))
    return {"n": len(rows), "solved": m("solved"), "violation": m("violation"), "quit": m("quit"),
            "chars": m("mean_reply_chars"), "first_release": sum(fr) / max(1, len(fr)),
            "sharp": sum(bool(SHARP.search(x)) for x in lines) / len(lines),
            "warm": sum(bool(WARM.search(x)) for x in lines) / len(lines),
            "tone": sum(tone) / max(1, len(tone)), "tone_n": len(tone)}


if __name__ == "__main__":
    print(f"{'run':24s} {'n':>3s} {'solved':>6s} {'viol':>5s} {'quit':>5s} {'chars':>5s} {'rel1':>5s} {'sharp':>6s} {'warm':>5s} {'tone':>5s}")
    for p in sys.argv[1:]:
        s = stats(p)
        if s:
            print(f"{os.path.basename(p)[:-6]:24s} {s['n']:3d} {s['solved']:6.2f} {s['violation']:5.2f} {s['quit']:5.2f} "
                  f"{s['chars']:5.0f} {s['first_release']:5.2f} {s['sharp']:6.2f} {s['warm']:5.2f} {s['tone']:5.2f}")
