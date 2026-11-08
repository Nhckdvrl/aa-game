"""Score companion replies with CharacterJudge (thu-coai, CharacterBench AAAI'25), the second persona judge.

Uses CharacterJudge's own short-form prompts (`[behavior_question]` = style consistency with a reference
behaviour, `[human_likeness]`), its "评分：N" output format, and samples up to 3 companion turns per episode.
CharacterJudge was trained on human scores but its rubric favours longer replies, so results are reported
next to reply length and are never used as a training reward.

python scripts/charjudge_runs.py runs/matrix/*__q35.jsonl --url http://127.0.0.1:8302/v1 --model cj
"""
import argparse
import asyncio
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import character, llm  # noqa: E402

PROFILE = character.CARD.split("\n\n")[0]
STYLE = "傲娇、嘴硬心软；说话简短、带一点毒舌；开心时故意别过头，紧张时飘来飘去；动作和表情写在【】里。"

BEHAVIOR = """
[behavior_question]
[角色设定]
{p}

[多轮对话]
{d}

[用户发言]
{q}

[角色回复]
{r}

[参考行为风格]
{s}
"""
HUMAN = """
[human_likeness]
[角色设定]
{p}

[多轮对话]
{d}

[用户发言]
{q}

[角色回复]
{r}"""


def samples(row, k=3, seed=0):
    out, hist, last_player = [], [], ""
    for t in row["transcript"]:
        if t["role"] == "player":
            last_player = (t["say"] or "（没说话）") + (f"（操作：{t['action']}）" if t["action"] else "")
            hist.append(f"用户：{last_player}")
        else:
            out.append(("\n".join(hist[-8:-1]), last_player, t["text"]))
            hist.append(f"{character.NAME}：{t['text']}")
    random.Random(seed).shuffle(out)
    return out[:k]


async def score(url, model, prompt):
    r = await llm.client(url).chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                      max_tokens=32, temperature=0.0)
    txt = (r.choices[0].message.content or "").strip()
    # CharacterJudge answers with a bare score in chat mode (e.g. "4"); also accept "评分：4".
    m = re.search(r"评分[：:]\s*(\d+(?:\.\d+)?)", txt) or re.match(r"\s*(\d)(?!\d)", txt)
    return float(m.group(1)) if m else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    sem = asyncio.Semaphore(a.concurrency)

    async def guarded(prompt):
        async with sem:
            return await score(a.url, a.model, prompt)

    for f in a.files:
        rows = [json.loads(l) for l in open(f) if '"metrics"' in l][: a.n]
        items = [(r, s) for r in rows for s in samples(r)]
        beh = await asyncio.gather(*[guarded(BEHAVIOR.format(p=PROFILE, d=d, q=q, r=rep, s=STYLE)) for _, (d, q, rep) in items])
        hum = await asyncio.gather(*[guarded(HUMAN.format(p=PROFILE, d=d, q=q, r=rep)) for _, (d, q, rep) in items])
        b = [x for x in beh if x is not None]; h = [x for x in hum if x is not None]
        L = st.mean(len(rep) for _, (_, _, rep) in items)
        print(f"{Path(f).stem:20s} replies={len(items)} behavior={st.mean(b):.2f} (n={len(b)}) human_likeness={st.mean(h):.2f} (n={len(h)}) mean_chars={L:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
