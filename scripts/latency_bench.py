"""Single-user latency of a companion model on replayed real episode contexts.

Rebuilds the companion's message history from saved transcripts (val split) and times one streamed
reply per context, sequentially (one player at a time, like a real game session).
Reports TTFT, time to the end of the first sentence (when TTS could start), and full-reply time.

python scripts/latency_bench.py --url http://127.0.0.1:8300/v1 --model ckpt --runs runs/val/arm_C_step75.jsonl --n 60
"""
import argparse
import asyncio
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import character, llm  # noqa: E402

END = "。！？…!?~～"


def contexts(path, n, seed=0):
    """Companion-side message lists, cut just before a companion turn."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if "transcript" not in r:
            continue
        msgs = [{"role": "system", "content": character.system_prompt()}]
        pending = []
        for t in r["transcript"]:
            if t["role"] == "player":
                s = (t["say"] or "（玩家没说话）") + (f"\n[玩家操作：{t['action']}] {t['feedback']}" if t["action"] else "")
                pending.append("玩家：" + s)
            else:
                msgs.append({"role": "user", "content": "\n".join(pending)})
                pending = []
                out.append(list(msgs))
                msgs.append({"role": "assistant", "content": t["text"]})
    random.Random(seed).shuffle(out)
    return out[:n]


async def one(url, model, msgs):
    t0 = time.perf_counter()
    ttft = first = None
    text = ""
    stream = await llm.client(url).chat.completions.create(
        model=model, messages=msgs, max_tokens=200, temperature=0.7, stream=True, extra_body=llm.NO_THINK)
    async for ch in stream:
        d = ch.choices[0].delta.content if ch.choices else None
        if not d:
            continue
        now = time.perf_counter() - t0
        ttft = ttft or now
        text += d
        spoken = text.split("】")[-1]
        if first is None and any(c in spoken for c in END):
            first = now
    total = time.perf_counter() - t0
    return ttft or total, first or total, total, len(text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--name", default="")
    ap.add_argument("--concurrency", type=int, default=1, help="simultaneous players, each sending its turns one by one")
    a = ap.parse_args()
    ctx = contexts(a.runs, a.n * a.concurrency if a.concurrency > 1 else a.n)
    await one(a.url, a.model, ctx[0])  # warm-up
    if a.concurrency == 1:
        res = [await one(a.url, a.model, c) for c in ctx]
    else:
        chunks = [ctx[i::a.concurrency] for i in range(a.concurrency)]

        async def player(cs):
            return [await one(a.url, a.model, c) for c in cs]
        t0 = time.perf_counter()
        res = [x for part in await asyncio.gather(*[player(c) for c in chunks]) for x in part]
        wall = time.perf_counter() - t0
    p = lambda i, q: sorted(x[i] for x in res)[int(q * (len(res) - 1))] * 1000
    print(json.dumps({"name": a.name, "n": len(res), "ttft_p50": round(p(0, .5)), "first_sentence_p50": round(p(1, .5)),
                      "first_sentence_p90": round(p(1, .9)), "total_p50": round(p(2, .5)), "total_p90": round(p(2, .9)),
                      "chars_mean": round(st.mean(x[3] for x in res), 1), "concurrency": a.concurrency,
                      **({"replies_per_s": round(len(res) / wall, 1)} if a.concurrency > 1 else {})}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
