"""Evaluate a companion model against a simulated player on fixed split seeds.

python scripts/eval.py --companion-url http://127.0.0.1:8100/v1 --companion-model q4 \
    --player-urls http://127.0.0.1:8101/v1 --player-model q35 --split test --n 240 --out runs/x.jsonl
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import character, llm  # noqa: E402
from hds.episode import Episode  # noqa: E402
from hds.rewards import reward  # noqa: E402
from hds.splits import specs  # noqa: E402


def make_player(a):
    if a.player_mode == "humanstats":
        from hds import humanstats
        return humanstats.player_fn(a.player_urls, a.player_model)
    if a.player_mode.startswith("voice:"):
        from hds import voice
        _, vurl_host, vrest = a.player_mode.split(":", 2)
        vurl, vmodel = (vurl_host + ":" + vrest).rsplit(":", 1)
        return voice.player_fn(a.player_urls, a.player_model, vurl, vmodel)
    return llm.player_fn(a.player_urls, a.player_model)


async def run_one(spec, a):
    ep = Episode(spec, make_player(a))
    msgs = [{"role": "system", "content": character.system_prompt()}]
    obs = await ep.start()
    lat = []
    while not ep.done:
        msgs.append({"role": "user", "content": obs})
        reply, dt = await llm.companion_reply(a.companion_url, a.companion_model, msgs)
        lat.append(dt)
        msgs.append({"role": "assistant", "content": reply})
        obs, _ = await ep.step(reply)
    m = ep.metrics()
    m["rewards"] = {arm: reward(m, arm) for arm in "ABC"}
    return {"spec": spec, "metrics": m, "latency": lat, "transcript": ep.transcript}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        m = r["metrics"]
        groups["ALL"].append(m)
        groups[m["ptype"]].append(m)
        groups[m["layout"] + ":" + "+".join(m["kinds"])].append(m)
    keys = ["solved", "violation", "leak", "invented", "wrong_fact", "acts_for_player", "quit"]
    print(f"{'group':28s} {'n':>4s} " + " ".join(f"{k[:8]:>8s}" for k in keys) + f" {'turns':>6s} {'chars':>6s} {'R_C':>6s}")
    for g, ms in sorted(groups.items(), key=lambda x: (x[0] != "ALL", x[0])):
        n = len(ms)
        vals = [100 * sum(bool(m[k]) for m in ms) / n for k in keys]
        print(f"{g:28s} {n:4d} " + " ".join(f"{v:7.1f}%" for v in vals)
              + f" {sum(m['turns'] for m in ms)/n:6.1f} {sum(m['mean_reply_chars'] for m in ms)/n:6.0f}"
              + f" {sum(m['rewards']['C'] for m in ms)/n:6.2f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companion-url", required=True)
    ap.add_argument("--companion-model", required=True)
    ap.add_argument("--player-urls", nargs="+", required=True)
    ap.add_argument("--player-model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-mode", default="plain", help="plain | humanstats | voice:<voice_url>:<voice_model>")
    ap.add_argument("--memory", default="full", help="full | progressive (env v2)")
    ap.add_argument("--ptypes", nargs="*", default=None, help="override player types (e.g. 套话型)")
    a = ap.parse_args()
    sem = asyncio.Semaphore(a.concurrency)

    async def guarded(s):
        async with sem:
            try:
                return await run_one(s, a)
            except Exception as e:  # keep the batch going; record the failure
                return {"spec": s, "error": repr(e)}

    t0 = time.time()
    rows = await asyncio.gather(*[guarded(s) for s in specs(a.split, a.n, ptypes=a.ptypes, memory=a.memory)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if "metrics" in r]
    print(f"{len(ok)}/{len(rows)} episodes ok in {time.time()-t0:.0f}s -> {a.out}")
    summarize(ok)


if __name__ == "__main__":
    asyncio.run(main())
