"""Evaluate a companion on a filtered slice of the test split (e.g. two-world episodes where both worlds are
cipher locks), optionally under an eval-only room variant (env HDS_CHAIN_MODE, see hds/episode.py).

python scripts/eval_subset.py --companion-url URL --companion-model ckpt --player-urls U1 U2 --player-model q35 \
    --first cipher_lock --n 120 --memory progressive --out runs/chainfix/X.jsonl
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds.splits import spec  # noqa: E402
from scripts.eval import run_one  # noqa: E402


async def main(a):
    sp, i = [], 0
    while len(sp) < a.n and i < 20000:
        s = spec("test", i, None, a.memory)
        if s["layout"] == "chain" and s["kinds"][0] == a.first:
            sp.append(s)
        i += 1
    sem = asyncio.Semaphore(a.concurrency)

    async def one(s):
        async with sem:
            try:
                return await run_one(s, a)
            except Exception as e:
                return {"spec": s, "error": repr(e)}
    rows = await asyncio.gather(*[one(s) for s in sp])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if "metrics" in r]
    m = lambda k: sum(float(bool(r["metrics"][k])) for r in ok) / max(1, len(ok))
    print(json.dumps({"out": a.out, "mode": os.environ.get("HDS_CHAIN_MODE", "orig"), "n": len(ok),
                      "errors": len(rows) - len(ok), "solved": m("solved"), "violation": m("violation"),
                      "wrong_fact": m("wrong_fact")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--companion-url", required=True)
    ap.add_argument("--companion-model", required=True)
    ap.add_argument("--player-urls", nargs="+", required=True)
    ap.add_argument("--player-model", required=True)
    ap.add_argument("--player-mode", default="plain")
    ap.add_argument("--first", default="cipher_lock")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--memory", default="progressive")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
