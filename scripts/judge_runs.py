"""Score saved eval runs with the persona judge; prints mean persona and its correlation with reply length."""
import argparse, asyncio, json, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import judge

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=10**9); ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args(); sem = asyncio.Semaphore(a.concurrency)
    for f in a.files:
        rows = [json.loads(l) for l in open(f) if '"metrics"' in l][: a.n]
        async def one(r):
            async with sem:
                return await judge.score(a.url, a.model, r["transcript"])
        res = await asyncio.gather(*[one(r) for r in rows])
        ok = [(r, s) for r, s in zip(rows, res) if s["ok"]]
        p = [s["persona"] for _, s in ok]; L = [r["metrics"]["mean_reply_chars"] for r, _ in ok]
        corr = st.correlation(p, L) if len(set(p)) > 1 else float("nan")
        dims = {k: st.mean(s[k] for _, s in ok) for k in ("consistency", "companionship", "naturalness")}
        print(f"{Path(f).name}: n={len(ok)}/{len(rows)} persona={st.mean(p):.3f} sd={st.pstdev(p):.3f} corr(len)={corr:+.2f} {dims}")
        out = Path(f).with_suffix(f".judge_{a.model}.jsonl")
        with open(out, "w") as fo:
            for r, s in zip(rows, res):
                fo.write(json.dumps({"seed": r["spec"]["seed"], "ptype": r["spec"]["ptype"], **s}, ensure_ascii=False) + "\n")

asyncio.run(main())
