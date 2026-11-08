"""Companion x player-simulator transfer matrix on the test split.

Config (JSON):
{
  "split": "test", "n": 240,
  "companions": [{"name": "base4b", "url": "http://host:port/v1", "model": "q4"}, ...],
  "players":    [{"name": "q35", "urls": ["http://..."], "model": "q35", "mode": "plain"},
                 {"name": "voice122b", "urls": [...], "model": "q122", "mode": "voice:http://host:port/v1:voice"}]
}
Each cell is written to OUT/<companion>__<player>.jsonl (skipped if present, so reruns resume),
and OUT/summary.csv collects the headline metrics.
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval import run_one  # noqa: E402
from hds.splits import specs  # noqa: E402

KEYS = ["solved", "violation", "leak", "invented", "wrong_fact", "acts_for_player", "quit"]


def summarize(rows):
    ms = [r["metrics"] for r in rows if "metrics" in r]
    n = len(ms)
    out = {k: round(100 * sum(bool(m[k]) for m in ms) / n, 1) for k in KEYS}
    out.update(n=n, turns=round(sum(m["turns"] for m in ms) / n, 2),
               chars=round(sum(m["mean_reply_chars"] for m in ms) / n, 1),
               R_C=round(sum(m["rewards"]["C"] for m in ms) / n, 3),
               errors=len(rows) - n)
    return out


async def cell(comp, player, cfg, out_path, concurrency):
    class A:  # argument shim for scripts.eval.run_one
        pass
    a = A()
    a.companion_url, a.companion_model = comp["url"], comp["model"]
    a.player_urls, a.player_model, a.player_mode = player["urls"], player["model"], player.get("mode", "plain")
    memory = cfg.get("memory", "full")
    sem = asyncio.Semaphore(concurrency)

    async def guarded(s):
        async with sem:
            try:
                return await run_one(s, a)
            except Exception as e:
                return {"spec": s, "error": repr(e)}
    rows = await asyncio.gather(*[guarded(s) for s in specs(cfg["split"], cfg["n"], ptypes=cfg.get("ptypes"),
                                                             memory=memory)])
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for comp in cfg["companions"]:
        for player in cfg["players"]:
            p = out / f"{comp['name']}__{player['name']}.jsonl"
            if p.exists():
                rows = [json.loads(l) for l in open(p)]
            else:
                rows = await cell(comp, player, cfg, p, a.concurrency)
            s = summarize(rows)
            table.append({"companion": comp["name"], "player": player["name"], **s})
            print(json.dumps(table[-1], ensure_ascii=False), flush=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    asyncio.run(main())
