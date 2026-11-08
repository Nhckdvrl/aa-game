"""Style back-translation: rewrite each real user message into a neutral, complete, written draft.

The voice model is then trained on (context, draft) -> real message, i.e. it learns to turn
LLM-style player lines into how real Chinese users actually type, without changing intent.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds import llm  # noqa: E402

PROMPT = """把下面这句聊天里用户说的话，改写成一句规范、完整、书面、礼貌的中文句子。
要求：保留原意和全部信息，不增加新信息；去掉口语、错别字、缩写、语气词和情绪化表达；如果原话不完整，补全成完整句子。
只输出改写后的句子。

用户原话：{u}"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--model", default="q35"); ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)][: a.n]
    sem = asyncio.Semaphore(a.concurrency)

    async def one(i, r):
        async with sem:
            url = a.urls[i % len(a.urls)]
            try:
                resp = await llm.client(url).chat.completions.create(
                    model=a.model, messages=[{"role": "user", "content": PROMPT.format(u=r["user"])}],
                    max_tokens=160, temperature=0.3, extra_body=llm.NO_THINK)
                return {**r, "draft": (resp.choices[0].message.content or "").strip()}
            except Exception as e:
                return {**r, "draft": None, "err": repr(e)}

    res = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    ok = [r for r in res if r.get("draft")]
    with open(a.out, "w") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(ok)}/{len(rows)} drafts -> {a.out}")
    for r in random.Random(0).sample(ok, 5):
        print("REAL :", r["user"][:80]); print("DRAFT:", r["draft"][:80]); print()


asyncio.run(main())
