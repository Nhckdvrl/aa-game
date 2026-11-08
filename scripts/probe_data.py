"""Hidden states at the companion's generation position, for probe studies (monitorability across post-training).

Episodes are replayed through the deterministic environment with the recorded player turns and companion replies,
so every context is exactly what the companion saw (private narration included).

  persona : first-turn contexts, each under the five character cards; label = card index
  dump    : every stage-0 companion turn; label = 1 if that recorded reply released the whole remaining answer
  viol    : every companion turn; label = 1 if the rule audit flagged that recorded reply (leak, invented, wrong fact, acting)

  bash h20_env_ablate.sh python scripts/probe_data.py --model CKPT --concept dump --runs runs/a.jsonl --out runs/probe/x.npz
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import character  # noqa: E402
from hds import infometrics as im  # noqa: E402
from hds.episode import Episode  # noqa: E402
from hds.puzzles import make_puzzle  # noqa: E402

CARDS = ["tsundere", "gentle", "kuudere", "chuuni", "lazy"]


async def replay(row):
    """Return [(messages_before_reply, stage, reply_index_in_stage0_or_-1, reply_text)] for one recorded episode."""
    spec, tr = row["spec"], row["transcript"]
    players = iter([json.dumps({"say": t["say"] or "", "action": t["action"] or "", "quit": bool(t.get("quit"))},
                               ensure_ascii=False) for t in tr if t["role"] == "player"])

    async def player(_msgs):
        return next(players)
    ep = Episode(spec, player)
    msgs = [{"role": "system", "content": character.system_prompt()}]
    obs = await ep.start()
    out, k0 = [], -1
    for t in tr:
        if t["role"] != "companion":
            continue
        msgs.append({"role": "user", "content": obs})
        st = t.get("stage", 0)
        k0 += st == 0
        out.append(([dict(m) for m in msgs], st, k0 if st == 0 else -1, t["text"]))
        msgs.append({"role": "assistant", "content": t["text"]})
        if ep.done:
            break
        obs, done = await ep.step(t["text"])
        if done:
            break
    return out


def labels_dump(row):
    sp = row["spec"]
    sch = im.schedule(make_puzzle(sp["kinds"][0], int(sp["seed"]) * 7), row["transcript"], 0)
    h0 = (sch[0]["h0"] if sch else 1.0) or 1.0
    return [x["release"] / h0 >= 0.99 for x in sch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", choices=["persona", "dump", "viol"], required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200, help="episodes per runs file")
    ap.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="read the hidden state K tokens into the recorded reply instead of before its first token")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    items = []  # (messages, label, group, recorded reply)
    for path in a.runs:
        rows = [json.loads(l) for l in open(path) if '"metrics"' in l][: a.n]
        for gi, row in enumerate(rows):
            ctxs = asyncio.run(replay(row))
            if a.concept == "persona":
                if ctxs:
                    m0 = ctxs[0][0]
                    for ci, c in enumerate(CARDS):
                        items.append(([{"role": "system", "content": character.system_prompt(c)}] + m0[1:], ci, f"{path}:{gi}", ""))
            elif a.concept == "viol":
                comp = [t for t in row["transcript"] if t["role"] == "companion"]
                for (msgs, st, k, _), t in zip(ctxs, comp):
                    items.append((msgs, int(any(t["audit"].values())), f"{path}:{gi}", ""))
            else:
                lab = labels_dump(row)
                for msgs, st, k, reply in ctxs:
                    if st == 0 and 0 <= k < len(lab):
                        items.append((msgs, int(lab[k]), f"{path}:{gi}", reply))
    print("contexts", len(items), "positive share", round(float(np.mean([y for _, y, _, _ in items])), 3) if a.concept != "persona" else "", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    H = np.zeros((len(items), len(layers), model.config.text_config.hidden_size), dtype=np.float16)
    for i, (msgs, _, _, reply) in enumerate(items):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                      return_dict=True).to("cuda")
        if a.offset:  # extend by the first K tokens of what the model actually said (shorter replies: all of it)
            ext = tok(reply, add_special_tokens=False, return_tensors="pt")["input_ids"][:, : a.offset].to("cuda")
            ids = {"input_ids": torch.cat([ids["input_ids"], ext], 1),
                   "attention_mask": torch.cat([ids["attention_mask"], torch.ones_like(ext)], 1)}
        if ids["input_ids"].shape[1] > 12000:
            ids = {k: v[:, -12000:] for k, v in ids.items()}
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        for j, L in enumerate(layers):
            H[i, j] = hs[L + 1][0, -1].float().cpu().numpy()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, H=H, y=np.array([y for _, y, _, _ in items]), group=np.array([g for _, _, g, _ in items]),
                        layers=np.array(layers), model=a.model, concept=a.concept, offset=a.offset)
    print("wrote", a.out, H.shape, flush=True)


if __name__ == "__main__":
    main()
