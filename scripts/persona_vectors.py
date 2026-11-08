"""Persona steering vectors by contrastive prompting (mean difference at the generation position).

Persona A = the shipped card (tsundere Rin). Persona B = the same card with only the personality line swapped
(gentle Rin). Contexts are real opening observations from validation episodes (player pool answers the first
turn). For every decoder layer we store mean(h_B) - mean(h_A) at the last prompt token; for every head of the
full-attention layers we store the same difference at the o_proj input, and rank heads by
|d_h| / mean|a_h| (Plan-2 style "style heads").

  bash h20_env_ablate.sh python scripts/persona_vectors.py CKPT runs/steer/vectors.pt [N]
"""
import asyncio
import os
import sys

import torch
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import character, llm  # noqa: E402
from hds.episode import Episode  # noqa: E402
from hds.splits import specs  # noqa: E402

POOL = ["http://172.31.3.227:8101/v1", "http://172.31.3.227:8102/v1", "http://172.31.3.227:8103/v1",
        "http://172.31.3.227:8104/v1", "http://172.31.3.224:8101/v1", "http://172.31.3.224:8102/v1"]
TSUNDERE = "性格：嘴硬心软的傲娇，说话简短、带一点毒舌，但其实很怕孤单，很在意玩家。开心时会故意别过头，紧张时会飘来飘去。"
GENTLE = "性格：温柔体贴，说话轻声细语，总是先肯定玩家，再耐心地鼓励。开心时会微笑，担心时会轻轻飘到玩家身边。"
assert TSUNDERE in character.CARD
CARD_B = character.CARD.replace(TSUNDERE, GENTLE)


async def contexts(n):
    out = []
    sp = specs("val", n // 2, memory="progressive") + specs("val", n // 2)
    sem = asyncio.Semaphore(48)

    async def one(s):
        async with sem:
            try:
                ep = Episode(s, llm.player_fn(POOL, "q35"))
                return await ep.start()
            except Exception:
                return None
    out = await asyncio.gather(*[one(s) for s in sp])
    return [o for o in out if o]


def main(ckpt, path, n=160):
    obs = asyncio.run(contexts(n))
    print("contexts", len(obs), flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
    cfg = model.config.text_config
    layers = model.model.language_model.layers
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    nh, hd = cfg.num_attention_heads, cfg.head_dim
    cap = {}

    def mk(L):
        def pre(mod, inp):
            cap[L] = inp[0][:, -1, :].float().cpu()
        return pre
    for L in full:
        layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L))
    sums = {}
    for name, card in (("A", character.CARD), ("B", CARD_B)):
        R = torch.zeros(len(layers), cfg.hidden_size)
        Rn = torch.zeros(len(layers))
        H = {L: torch.zeros(nh, hd) for L in full}
        Hn = {L: torch.zeros(nh) for L in full}
        for o in obs:
            msgs = [{"role": "system", "content": card}, {"role": "user", "content": o}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                          return_tensors="pt", return_dict=True).to("cuda")
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            for L in range(len(layers)):
                h = hs[L + 1][0, -1].float().cpu()
                R[L] += h
                Rn[L] += h.norm()
            for L in full:
                a = cap[L][0].view(nh, hd)
                H[L] += a
                Hn[L] += a.norm(dim=-1)
        k = len(obs)
        sums[name] = (R / k, Rn / k, {L: H[L] / k for L in full}, {L: Hn[L] / k for L in full})
    (RA, RnA, HA, HnA), (RB, RnB, HB, HnB) = sums["A"], sums["B"]
    resid = RB - RA
    heads = {L: HB[L] - HA[L] for L in full}
    rank = sorted(((L, h, float(heads[L][h].norm() / (0.5 * (HnA[L][h] + HnB[L][h])))) for L in full for h in range(nh)),
                  key=lambda x: -x[2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"resid": resid, "resid_norm": 0.5 * (RnA + RnB), "heads": heads, "head_rank": rank, "head_dim": hd,
                "n_contexts": len(obs), "ckpt": ckpt}, path)
    print("resid |v|/|h| by layer:", [round(float(resid[L].norm() / (0.5 * (RnA[L] + RnB[L]))), 3) for L in range(len(layers))])
    print("top heads:", [(L, h, round(s, 3)) for L, h, s in rank[:12]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 160)
