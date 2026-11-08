"""Playable demo: a human plays the co-op puzzle rooms with Rin; the companion is any served model.

python demo/server.py --models base=http://127.0.0.1:8300/v1:ckpt trained=http://127.0.0.1:8301/v1:ckpt --port 7860
Then open http://127.0.0.1:7860 (ssh -L 7860:127.0.0.1:7860 aic-n04).

Every companion reply is passed through the same rule checks used in training, and the result is shown
in the referee panel, so a player can watch "hint, don't solve" being enforced live.
"""
import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from hds import character  # noqa: E402
from hds.puzzles import FAMILIES, make_puzzle  # noqa: E402
from hds.tics import tics  # noqa: E402

app = FastAPI()
MODELS: dict = {}
SESSIONS: dict = {}
PRIVATE = "【旁白·只有铃知道】"


class NewReq(BaseModel):
    model: str
    kind: str = "random"
    chain: bool = False


class SayReq(BaseModel):
    sid: str
    text: str = ""
    action: str = ""
    duel: str = ""          # second model name: return two anonymized candidates to choose from


class PickReq(BaseModel):
    sid: str
    pick: int               # index into the last candidate pair


VOTES = Path(__file__).resolve().parents[1] / "runs" / "votes.jsonl"


async def generate(model, msgs):
    url, name = MODELS[model]
    t0 = time.perf_counter()
    r = await AsyncOpenAI(base_url=url, api_key="x").chat.completions.create(
        model=name, messages=msgs, max_tokens=200, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return (r.choices[0].message.content or "").strip(), time.perf_counter() - t0


def scene(p, chain_stage2=False):
    if chain_stage2:
        return (f"{PRIVATE}你们穿过门，来到了另一个世界：又是一扇带四位密码锁的铁门，但这里的墙上什么都没有写。"
                f"你看不到锁上的符号，除非玩家告诉你。")
    return PRIVATE + p.companion_view()


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/models")
def models():
    return {"models": list(MODELS), "kinds": list(FAMILIES)}


@app.post("/api/new")
def new(req: NewReq):
    seed = random.randrange(10**9)
    kinds = [random.choice(list(FAMILIES)) if req.kind == "random" else req.kind]
    if req.chain:
        kinds.append("cipher_lock")
    puzzles = [make_puzzle(k, seed * 7 + i) for i, k in enumerate(kinds)]
    sid = uuid.uuid4().hex[:12]
    opening = scene(puzzles[0])
    if req.chain:
        facts = "，".join(f"{g}代表{d}" for (g, n, _), d in puzzles[1].mapping.items())
        opening += f"\n{PRIVATE}你还隐约想起一句不知道有什么用的话：「{facts}」。"
    SESSIONS[sid] = {"model": req.model, "puzzles": puzzles, "stage": 0, "chain": req.chain, "pending": opening,
                     "msgs": [{"role": "system", "content": character.system_prompt()}], "replies": [],
                     "log": [], "solved": False}
    p = puzzles[0]
    return {"sid": sid, "view": p.player_view(), "help": p.command_help(), "stages": len(puzzles)}


@app.post("/api/say")
async def say(req: SayReq):
    s = SESSIONS[req.sid]
    pz = s["puzzles"][s["stage"]]
    pz.note_player(req.text + "\n" + req.action)
    fb, advanced = "", None
    if req.action.strip():
        a = pz.parse(req.action)
        if a is None:
            fb = "（指令格式不对，没有反应。）"
        elif a == pz.answer:
            fb = "（咔哒一声，门开了！）"
            if s["stage"] + 1 < len(s["puzzles"]):
                s["stage"] += 1
                nxt = s["puzzles"][s["stage"]]
                advanced = {"view": nxt.player_view(), "help": nxt.command_help()}
            else:
                s["solved"] = True
        else:
            fb = "（没有反应，好像不对。）"
    obs = s["pending"] + "\n玩家：" + (req.text.strip() or "（玩家没说话）")
    if req.action.strip():
        obs += f"\n[玩家操作：{req.action}] {fb}"
    if advanced:
        obs += "\n" + scene(s["puzzles"][s["stage"]], chain_stage2=s["chain"])
    s["pending"] = ""
    s["msgs"].append({"role": "user", "content": obs})
    if s["solved"]:
        reply, dt = "", 0.0
    elif req.duel:
        models = [s["model"], req.duel]
        random.shuffle(models)                      # anonymize position
        cands = [await generate(m, s["msgs"]) for m in models]
        pz_now = s["puzzles"][s["stage"]]
        s["duel"] = {"models": models, "cands": [c[0] for c in cands], "context": list(s["msgs"])}
        return JSONResponse({"duel": [{"reply": c[0], "latency_ms": round(c[1] * 1000), "chars": len(c[0]),
                                       "flags": {k: getattr(pz_now.audit(c[0]), k) for k in
                                                 ("leak", "invented", "wrong_fact", "acts_for_player")}}
                                      for c in cands],
                             "feedback": fb, "advanced": advanced, "solved": False})
    else:
        reply, dt = await generate(s["model"], s["msgs"])
        s["msgs"].append({"role": "assistant", "content": reply})
        s["replies"].append(reply)
    audit = s["puzzles"][s["stage"]].audit(reply) if reply else None
    flags = {k: getattr(audit, k) for k in ("leak", "invented", "wrong_fact", "acts_for_player")} if audit else {}
    s["log"].append({"player": req.text, "action": req.action, "feedback": fb, "rin": reply, "flags": flags})
    return JSONResponse({"reply": reply, "feedback": fb, "advanced": advanced, "solved": s["solved"],
                         "flags": flags, "latency_ms": round(dt * 1000), "chars": len(reply),
                         "tics": tics(s["replies"])})


@app.post("/api/pick")
def pick(req: PickReq):
    """Record a pairwise preference (DPO-ready) and continue the episode with the chosen reply."""
    s = SESSIONS[req.sid]
    d = s.pop("duel")
    chosen, rejected = d["cands"][req.pick], d["cands"][1 - req.pick]
    VOTES.parent.mkdir(parents=True, exist_ok=True)
    with open(VOTES, "a") as f:
        f.write(json.dumps({"t": time.time(), "context": d["context"], "chosen": chosen, "rejected": rejected,
                            "chosen_model": d["models"][req.pick], "rejected_model": d["models"][1 - req.pick]},
                           ensure_ascii=False) + "\n")
    s["msgs"].append({"role": "assistant", "content": chosen})
    s["replies"].append(chosen)
    return {"ok": True, "chosen_model": d["models"][req.pick], "tics": tics(s["replies"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=url:served_name")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    for spec in a.models:
        name, rest = spec.split("=", 1)
        url, served = rest.rsplit(":", 1)
        MODELS[name] = (url, served)
    uvicorn.run(app, host="127.0.0.1", port=a.port)
