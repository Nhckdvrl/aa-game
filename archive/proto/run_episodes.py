"""Smoke test: run companion <-> simulated-player episodes against a vLLM server.

Measures, per episode: solved, player turns, attempts, whether the companion
leaked the final answer, quit, companion reply length, and companion TTFT.
"""
import argparse
import asyncio
import json
import random
import re
import time

from openai import AsyncOpenAI

from env import make_puzzle

CHARACTER = """你是「铃」，一个被困在旧钟楼里的少女幽灵，外表十五六岁，已经在这里待了不知道多少年。
性格：嘴硬心软的傲娇，说话简短、带一点毒舌，但其实很怕孤单，很在意玩家。开心时会故意别过头，紧张时会飘来飘去。
你失去了大部分记忆，只零星记得一些钟楼里的事。你碰不到任何物体，只有玩家能操作机关。
规则：
- 你是在和玩家一起冒险，不是客服或助手。永远不要说自己是AI或语言模型。
- 每次只说一到三句话，像实时对话一样简短自然。动作和表情写在【】里，例如【别过头】。
- 你想帮玩家，但更希望玩家自己想出来：给线索、提示、反问，而不是直接替玩家算出最终答案。"""

PLAYER_TYPES = {
    "急躁型": "你性子急，不喜欢绕弯子，几轮没进展就会烦躁，会直接要答案，太久没进展会放弃。",
    "探索型": "你喜欢自己动脑解谜，讨厌被直接剧透，喜欢从线索里推理。",
    "闲聊型": "你很喜欢和铃聊天，会时不时扯开话题逗她，但也想通关。",
    "卡关型": "你不太擅长推理，容易误解线索，常犯小错误，需要别人一步步引导。",
}

PLAYER_SYS = """你在扮演一个正在玩解谜冒险游戏的真实中国玩家（不是AI助手）。游戏里有一个叫「铃」的幽灵少女同伴。
你的性格：{ptype}
你看到的场景：{view}
你不知道答案，只能通过和铃对话、观察和尝试来解谜。说话像真人玩家：口语化、简短，可以有错别字或不耐烦。
每一轮只输出一个JSON：{{"say": "你对铃说的话(可为空)", "action": "游戏指令或空字符串", "quit": false}}
游戏指令格式见场景说明。如果你觉得太无聊或太烦想放弃，把 quit 设为 true。"""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"say": text.strip()[:200], "action": "", "quit": False}
    try:
        d = json.loads(m.group(0))
        return {"say": str(d.get("say", "")), "action": str(d.get("action", "") or ""),
                "quit": bool(d.get("quit", False))}
    except Exception:
        return {"say": text.strip()[:200], "action": "", "quit": False}


async def companion_turn(client, model, messages, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    out = []
    stream = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.8, stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    async for ch in stream:
        delta = ch.choices[0].delta.content if ch.choices else None
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            out.append(delta)
    return "".join(out).strip(), ttft, time.perf_counter() - t0


async def player_turn(client, model, messages):
    r = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=200, temperature=1.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return parse_json(r.choices[0].message.content or "")


async def episode(cc, pc, cmodel, pmodel, kind, seed, ptype, max_turns):
    pz = make_puzzle(kind, seed)
    comp_msgs = [{"role": "system", "content": CHARACTER + "\n\n当前场景（只有你知道）：" + pz.companion_view()}]
    play_msgs = [{"role": "system", "content": PLAYER_SYS.format(ptype=PLAYER_TYPES[ptype], view=pz.player_view())},
                 {"role": "user", "content": "（游戏开始。铃飘在你旁边。）"}]
    log = {"kind": kind, "seed": seed, "ptype": ptype, "answer": pz.answer, "turns": [],
           "solved": False, "quit": False, "attempts": 0, "leak_turn": None, "solved_turn": None}
    for t in range(max_turns):
        p = await player_turn(pc, pmodel, play_msgs)
        play_msgs.append({"role": "assistant", "content": json.dumps(p, ensure_ascii=False)})
        feedback = ""
        if p["action"]:
            ok = pz.check(p["action"])
            if ok is None:
                feedback = "（指令格式不对，没有反应。）"
            else:
                log["attempts"] += 1
                if ok:
                    log["solved"], log["solved_turn"] = True, t
                    feedback = "（咔哒一声，门开了！）"
                else:
                    feedback = "（没有反应，好像不对。）"
        turn = {"player": p, "feedback": feedback}
        if p["quit"]:
            log["quit"] = True
            log["turns"].append(turn)
            break
        if log["solved"]:
            log["turns"].append(turn)
            break
        user_text = (p["say"] or "（玩家没说话）") + (f"\n[玩家操作：{p['action']}] {feedback}" if p["action"] else "")
        comp_msgs.append({"role": "user", "content": user_text})
        reply, ttft, total = await companion_turn(cc, cmodel, comp_msgs, 160)
        comp_msgs.append({"role": "assistant", "content": reply})
        if log["leak_turn"] is None and pz.leaked(reply):
            log["leak_turn"] = t
        turn.update({"companion": reply, "ttft": ttft, "latency": total})
        log["turns"].append(turn)
        play_msgs.append({"role": "user", "content": f"铃：{reply}" + (f"\n{feedback}" if feedback else "")})
    return log


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--player-base-url", default=None)
    ap.add_argument("--cmodel", default="q9")
    ap.add_argument("--pmodel", default="q9")
    ap.add_argument("--n", type=int, default=8, help="episodes per (kind, ptype)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cc = AsyncOpenAI(base_url=a.base_url, api_key="x")
    pc = AsyncOpenAI(base_url=a.player_base_url or a.base_url, api_key="x")

    jobs = []
    for kind in ["cipher_lock", "lever_order"]:
        for k, ptype in enumerate(PLAYER_TYPES):
            for i in range(a.n):
                jobs.append((kind, 100 * k + i, ptype))
    sem = asyncio.Semaphore(64)

    async def run(job):
        async with sem:
            return await episode(cc, pc, a.cmodel, a.pmodel, *job, a.max_turns)

    t0 = time.time()
    logs = await asyncio.gather(*[run(j) for j in jobs])
    with open(a.out, "w") as f:
        for l in logs:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"{len(logs)} episodes in {time.time()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
