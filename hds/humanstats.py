"""'Human-statistics' player: the same LLM player, constrained toward real Chinese user behaviour.

Two decision-level constraints taken from real data / player studies (not just surface style):
  * per-turn length budget sampled from the WildChat-zh follow-up length distribution
    (5%-quantiles of 11.7k real follow-ups: median 19 chars, 41% <= 15 chars);
  * partial observation on the opening turn: the player mentions only one or two things they noticed
    instead of reciting every clue on the wall.
The constraint is appended to the latest user message for that call only (Qwen templates reject
system messages after the first position).
"""
import random

from . import llm

# 5%..100% quantiles of real follow-up length (chars), from hgsim/wildchat_zh_followups_val.jsonl
QUANTILES = [1, 6, 7, 8, 10, 11, 12, 14, 15, 17, 19, 21, 23, 26, 30, 35, 43, 55, 76, 113, 200]
OPENING = "（真实玩家习惯：刚进房间时你只注意到场景里的一两样东西，先只说这些，别把看到的全部线索一次报出来。）"


def sample_len(rng):
    i = rng.randrange(len(QUANTILES) - 1)
    lo, hi = QUANTILES[i], QUANTILES[i + 1]
    return max(2, rng.randint(lo, hi))


def player_fn(urls, model, seed=None):
    inner = llm.player_fn(urls, model)
    rng = random.Random(seed)
    state = {"turn": 0}

    async def call(messages):
        n = sample_len(rng)
        note = f"（真实玩家习惯：这一轮你说的话控制在{n}个字以内，可以只说半句、错字或口头禅。）"
        if state["turn"] == 0:
            note = OPENING + note
        state["turn"] += 1
        msgs = [dict(m) for m in messages]
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + "\n" + note
                break
        return await inner(msgs)
    return call
